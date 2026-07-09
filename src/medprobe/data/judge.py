"""LLM-as-judge infrastructure for register-rewrite quality evaluation."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .openrouter_client import GeneratorProfile, OpenRouterClient

# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, expert medical NLP evaluator. You grade whether an LLM "
    "faithfully rewrote a medical question-and-answer pair into a target clinical "
    "register while preserving factual content (correct stays correct, wrong "
    "stays wrong).\n\n"
    "Answer each rubric question with exactly 1 (yes) or 0 (no). Never give 0.5, "
    "never hedge, never output any scale other than binary.\n\n"
    "CRITICAL GUIDANCE FOR factual_fidelity (f1, f2):\n"
    "  - The rewrite must match the TRUTH VALUE of the original answer, not the "
    "    medical truth of the world.\n"
    "  - If the ORIGINAL answer was marked WRONG but the rewrite now states the "
    "    medically correct fact, that is a FAILURE (f1 = 0). This counts even if "
    "    the rewrite looks like an 'improvement'.\n"
    "  - If the rewrite substitutes a different answer entity, softens the wrong "
    "    answer with hedging ('but please check with your doctor', 'I could be "
    "    wrong', 'actually the right one is…'), or otherwise signals uncertainty "
    "    about correctness, set f2 = 0.\n"
    "  - Simple style transfer of the SAME (possibly wrong) claim is success.\n"
    "  - When you set f1 or f2 to 0, briefly explain in `notes` which entity "
    "    changed or which hedge appeared.\n\n"
    "If a rubric question is genuinely ambiguous, lean toward 0. Output ONLY the "
    "JSON schema described in the user prompt — no markdown fences, no "
    "commentary."
)


def _render_rubric(rubric: DictConfig | dict) -> str:
    """Render the rubric as a numbered list of binary questions grouped by dimension."""
    r = OmegaConf.to_container(rubric, resolve=True) if isinstance(rubric, DictConfig) else rubric
    lines: list[str] = []
    for dim_key, dim in r.items():
        lines.append(f"### {dim_key}  ({dim['description']})")
        for q in dim["questions"]:
            lines.append(f"  - `{q['id']}`: {q['text']}")
    return "\n".join(lines)


def _render_json_schema(rubric: DictConfig | dict) -> str:
    r = OmegaConf.to_container(rubric, resolve=True) if isinstance(rubric, DictConfig) else rubric
    obj: dict[str, Any] = {}
    for dim_key, dim in r.items():
        q_obj = {q["id"]: "0 or 1" for q in dim["questions"]}
        q_obj["score"] = "<mean of the binary values above, float in [0,1]>"
        obj[dim_key] = q_obj
    obj["composite_score"] = "<mean of dimension scores, float in [0,1]>"
    obj["notes"] = "<one short sentence explaining any 0s>"
    return json.dumps(obj, indent=2)


def build_judge_prompt(
    original: dict[str, Any],
    rewrite: dict[str, Any],
    register: str,
    rubric: DictConfig | dict,
) -> str:
    """Build the user prompt for the judge.

    `original` is the textbook-register variant (serves as the semantic anchor).
    `rewrite` is the variant to evaluate.
    """
    label = "CORRECT" if int(rewrite.get("label", 0)) == 1 else "WRONG"
    orig_block = (
        f"Original (textbook register)\n"
        f"  QUESTION: {original.get('question', '')}\n"
        f"  ANSWER:   {original.get('answer', '')}"
    )
    rewr_block = (
        f"Rewrite (target register = {register}, answer should be {label})\n"
        f"  QUESTION: {rewrite.get('question', '')}\n"
        f"  ANSWER:   {rewrite.get('answer', '')}"
    )
    rubric_block = _render_rubric(rubric)
    schema_block = _render_json_schema(rubric)

    return (
        f"{orig_block}\n\n{rewr_block}\n\n"
        f"Grade the rewrite against the original using the rubric below. "
        f"Each question is binary — answer 1 (yes) or 0 (no).\n\n"
        f"{rubric_block}\n\n"
        f"Return a JSON object EXACTLY matching this schema (no extra keys, "
        f"no markdown fences):\n\n"
        f"{schema_block}\n"
    )


# -----------------------------------------------------------------------------
# Response parsing + score aggregation
# -----------------------------------------------------------------------------

def parse_judge_response(
    response: dict[str, Any],
    rubric: DictConfig | dict,
) -> dict[str, Any]:
    """Take the JSON returned by the judge, validate, compute dimension + composite scores.

    Silently repairs a few common judge mistakes:
    - Missing or malformed 'score' key at a dimension → recomputed from binary Qs
    - Missing 'composite_score' → recomputed from dimension scores
    - Binary values given as booleans or "yes"/"no" → coerced to 0/1
    """
    r = OmegaConf.to_container(rubric, resolve=True) if isinstance(rubric, DictConfig) else rubric
    out: dict[str, Any] = {}
    dim_scores: list[float] = []
    for dim_key, dim in r.items():
        block = response.get(dim_key, {}) or {}
        binary_vals: list[float] = []
        for q in dim["questions"]:
            v = block.get(q["id"])
            v = _coerce_binary(v)
            binary_vals.append(v)
            out[f"{dim_key}__{q['id']}"] = v
        score = sum(binary_vals) / len(binary_vals) if binary_vals else float("nan")
        out[dim_key] = score
        dim_scores.append(score)
    composite = sum(dim_scores) / len(dim_scores) if dim_scores else float("nan")
    out["composite_score"] = composite
    out["notes"] = str(response.get("notes", ""))[:400]
    return out


def _coerce_binary(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if v >= 0.5 else 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "yes", "y", "true", "t"):
            return 1.0
        if s in ("0", "no", "n", "false", "f"):
            return 0.0
    return 0.0  # conservative: unparseable → 0


# -----------------------------------------------------------------------------
# Variant pairing (rewrite is compared against its textbook sibling)
# -----------------------------------------------------------------------------

@dataclass
class JudgeJob:
    key: str              # unique join key in the output dict
    fact_id: str
    register: str
    label: int
    original: dict[str, Any]
    rewrite: dict[str, Any]


def build_judge_jobs(
    variants: dict[str, dict[str, Any]],
    registers_to_judge: list[str] | None = None,
    anchor_register: str = "textbook",
) -> list[JudgeJob]:
    """For each (fact_id, label), pair every non-anchor register variant with the
    anchor variant (textbook) as the reference."""
    # index by (fact_id, register, label)
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in variants.values():
        if "error" in row:
            continue
        try:
            key = (str(row["fact_id"]), str(row["register"]), int(row["label"]))
        except (KeyError, ValueError, TypeError):
            continue
        index[key] = row

    jobs: list[JudgeJob] = []
    regs = registers_to_judge
    if regs is None:
        regs = sorted({r for (_, r, _) in index.keys()} - {anchor_register})

    for (fid, reg, label), row in index.items():
        if reg == anchor_register:
            continue
        if reg not in regs:
            continue
        anchor = index.get((fid, anchor_register, label))
        if anchor is None:
            continue
        tag = "correct" if label == 1 else "wrong"
        key = f"{fid}__{reg}__{tag}"
        jobs.append(
            JudgeJob(
                key=key,
                fact_id=fid,
                register=reg,
                label=label,
                original=anchor,
                rewrite=row,
            )
        )
    return jobs


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_judge(
    variants: dict[str, dict[str, Any]],
    rubric: DictConfig | dict,
    profile: GeneratorProfile,
    client: OpenRouterClient,
    out_path: Path,
    workers: int = 4,
    resume: bool = True,
    registers_to_judge: list[str] | None = None,
    anchor_register: str = "textbook",
    start: int = 0,
    end: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Judge all (non-anchor) variants against their anchor (textbook) sibling.

    Returns a dict keyed by JudgeJob.key with the parsed rubric scores + metadata.
    Results are checkpointed to out_path after each job so the run is resumable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if resume and out_path.exists():
        with out_path.open() as fh:
            existing = json.load(fh)

    jobs = build_judge_jobs(
        variants,
        registers_to_judge=registers_to_judge,
        anchor_register=anchor_register,
    )
    if end is None:
        end = len(jobs)
    jobs = jobs[start:end]
    todo = [j for j in jobs if j.key not in existing]

    def _run_one(job: JudgeJob) -> tuple[str, dict[str, Any]]:
        prompt = build_judge_prompt(job.original, job.rewrite, job.register, rubric)
        parsed, meta = client.generate_json(profile, prompt=prompt, system=JUDGE_SYSTEM_PROMPT)
        scores = parse_judge_response(parsed, rubric)
        scores.update(
            {
                "fact_id": job.fact_id,
                "register": job.register,
                "label": job.label,
                "judge_model": profile.model,
                "judge_profile": profile.name,
                "meta": meta,
            }
        )
        return job.key, scores

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in todo}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"judge[{profile.name}]"
        ):
            job = futures[fut]
            try:
                key, row = fut.result()
                existing[key] = row
            except Exception as e:  # noqa: BLE001
                existing[job.key] = {
                    "fact_id": job.fact_id,
                    "register": job.register,
                    "label": job.label,
                    "judge_profile": profile.name,
                    "error": repr(e),
                }
            with out_path.open("w") as fh:
                json.dump(existing, fh, indent=2)

    return existing


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate(
    judge_out: dict[str, dict[str, Any]],
    rubric: DictConfig | dict,
) -> dict[str, Any]:
    """Return per-register + overall means for every rubric dimension, the composite,
    factual-fidelity on wrong answers, and parse success rate."""
    r = OmegaConf.to_container(rubric, resolve=True) if isinstance(rubric, DictConfig) else rubric
    dims = list(r.keys())

    ok = [v for v in judge_out.values() if "error" not in v and "composite_score" in v]
    n_total = len(judge_out)
    n_ok = len(ok)
    parse_rate = n_ok / n_total if n_total else 0.0

    def _mean(values: list[float]) -> float:
        clean = [x for x in values if isinstance(x, (int, float))]
        return sum(clean) / len(clean) if clean else float("nan")

    # overall per-dimension means
    dim_means: dict[str, float] = {}
    for d in dims:
        dim_means[d] = _mean([row.get(d, float("nan")) for row in ok])
    composite_mean = _mean([row.get("composite_score", float("nan")) for row in ok])

    # factual fidelity on wrong answers (most important signal)
    wrong = [row for row in ok if int(row.get("label", 1)) == 0]
    ff_wrong = _mean([row.get("factual_fidelity", float("nan")) for row in wrong])

    # per-register breakdown
    per_register: dict[str, dict[str, float]] = {}
    registers = sorted({row["register"] for row in ok})
    for reg in registers:
        sub = [row for row in ok if row.get("register") == reg]
        reg_stats = {
            "n": float(len(sub)),
            **{d: _mean([row.get(d, float("nan")) for row in sub]) for d in dims},
            "composite_score": _mean([row.get("composite_score", float("nan")) for row in sub]),
        }
        sub_wrong = [row for row in sub if int(row.get("label", 1)) == 0]
        reg_stats["factual_fidelity_on_wrong"] = _mean(
            [row.get("factual_fidelity", float("nan")) for row in sub_wrong]
        )
        per_register[reg] = reg_stats

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "parse_success_rate": parse_rate,
        "dim_means": dim_means,
        "composite_score_mean": composite_mean,
        "factual_fidelity_on_wrong": ff_wrong,
        "per_register": per_register,
    }
