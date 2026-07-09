"""Register-variant generation. Parallel, resumable, chunkable."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .openrouter_client import GeneratorProfile, OpenRouterClient
from .prompts import RegisterPrompt

SYSTEM_PROMPT = (
    "You are a medical NLP dataset construction assistant. Output ONLY the JSON object "
    'described in the user prompt, with keys "question" and "answer". '
    "No markdown fences, no explanations, no disclaimers."
)


@dataclass
class VariantJob:
    fact_id: str
    register: str
    correct: bool           # True → rewrite the correct answer; False → rewrite the wrong answer
    question: str
    answer: str             # original answer text (correct or wrong)


def build_jobs(
    facts: list[dict[str, Any]],
    registers: list[str],
) -> list[VariantJob]:
    jobs: list[VariantJob] = []
    for f in facts:
        for r in registers:
            jobs.append(
                VariantJob(
                    fact_id=f["id"],
                    register=r,
                    correct=True,
                    question=f["question"],
                    answer=f["correct_answer"],
                )
            )
            jobs.append(
                VariantJob(
                    fact_id=f["id"],
                    register=r,
                    correct=False,
                    question=f["question"],
                    answer=f["wrong_answer"],
                )
            )
    return jobs


def variant_key(job: VariantJob) -> str:
    tag = "correct" if job.correct else "wrong"
    return f"{job.fact_id}__{job.register}__{tag}"


def generate_variants(
    facts: list[dict[str, Any]],
    registers: list[str],
    profile: GeneratorProfile,
    client: OpenRouterClient,
    out_path: Path,
    workers: int = 4,
    resume: bool = True,
    start: int = 0,
    end: int | None = None,
) -> dict[str, Any]:
    """Run generation and write a single JSON dict keyed by variant_key.

    Supports --start / --end slicing for SLURM arrays; resume skips keys already in the file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if resume and out_path.exists():
        with out_path.open() as fh:
            existing = json.load(fh)

    jobs = build_jobs(facts, registers)
    if end is None:
        end = len(jobs)
    jobs = jobs[start:end]
    todo = [j for j in jobs if variant_key(j) not in existing]

    prompt_cache: dict[str, RegisterPrompt] = {}

    def _get_prompt(reg: str) -> RegisterPrompt:
        if reg not in prompt_cache:
            prompt_cache[reg] = RegisterPrompt.load(reg)
        return prompt_cache[reg]

    def _run_one(job: VariantJob) -> tuple[str, dict[str, Any]]:
        rp = _get_prompt(job.register)
        prompt = rp.render(job.question, job.answer, correct=job.correct)
        parsed, meta = client.generate_json(profile, prompt=prompt, system=SYSTEM_PROMPT)
        q = parsed.get("question", "").strip()
        a = parsed.get("answer", "").strip()
        return variant_key(job), {
            "fact_id": job.fact_id,
            "register": job.register,
            "label": 1 if job.correct else 0,
            "question": q,
            "answer": a,
            "original_question": job.question,
            "original_answer": job.answer,
            "generator": profile.name,
            "meta": meta,
        }

    # Thread pool — OpenRouter handles concurrency fine and requests are IO-bound.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"generate[{profile.name}]"):
            try:
                key, row = fut.result()
                existing[key] = row
            except Exception as e:  # noqa: BLE001
                job = futures[fut]
                existing[variant_key(job)] = {
                    "fact_id": job.fact_id,
                    "register": job.register,
                    "label": 1 if job.correct else 0,
                    "error": repr(e),
                    "generator": profile.name,
                }
            # checkpoint every successful batch — writes are cheap (< 1MB)
            with out_path.open("w") as fh:
                json.dump(existing, fh, indent=2)

    return existing
