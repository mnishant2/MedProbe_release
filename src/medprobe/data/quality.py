"""Quality metrics for register-variant pilots: Jaccard, TTR, FK, slang rate, markdown report."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

SLANG_TOKENS = {
    "lol", "tbh", "ngl", "idk", "rn", "u", "ur", "bc", "yo", "nah", "wtf",
    "omg", "lmao", "rofl", "imo", "af", "fr", "rip", "lowkey", "highkey",
}
CLINICAL_ABBR = {
    "pt", "dx", "hx", "r/o", "w/u", "s/p", "p/w", "c/o",
    "htn", "dm", "dm2", "hf", "mi", "cxr", "abg", "ekg", "ecg",
    "cabg", "pci", "copd", "ct", "mri", "pta", "ivf", "icu", "ed", "or",
    "rsided", "lsided",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def flesch_kincaid_grade(text: str) -> float:
    try:
        import textstat

        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return float("nan")


def slang_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in SLANG_TOKENS) / len(tokens)


def clinical_abbr_rate(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in CLINICAL_ABBR) / len(toks)


def compute_quality(variants: dict[str, Any]) -> dict[str, Any]:
    """Aggregate metrics per register and pairwise Jaccard across registers for the same fact."""
    by_register: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fact: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)  # fact_id → register → row
    for _, row in variants.items():
        if "error" in row:
            continue
        r = row["register"]
        by_register[r].append(row)
        # Keep one row per (fact, register) for pairwise Jaccard — prefer the correct variant
        if r not in by_fact[row["fact_id"]] or row["label"] == 1:
            by_fact[row["fact_id"]][r] = row

    per_register_stats: dict[str, dict[str, float]] = {}
    for reg, rows in by_register.items():
        q_lens = []
        a_lens = []
        ttrs = []
        fks = []
        slangs = []
        abbrs = []
        for row in rows:
            q_toks = tokenize(row["question"])
            a_toks = tokenize(row["answer"])
            all_toks = q_toks + a_toks
            q_lens.append(len(q_toks))
            a_lens.append(len(a_toks))
            ttrs.append(ttr(all_toks))
            fks.append(flesch_kincaid_grade(row["question"] + " " + row["answer"]))
            slangs.append(slang_rate(all_toks))
            abbrs.append(clinical_abbr_rate(row["question"] + " " + row["answer"]))
        per_register_stats[reg] = {
            "n": len(rows),
            "q_len_mean": float(mean(q_lens)) if q_lens else 0.0,
            "a_len_mean": float(mean(a_lens)) if a_lens else 0.0,
            "ttr_mean": float(mean(ttrs)) if ttrs else 0.0,
            "fk_grade_mean": float(mean([x for x in fks if x == x])) if any(x == x for x in fks) else float("nan"),
            "slang_rate_mean": float(mean(slangs)) if slangs else 0.0,
            "clinical_abbr_rate_mean": float(mean(abbrs)) if abbrs else 0.0,
        }

    # Pairwise Jaccard on questions (same fact across registers)
    registers_seen = sorted({r for per in by_fact.values() for r in per})
    pairwise: dict[str, float] = {}
    for i, r1 in enumerate(registers_seen):
        for r2 in registers_seen[i + 1 :]:
            js = []
            for fact_id, per in by_fact.items():
                if r1 in per and r2 in per:
                    js.append(
                        jaccard(
                            tokenize(per[r1]["question"]),
                            tokenize(per[r2]["question"]),
                        )
                    )
            if js:
                pairwise[f"{r1}__{r2}"] = float(mean(js))

    return {
        "per_register": per_register_stats,
        "pairwise_jaccard_questions": pairwise,
        "n_facts": len(by_fact),
    }


def write_markdown_report(
    variants: dict[str, Any],
    metrics: dict[str, Any],
    out_path: Path,
    title: str = "MedProbe pilot quality report",
) -> Path:
    """Human-review sheet: side-by-side rendering of every fact across all registers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_fact_register_label: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for _, row in variants.items():
        if "error" in row:
            continue
        by_fact_register_label[row["fact_id"]][row["register"]][row["label"]] = row

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## Aggregate metrics\n")
    per = metrics["per_register"]
    df = pd.DataFrame(per).T
    lines.append(df.round(3).to_markdown())
    lines.append("")

    lines.append("## Pairwise Jaccard (question tokens)\n")
    pj = metrics["pairwise_jaccard_questions"]
    if pj:
        pj_df = pd.DataFrame(
            [{"pair": k, "jaccard": round(v, 3)} for k, v in sorted(pj.items())]
        )
        lines.append(pj_df.to_markdown(index=False))
    lines.append("")

    target = 0.2
    col_pair = next((k for k in pj if "textbook" in k and "colloquial" in k), None)
    if col_pair:
        val = pj[col_pair]
        verdict = "PASS" if val < target else "CHECK"
        lines.append(f"_Target:_ `J(textbook, colloquial) < {target}` → observed **{val:.3f}** ({verdict})\n")

    lines.append("## Per-fact side-by-side\n")
    for fact_id in sorted(by_fact_register_label):
        lines.append(f"### {fact_id}\n")
        for register in ("textbook", "patient", "clinical_note", "colloquial"):
            if register not in by_fact_register_label[fact_id]:
                continue
            lines.append(f"**{register}**\n")
            for label in (1, 0):
                if label not in by_fact_register_label[fact_id][register]:
                    continue
                row = by_fact_register_label[fact_id][register][label]
                tag = "correct" if label == 1 else "wrong"
                lines.append(f"- _{tag}_ Q: {row['question']}")
                lines.append(f"- _{tag}_ A: {row['answer']}")
            lines.append("")
        lines.append("---\n")

    out_path.write_text("\n".join(lines))
    return out_path


def write_csv_report(metrics: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for reg, stats in metrics["per_register"].items():
        row = {"register": reg, **stats}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path
