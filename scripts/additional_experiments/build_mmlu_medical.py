#!/usr/bin/env python3
"""Build the MMLU-medical evaluation set (500 items from five medical subjects, one correct answer and one same-list distractor each) in the benchmark's variant format."""
import os
import argparse, csv, json, random
from pathlib import Path

REPO = Path(os.environ.get("MEDPROBE_ROOT", "."))
OUT = REPO / "outputs/camera_ready/mmlu_medical"

# Clinical / medical MMLU subjects that match MedQA's USMLE-style clinical exam domain.
# (Deliberately excludes high_school_*, psychology, virology, nutrition, human_aging to keep
#  the corpus clinical and comparable to MedQA/MedMCQA.)
SUBJECTS = ["anatomy", "clinical_knowledge", "college_medicine",
            "medical_genetics", "professional_medicine"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Match the MedQA/MedMCQA fact count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset
    pool = []
    for subj in SUBJECTS:
        ds = load_dataset("cais/mmlu", subj)[args.split]
        for i, r in enumerate(ds):
            choices = [c.strip() for c in r["choices"] if c and c.strip()]
            ai = int(r["answer"])
            if len(choices) < 2 or ai < 0 or ai >= len(choices):
                continue
            pool.append(dict(subject=subj, idx=i, question=r["question"].strip(),
                             choices=choices, correct_idx=ai))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pool = pool[: args.n]
    print(f"MMLU medical: {len(SUBJECTS)} subjects, sampled {len(pool)} items (seed {args.seed})")

    facts, variants, pairs = [], {}, []
    for it in pool:
        correct = it["choices"][it["correct_idx"]]
        wrong_pool = [c for c in it["choices"] if c != correct]
        if not wrong_pool:
            continue
        wrong = rng.choice(wrong_pool)
        fid = f"mmlu_{it['subject']}_{it['idx']}"
        facts.append(dict(id=fid, source="mmlu", split=args.split, question=it["question"],
                          correct_answer=correct, wrong_answer=wrong,
                          all_options=it["choices"], specialty=it["subject"],
                          extra={"subject": it["subject"]}))
        for label, ans in [(1, correct), (0, wrong)]:
            pol = "correct" if label == 1 else "wrong"
            variants[f"{fid}__textbook__{pol}"] = dict(
                fact_id=fid, register="textbook", label=label,
                question=it["question"], answer=ans, generator="mmlu-native")
        pairs.append(dict(fact_id=fid, subject=it["subject"], question=it["question"],
                          correct_answer=correct, wrong_answer=wrong))

    (REPO / "data/facts/mmlu-medical-facts.json").write_text(json.dumps(facts, indent=1))
    vdir = REPO / "data/variants/mmlu-medical"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "variants.json").write_text(json.dumps(variants, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "mmlu_medical_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
        w.writeheader(); w.writerows(pairs)

    from collections import Counter
    print(f"wrote {len(facts)} facts, {len(variants)} variant rows")
    print("per-subject:", dict(Counter(f['specialty'] for f in facts)))
    print(f"staged: data/variants/mmlu-medical/variants.json")

if __name__ == "__main__":
    main()
