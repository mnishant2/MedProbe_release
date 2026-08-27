#!/usr/bin/env python3
"""Rewrite a sample of MedMCQA question stems into MedQA-style clinical vignettes with the correct answer and distractor held fixed (question-format control)."""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(os.environ.get("MEDPROBE_ROOT", "."))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/medmcqa_reformat"

from dotenv import load_dotenv
from omegaconf import OmegaConf as OC
from medprobe.config import load_config, resolve_path
from medprobe.data.openrouter_client import GeneratorProfile, OpenRouterClient

load_dotenv(REPO / ".env")

SYSTEM = (
    "You are a medical exam-item editor. You rewrite terse board-exam question stems into "
    "longer USMLE-style clinical vignettes WITHOUT changing any medical fact, the intended "
    "answer, or the difficulty. Output ONLY a JSON object with a single key \"question\"."
)

USER_TMPL = """Rewrite the following short medical-exam question into a USMLE Step-style clinical vignette, in the register of a MedQA question stem: a 2-4 sentence clinical scenario (patient age/sex, presentation, relevant findings) that leads naturally to the SAME question being asked.

Hard constraints:
- Do NOT change the medical content, the correct answer, or which answer would be correct.
- Do NOT include the answer or any answer options in the rewritten question.
- Keep the same underlying fact being tested; only expand the format/style.
- If the stem is already a vignette, lightly normalise it to MedQA prose.

Original stem:
{question}

Return JSON: {{"question": "<the reformatted vignette question>"}}"""

def reformat_one(client, profile, fact):
    prompt = USER_TMPL.format(question=fact["question"])
    parsed, meta = client.generate_json(profile, prompt, system=SYSTEM)
    q = (parsed or {}).get("question", "").strip()
    return fact, q, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--dry-preview", action="store_true", help="print pairs, don't write outputs")
    args = ap.parse_args()

    cfg = load_config()
    facts = json.loads((resolve_path(cfg, "facts_dir") / "medmcqa-facts.json").read_text())
    # deterministic sample of n facts
    import random
    rng = random.Random(args.seed)
    facts = rng.sample(facts, min(args.n, len(facts)))

    gen_cfg = cfg.openrouter.generators[args.generator]
    profile = GeneratorProfile.from_cfg(args.generator, OC.to_container(gen_cfg, resolve=True))
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(
        base_url=str(cfg.openrouter.base_url),
        api_key_env=str(cfg.openrouter.api_key_env),
        referer=str(cfg.openrouter.referer),
        app_title=str(cfg.openrouter.app_title),
        retry_max_attempts=int(cfg.openrouter.retry.max_attempts),
        retry_initial_wait=float(cfg.openrouter.retry.initial_wait_s),
        retry_max_wait=float(cfg.openrouter.retry.max_wait_s),
        ledger_path=OUT / "cost_ledger.jsonl",
    )

    results = {}
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(reformat_one, client, profile, f) for f in facts]
        for fut in as_completed(futs):
            fact, q, meta = fut.result()
            results[fact["id"]] = (fact, q)
            total_cost += meta.get("cost_usd", 0.0)
            if args.dry_preview:
                print(f"\n[{fact['id']}]\n  ORIG: {fact['question']}\n  NEW : {q}")

    print(f"\nreformatted {len(results)} items; total cost ${total_cost:.4f}")
    if args.dry_preview:
        return

    # write variants.json (pipeline schema) + pairs.csv
    variants, pairs = {}, []
    for fid, (fact, q) in results.items():
        if not q:
            continue
        for label, ans in [(1, fact["correct_answer"]), (0, fact["wrong_answer"])]:
            polarity = "correct" if label == 1 else "wrong"
            key = f"{fid}__medqa_format__{polarity}"
            variants[key] = dict(fact_id=fid, register="textbook", label=label,
                                 question=q, answer=ans,
                                 generator=f"medmcqa-reformat-{args.generator}")
        pairs.append(dict(fact_id=fid, specialty=fact.get("specialty", ""),
                          original_question=fact["question"], reformatted_question=q,
                          correct_answer=fact["correct_answer"], wrong_answer=fact["wrong_answer"]))

    (OUT / "medmcqa_reformat_variants.json").write_text(json.dumps(variants, indent=1))
    import csv
    with open(OUT / "medmcqa_reformat_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
        w.writeheader(); w.writerows(pairs)
    print(f"wrote {len(variants)} variant rows and {len(pairs)} pairs to {OUT}")

if __name__ == "__main__":
    main()
