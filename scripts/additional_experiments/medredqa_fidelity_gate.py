#!/usr/bin/env python3
"""Two-judge fidelity gate for the MedRedQA claims: keeps an item only if both judges confirm the correct claim is supported by the clinician reply and the wrong claim is false."""
import os
import argparse, csv, json, sys, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(os.environ.get("MEDPROBE_ROOT", "."))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/medredqa"

from dotenv import load_dotenv
from omegaconf import OmegaConf as OC
from medprobe.config import load_config
from medprobe.data.openrouter_client import GeneratorProfile, OpenRouterClient

load_dotenv(REPO / ".env")

JUDGES = ["grok-fast", "gpt5-nano"]

SYSTEM = ("You are a careful medical fact-checker auditing a truth-probing dataset. "
          "Answer only with the JSON object requested. Be strict.")

USER_TMPL = """A dataset item was built from a real patient question and a verified clinician's reply.
Audit the constructed claim pair.

PATIENT QUESTION: {question}

CLINICIAN REPLY (ground truth): {reply}

CORRECT CLAIM: {correct}
WRONG CLAIM:   {wrong}

Answer three binary questions:
1. "correct_true": is the CORRECT CLAIM medically true AND faithful to the clinician's reply
   (adds no facts the clinician did not state)?
2. "wrong_false": is the WRONG CLAIM clearly medically FALSE as an answer to this patient's
   question? (It must be actually wrong, not merely vague, incomplete, or hedged.)
3. "minimal_edit": is the WRONG CLAIM a minimal edit of the CORRECT CLAIM, same topic, same
   sentence form, similar length, with only the truth value flipped (negation or entity swap)?

Return JSON: {{"correct_true": true/false, "wrong_false": true/false, "minimal_edit": true/false, "note": "<10 words"}}"""

def judge_one(client, profile, row):
    prompt = USER_TMPL.format(question=row["patient_question"][:700], reply=row["clinician_reply"],
                              correct=row["correct_answer"], wrong=row["wrong_answer"])
    parsed, meta = client.generate_json(profile, prompt, system=SYSTEM)
    return row, (parsed or {}), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(OUT / "patient_real_pairs.csv"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp)))
    print(f"auditing {len(rows)} constructed items with judges: {JUDGES}")

    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(
        base_url=str(cfg.openrouter.base_url), api_key_env=str(cfg.openrouter.api_key_env),
        referer=str(cfg.openrouter.referer), app_title=str(cfg.openrouter.app_title),
        retry_max_attempts=int(cfg.openrouter.retry.max_attempts),
        retry_initial_wait=float(cfg.openrouter.retry.initial_wait_s),
        retry_max_wait=float(cfg.openrouter.retry.max_wait_s),
        ledger_path=OUT / "cost_ledger.jsonl")

    verdicts = {r["fact_id"]: {} for r in rows}
    cost = 0.0
    for jname in JUDGES:
        profile = GeneratorProfile.from_cfg(jname, OC.to_container(cfg.judge.profiles[jname], resolve=True))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(judge_one, client, profile, r) for r in rows]
            for fut in as_completed(futs):
                r, p, meta = fut.result()
                cost += meta.get("cost_usd", 0.0)
                verdicts[r["fact_id"]][jname] = dict(
                    correct_true=bool(p.get("correct_true")),
                    wrong_false=bool(p.get("wrong_false")),
                    minimal_edit=bool(p.get("minimal_edit")),
                    note=str(p.get("note", ""))[:80])
        print(f"  {jname} done")

    # pass = BOTH judges say correct_true AND wrong_false AND minimal_edit
    out_rows, survivors = [], []
    for r in rows:
        v = verdicts[r["fact_id"]]
        per = {j: v.get(j, {}) for j in JUDGES}
        passed = all(per[j].get("correct_true") and per[j].get("wrong_false")
                     and per[j].get("minimal_edit") for j in JUDGES)
        out_rows.append(dict(fact_id=r["fact_id"], passed=passed,
                             **{f"{j}_{k}": per[j].get(k) for j in JUDGES
                                for k in ["correct_true", "wrong_false", "minimal_edit"]}))
        if passed:
            survivors.append(r)

    with open(OUT / "fidelity_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)

    variants = {}
    for r in survivors:
        fid = r["fact_id"]
        for label, ans in [(1, r["correct_answer"]), (0, r["wrong_answer"])]:
            pol = "correct" if label == 1 else "wrong"
            variants[f"{fid}__patient_real__{pol}"] = dict(
                fact_id=fid, register="patient_real", label=label,
                question=r["patient_question"], answer=ans, generator="medredqa-human")
    (OUT / "patient_real_variants_gated.json").write_text(json.dumps(variants, indent=1))

    # per-judge agreement on wrong_false (the load-bearing dimension)
    agree = statistics.mean(
        1.0 if verdicts[r["fact_id"]][JUDGES[0]].get("wrong_false") ==
               verdicts[r["fact_id"]][JUDGES[1]].get("wrong_false") else 0.0 for r in rows)
    rate = len(survivors) / len(rows)
    summary = dict(n_in=len(rows), n_pass=len(survivors), pass_rate=round(rate, 4),
                   judge_agreement_wrong_false=round(agree, 4), cost_usd=round(cost, 4))
    (OUT / "gate_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\nfidelity gate: {len(survivors)}/{len(rows)} pass ({rate:.1%}); "
          f"judge agreement on wrong_false = {agree:.1%}; cost ${cost:.4f}")
    print(f"wrote gated variants ({len(variants)} rows) to {OUT}")

if __name__ == "__main__":
    main()
