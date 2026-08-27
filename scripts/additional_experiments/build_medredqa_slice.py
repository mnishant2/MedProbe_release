#!/usr/bin/env python3
"""Build the MedRedQA evaluation slice: verbatim patient questions, a correct claim extracted from the clinician reply, and a length-matched minimal-edit wrong claim (negation or entity swap)."""
import os
import argparse, csv, json, sys, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(os.environ.get("MEDPROBE_ROOT", "."))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/medredqa"

from dotenv import load_dotenv
from omegaconf import OmegaConf as OC
from medprobe.config import load_config, resolve_path
from medprobe.data.openrouter_client import GeneratorProfile, OpenRouterClient

load_dotenv(REPO / ".env")

MAX_WORD_DIFF = 2      # correct/wrong claim length must match within 2 words
MIN_JACCARD = 0.55     # minimal edit: high token overlap between correct and wrong

SYSTEM = (
    "You build truth-probing items from a real patient forum thread. You output ONE true "
    "medical claim and ONE minimally-edited FALSE version of that same claim. "
    "Output ONLY a JSON object with keys usable, correct_answer, wrong_answer, edit_type."
)

USER_TMPL = """PATIENT QUESTION (verbatim, human-written):
{question}

VERIFIED CLINICIAN REPLY (source of truth; will NOT be shown to the model under test):
{response}

Task:
1. "usable": true only if the clinician reply makes a substantive factual medical assertion.
   If it is only a clarifying question or gives no assertion, set usable=false and leave the rest empty.
2. "correct_answer": ONE declarative medical statement (12-25 words) that faithfully states the
   clinician's answer to this patient. Add no facts the clinician did not state.
3. "wrong_answer": a MINIMAL EDIT of correct_answer that makes it MEDICALLY FALSE.
   HARD CONSTRAINTS:
     - Change as few words as possible: either negate the key proposition, or swap the key
       clinical entity (drug, organ, diagnosis, direction, number) for a wrong one.
     - Keep the SAME topic, SAME sentence structure, and the SAME length (within ~2 words).
     - It must be clearly medically WRONG, not merely vague or incomplete.
     - Do NOT introduce a new topic. Do NOT make it longer or shorter than correct_answer.
4. "edit_type": either "negation" or "entity_swap".

Example:
  correct_answer: "Your one-sided orthostatic headaches are most likely migraine and do not require imaging."
  wrong_answer:   "Your one-sided orthostatic headaches are most likely a brain tumour and do require imaging."
  edit_type: "entity_swap"

Return JSON: {{"usable": bool, "correct_answer": "...", "wrong_answer": "...", "edit_type": "..."}}"""

def clean_q(r, max_chars=700):
    return (r["title"].strip() + ". " + r["body"].strip()).strip()[:max_chars]

def toks(s):
    return {w.strip(".,;:!?()").lower() for w in s.split() if w.strip(".,;:!?()")}

def minimal_edit_ok(correct, wrong):
    """Enforce: same length (within MAX_WORD_DIFF) and high token overlap."""
    if not correct or not wrong or correct.strip().lower() == wrong.strip().lower():
        return False, "identical/empty"
    wc, ww = len(correct.split()), len(wrong.split())
    if abs(wc - ww) > MAX_WORD_DIFF:
        return False, f"length mismatch ({wc} vs {ww})"
    a, b = toks(correct), toks(wrong)
    j = len(a & b) / max(1, len(a | b))
    if j < MIN_JACCARD:
        return False, f"jaccard {j:.2f} < {MIN_JACCARD} (not a minimal edit)"
    return True, f"jaccard {j:.2f}, dlen {abs(wc-ww)}"

def build_one(client, profile, r):
    prompt = USER_TMPL.format(question=clean_q(r), response=r["response"].strip())
    parsed, meta = client.generate_json(profile, prompt, system=SYSTEM)
    return r, (parsed or {}), meta

def candidate_filter(r):
    o = r["occupation"].lower()
    resp = r["response"].strip()
    return (r["response_score"] >= 3
            and ("physician" in o or "md" in o or "doctor" in o)
            and 40 <= len(resp) <= 400
            and len(r["body"]) > 60
            and not resp.rstrip().endswith("?"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--dry-preview", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("bagga005/medredqa")["test"]
    cands = [r for r in ds if candidate_filter(r)]
    import random
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    pool = cands[: args.n * 3]   # overshoot: usable=false + minimal-edit rejects
    print(f"MedRedQA test {len(ds)}; {len(cands)} pass filter; pool {len(pool)}")

    cfg = load_config()
    profile = GeneratorProfile.from_cfg(
        args.generator, OC.to_container(cfg.openrouter.generators[args.generator], resolve=True))
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(
        base_url=str(cfg.openrouter.base_url), api_key_env=str(cfg.openrouter.api_key_env),
        referer=str(cfg.openrouter.referer), app_title=str(cfg.openrouter.app_title),
        retry_max_attempts=int(cfg.openrouter.retry.max_attempts),
        retry_initial_wait=float(cfg.openrouter.retry.initial_wait_s),
        retry_max_wait=float(cfg.openrouter.retry.max_wait_s),
        ledger_path=OUT / "cost_ledger.jsonl")

    usable, rejected, cost = [], {"unusable": 0, "minimal_edit": 0}, 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, client, profile, r) for r in pool]
        for fut in as_completed(futs):
            r, p, meta = fut.result()
            cost += meta.get("cost_usd", 0.0)
            if not (p.get("usable") and p.get("correct_answer") and p.get("wrong_answer")):
                rejected["unusable"] += 1
                continue
            ok, why = minimal_edit_ok(p["correct_answer"], p["wrong_answer"])
            if not ok:
                rejected["minimal_edit"] += 1
                if args.dry_preview:
                    print(f"  REJECT [{r['id']}] {why}")
                continue
            usable.append((r, p))
            if args.dry_preview:
                print(f"\n[{r['id']}] ({p.get('edit_type')}, {why})")
                print(f"   correct: {p['correct_answer']}")
                print(f"   wrong  : {p['wrong_answer']}")
            if len(usable) >= args.n and not args.dry_preview:
                break

    usable = usable[: args.n]
    print(f"\nusable {len(usable)}; rejected {rejected}; cost ${cost:.4f}")
    if not usable:
        return

    from sklearn.metrics import roc_auc_score
    L = [len(p["correct_answer"].split()) for _, p in usable] + [len(p["wrong_answer"].split()) for _, p in usable]
    Y = [1] * len(usable) + [0] * len(usable)
    print(f"length-only AUROC of constructed claims = {roc_auc_score(Y, L):.3f} (target ~0.50)")
    print(f"mean words: correct {statistics.mean(L[:len(usable)]):.1f}, wrong {statistics.mean(L[len(usable):]):.1f}")
    if args.dry_preview:
        return

    variants, pairs = {}, []
    for r, p in usable:
        fid = f"medredqa_{r['id']}"
        q = clean_q(r)
        for label, ans in [(1, p["correct_answer"]), (0, p["wrong_answer"])]:
            pol = "correct" if label == 1 else "wrong"
            variants[f"{fid}__patient_real__{pol}"] = dict(
                fact_id=fid, register="patient_real", label=label,
                question=q, answer=ans, generator="medredqa-human")
        pairs.append(dict(fact_id=fid, occupation=r["occupation"], response_score=r["response_score"],
                          edit_type=p.get("edit_type", ""), patient_question=q,
                          clinician_reply=r["response"].strip(),
                          correct_answer=p["correct_answer"], wrong_answer=p["wrong_answer"]))

    (OUT / "patient_real_variants.json").write_text(json.dumps(variants, indent=1))
    with open(OUT / "patient_real_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
        w.writeheader(); w.writerows(pairs)
    print(f"wrote {len(variants)} variant rows, {len(pairs)} pairs to {OUT}")

if __name__ == "__main__":
    main()
