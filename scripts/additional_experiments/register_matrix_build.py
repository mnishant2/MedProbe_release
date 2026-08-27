#!/usr/bin/env python3
"""Assemble the 4x4 single-source register training matrix (train on one register, evaluate on all four) from the per-source probe result files."""
import csv, glob, os
from collections import defaultdict

ROOT = os.environ.get("MEDPROBE_ROOT", ".")
RAW = os.path.join(ROOT, "outputs/camera_ready/register_matrix/raw")
OUT = os.path.join(ROOT, "outputs/camera_ready/register_matrix")
REGISTERS = ["textbook", "patient", "clinical_note", "colloquial"]
POSITION = "last_question_token"

def best_auroc_per_register(path):
    """per-cell argmax over layers at POSITION -> {register: auroc}"""
    best = defaultdict(lambda: -1.0)
    for r in csv.DictReader(open(path)):
        if r.get("method") != "probe" or r.get("position") != POSITION or r.get("rarity", "") != "":
            continue
        a = float(r["auroc"])
        if a > best[r["register"]]:
            best[r["register"]] = a
    return dict(best)

def main():
    files = glob.glob(os.path.join(RAW, "probe_results__train-*__*.csv"))
    if not files:
        print(f"No per-source CSVs in {RAW} yet, run run_register_matrix.sh first.")
        return
    # model -> src -> {tgt: auroc}
    mats = defaultdict(dict)
    for f in files:
        base = os.path.basename(f)
        src = base.split("train-")[1].split("__")[0]
        model = base.split("__")[-1].replace(".csv", "")
        mats[model][src] = best_auroc_per_register(f)

    summary_rows = []
    for model in sorted(mats):
        mpath = os.path.join(OUT, f"matrix__{model}.csv")
        with open(mpath, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["train\\eval"] + REGISTERS)
            for src in REGISTERS:
                row = mats[model].get(src, {})
                w.writerow([src] + [round(row.get(t, float("nan")), 4) for t in REGISTERS])
        # tidy summary: for each source, mean off-diagonal gap vs its own diagonal
        for src in REGISTERS:
            row = mats[model].get(src, {})
            diag = row.get(src, float("nan"))
            offs = [diag - row.get(t, float("nan")) for t in REGISTERS if t != src]
            mean_gap = sum(offs) / len(offs) if offs else float("nan")
            summary_rows.append(dict(model=model, train_register=src,
                                     diag_auroc=round(diag, 4),
                                     mean_transfer_gap=round(mean_gap, 4)))
        print(f"wrote {mpath}")

    with open(os.path.join(OUT, "matrix_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "train_register", "diag_auroc", "mean_transfer_gap"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"wrote {os.path.join(OUT, 'matrix_summary.csv')}")

    # heatmaps (optional; skip silently if matplotlib unavailable)
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for model in sorted(mats):
            M = np.array([[mats[model].get(s, {}).get(t, np.nan) for t in REGISTERS] for s in REGISTERS])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(M, vmin=0.45, vmax=0.85, cmap="viridis")
            ax.set_xticks(range(4)); ax.set_xticklabels(REGISTERS, rotation=45, ha="right")
            ax.set_yticks(range(4)); ax.set_yticklabels(REGISTERS)
            ax.set_xlabel("eval register"); ax.set_ylabel("train register")
            ax.set_title(model)
            for i in range(4):
                for j in range(4):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                color="white" if M[i, j] < 0.7 else "black", fontsize=8)
            fig.colorbar(im, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"heatmap__{model}.pdf"))
            plt.close(fig)
        print("wrote heatmap PDFs")
    except Exception as e:
        print(f"(heatmaps skipped: {e})")

if __name__ == "__main__":
    main()
