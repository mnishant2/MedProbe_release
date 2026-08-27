#!/bin/bash
# Extracts activations for the evaluation-only sets (MMLU-medical, reformatted MedMCQA, MedRedQA)
# and applies the MedQA-textbook probe to them without retraining.
# Usage: MEDPROBE_ROOT=<repo> bash scripts/additional_experiments/run_extract_new_sets.sh
set -euo pipefail
REPO="${REPO:-${MEDPROBE_ROOT:-.}}"
PY="${PY:-$REPO/.venv/bin/python}"
cd "$REPO"
MODELS=(gemma-2-2b-it gemma-3-4b-it qwen2.5-7b-instruct llama-3-8b-instruct)
OUTD="outputs/camera_ready"
mkdir -p "$OUTD"

for GEN in mmlu-medical medmcqa-reformat patient-real; do
  [ -f "data/variants/$GEN/variants.json" ] || { echo "skip $GEN (no variants file)"; continue; }
  for MODEL in "${MODELS[@]}"; do
    "$PY" scripts/extract_activations.py --model "$MODEL" --generator "$GEN" --resume
  done
done

for GEN in mmlu-medical medmcqa-reformat patient-real; do
  [ -f "data/variants/$GEN/variants.json" ] || continue
  for MODEL in "${MODELS[@]}"; do
    "$PY" scripts/cross_dataset_eval.py --model "$MODEL" \
      --medqa-generator sonnet --medmcqa-generator "$GEN" \
      --out "$OUTD/${GEN}_crosseval__${MODEL}.csv"
  done
done
