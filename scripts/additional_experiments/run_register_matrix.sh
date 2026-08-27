#!/bin/bash
# Trains one probe per training register (textbook, patient, clinical_note, colloquial) for each
# model and evaluates every register, producing the 4x4 register training matrix.
# Requires per-variant activation files. Usage: ACT_DIR=<activations>/sonnet bash run_register_matrix.sh
set -euo pipefail

REPO="${REPO:-${MEDPROBE_ROOT:-.}}"
GENERATOR="${GENERATOR:-sonnet}"
ACT_DIR="${ACT_DIR:-$REPO/outputs/activations}"   # override to the real activations dir
OUTDIR="${OUTDIR:-$REPO/outputs/camera_ready/register_matrix/raw}"
PY="${PY:-$REPO/.venv/bin/python}"

MODELS=(gemma-2-2b-it gemma-3-4b-it qwen2.5-7b-instruct llama-3-8b-instruct)
REGISTERS=(textbook patient clinical_note colloquial)

mkdir -p "$OUTDIR"
cd "$REPO"

# sanity: activations present?
if ! ls "$ACT_DIR"/*/*.npz >/dev/null 2>&1; then
  echo "ERROR: no .npz activations found under $ACT_DIR" >&2
  echo "       Set ACT_DIR to the activations directory." >&2
  exit 2
fi

for MODEL in "${MODELS[@]}"; do
  for TR in "${REGISTERS[@]}"; do
    OUT="$OUTDIR/probe_results__train-${TR}__${MODEL}.csv"
    echo "=== train_register=$TR  model=$MODEL -> $OUT ==="
    "$PY" scripts/train_probes.py --model "$MODEL" --generator "$GENERATOR" \
      --bootstrap 1000 --out "$OUT" \
      --override train_register="$TR" paths.activations_dir="$ACT_DIR"
  done
done

echo "=== all 16 runs done; assembling matrix ==="
"$PY" "$REPO/scripts/additional_experiments/register_matrix_build.py"
