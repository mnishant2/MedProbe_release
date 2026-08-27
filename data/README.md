# Released data

All files are derived from public benchmarks (MedQA, MedMCQA, MMLU) under their research licenses;
this directory is released under CC BY-NC 4.0 (code in this repository is MIT).

## `benchmark/` (as shared with the submission)
- `variants/medqa-sonnet.json` — the 4,000-variant register benchmark: 500 MedQA facts x 4 registers
  (textbook, patient, clinical_note, colloquial) x 2 polarities, rewritten by Claude Sonnet 4.5,
  with per-variant judge scores.
- `variants/medqa-gemini.json` — 800-variant replication of 100 facts with Gemini 3 Flash Preview.
- `variants/medmcqa-sonnet.json` — 100 MedMCQA facts x 4 registers x 2 (within-MedMCQA replication).
- `variants/medmcqa-textbook.json` — 500 native MedMCQA validation items (correct + one distractor).
- `medqa-facts.json`, `medmcqa-facts.json` — the sampled facts with gold answers and distractors.
- `specialty_map.json` — S-MedQA specialty labels for the 351 matched MedQA facts.

## `variants/` (evaluation-only sets added for the camera-ready)
- `mmlu-medical/variants.json` — 500 MMLU items from anatomy, clinical knowledge, college medicine,
  professional medicine and medical genetics (1,000 variants), built by `scripts/additional_experiments/build_mmlu_medical.py`.
- `medmcqa-reformat/variants.json` — 100 MedMCQA items rewritten into MedQA-style vignettes with the
  answers held fixed (200 variants), built by `scripts/additional_experiments/reformat_medmcqa.py`.
- `medqa-native/variants.json` — the 500 MedQA facts with their native (unrewritten) correct answer and
  distractor, used for the answer-length sanity check.
- MedRedQA (human-written patient questions) is **not** redistributed (Reddit-derived text). The build
  script `scripts/additional_experiments/build_medredqa_slice.py` reconstructs the 100-item slice from the
  public MedRedQA release; `medredqa_fidelity_gate.py` applies the two-judge gate that keeps 84 items.

## `results/` (as shared with the submission)
Aggregate result tables behind the paper's figures: per-register and layer-wise AUROC, specialty
transfer, cross-corpus transfer, common-vs-rare, mixed-register training, diff-means vs logistic,
output baselines, calibration ECE, generator selection, perplexity by register.

Trained probe weights, per-variant prediction dumps and the camera-ready result tables
(register matrix, fidelity filter, corpus ladder, MedRedQA, thresholds) are not part of this release.
