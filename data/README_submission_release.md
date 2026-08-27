# MedProbe: register-controlled medical-QA probing data

Dataset and numerical results for the accompanying paper on linear-probe transfer
in medical question answering. Models are referred to by open-weight identifiers
(Gemma-2-2B, Gemma-3-4B, Qwen2.5-7B, Llama-3-8B). No model weights, code, or
figures are included.

## benchmark/
- `medqa-facts.json` — 500 source MedQA facts: `id`, `source`, `split`, `question`,
  `correct_answer`, `wrong_answer`, `all_options`, `rarity`, `specialty`.
- `medmcqa-facts.json` — MedMCQA facts for the cross-corpus evaluation.
- `specialty_map.json` — fact-id to clinical-specialty label.
- `variants/` — each fact rewritten into four registers (`textbook`, `patient`,
  `clinical_note`, `colloquial`) with both correctness polarities. One file per
  generation source: `medqa-sonnet.json` (primary, 4,000 variants), `gemini.json`
  (robustness subset), `medmcqa-sonnet.json` (MedMCQA register rewrites),
  `medmcqa-textbook.json` (MedMCQA native, textbook only). Each record:
  `fact_id`, `register`, `label` (1=correct, 0=wrong), `question`, `answer`,
  `original_question`, `original_answer`, `generator`.

## results/
All AUROC results use 1,000-iteration fact-level bootstrap CIs. Register-transfer
results are reported on the held-out 20% fact split (the primary protocol).

- `register_transfer_medqa.csv` / `register_transfer_medmcqa.csv` — per
  (model, register) best-layer AUROC with CIs, for the logistic, MLP, and
  difference-of-means probes.
- `layerwise_auroc_medqa.csv` / `layerwise_auroc_medmcqa.csv` — per
  (model, layer, register) AUROC: held-out logistic / MLP / diff-means /
  mixed-register-trained, plus the all-facts logistic value.
- `mixed_register_training.csv` — textbook-only vs mixed-register-trained probe
  per (model, register), held-out, with CIs.
- `cross_corpus_transfer.csv` — MedQA-trained probe on MedQA-textbook held-out vs
  MedMCQA, per (model, layer).
- `specialty_transfer.csv` — specialty-disjoint transfer (5 random splits/model).
- `output_baselines.csv` — P(True), self-consistency, and token-entropy AUROC per
  (model, register), held-out.
- `calibration_ece.csv` — raw / Platt / isotonic ECE per register.
- `diff_means_vs_logistic.csv` — difference-of-means vs logistic probe per
  (model, register).
- `common_vs_rare.csv` — AUROC split by common vs rare disease.
- `perplexity_by_register.csv` — reference-LM perplexity per variant.
- `generator_selection.csv` — generator-selection pilot scores.
