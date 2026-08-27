# Additional experiments

Scripts for the analyses added in the camera-ready version. All of them reuse the main pipeline
(`scripts/extract_activations.py`, `scripts/train_probes.py`, `scripts/cross_dataset_eval.py`) and the
released data in `data/`. Set `MEDPROBE_ROOT` to the repository root; outputs go to
`outputs/camera_ready/<experiment>/`.

| Script | What it does |
|---|---|
| `build_mmlu_medical.py` | Builds the MMLU-medical evaluation set (500 items, five medical subjects) in the benchmark's variant format. |
| `reformat_medmcqa.py` | Rewrites MedMCQA question stems into MedQA-style vignettes with the answers held fixed (question-format control). |
| `run_extract_new_sets.sh` | Extracts activations for the new sets and applies the MedQA-textbook probe to them without retraining. |
| `mmlu_by_subject.py` | Per-subject AUROC of the MedQA-textbook probe on MMLU-medical. |
| `build_medredqa_slice.py`, `medredqa_fidelity_gate.py`, `eval_medredqa.py` | Build the human-written patient-question slice from MedRedQA (verbatim questions, length-matched minimal-edit wrong claims), apply the two-judge fidelity gate, and evaluate the MedQA-textbook probe on it. The MedRedQA text itself is not redistributed. |
| `heldout_predictions.py`, `fidelity_filter_compare.py` | Per-variant held-out predictions of the textbook probe at every layer, and the register gap after removing wrong variants below a judged fidelity threshold. |
| `run_register_matrix.sh`, `register_matrix_build.py` | Train one probe per register and evaluate on all four (the 4x4 register training matrix). |
| `register_pairs_train.py` | Probes trained on pairs of registers, single registers, and all four, evaluated on the shared held-out split. |
| `specialty_matched_vs_unmatched.py` | AUROC on S-MedQA-matched vs. unmatched facts with bootstrap CIs. |
| `answer_length_baseline.py` | Word-count-only baseline AUROC per variant set and register. |
