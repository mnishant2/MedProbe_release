# MedProbe

Code for constructing a register-controlled medical-QA benchmark and evaluating
linear truthfulness probes on open-weight LLMs across register, specialty, and
corpus shifts.

## Setup

```bash
uv sync            # or: pip install -e .
cp .env.example .env   # add OPENROUTER_API_KEY, HF_TOKEN
```

Configuration lives in `configs/` (`default.yaml`, `probe.yaml`, `openrouter.yaml`,
`judge.yaml`, `models.yaml`, and the register prompts in `configs/prompts/`).
Override any key from the command line with `--override key=value`.

## Pipeline

Scripts are ordered by stage. Each reads/writes paths defined in `configs/default.yaml`.

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `download_medqa.py` | Download the MedQA source corpus |
| 2 | `build_facts.py` | Sample facts and a paired wrong answer per item |
| 3 | `generate_variants.py` | Rewrite each fact into four registers (LLM generator) |
| 4 | `quality_report.py` | Lexical register-distinctness metrics |
| 5 | `extract_activations.py` | Extract hidden states per (model, layer, position) |
| 5b | `self_consistency.py` | Self-consistency output baseline |
| 6 | `train_probes.py` | Train/evaluate probes; output-only baselines |
| 7 | `ablations.py` | Mixed-register and related ablations |
| 9 | `judge_quality.py` | LLM-as-judge quality rubric |
| 10 | `select_generator.py` | Aggregate generator-selection scores |
| 11 | `judge_consistency.py` | Judge inter-rater agreement |
| 12 | `calibrate_and_errors.py` | Calibration (ECE, Platt, isotonic) and error analysis |
| 13 | `cross_dataset_eval.py` | Cross-corpus transfer (MedQA -> MedMCQA) |

Supporting analyses: `build_medmcqa_variants.py`, `run_diff_means_probe.py`,
`run_topic_transfer.py`, `run_perplexity_per_register.py`, `tag_specialty.py`.

## Models

Four open-weight instruction-tuned LLMs (configured in `configs/models.yaml`):
Gemma-2-2B, Gemma-3-4B, Qwen2.5-7B, Llama-3-8B.
