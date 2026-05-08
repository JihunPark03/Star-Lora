# Dataset-Aware and Stability-Aware AdaLoRA

This repository contains an experimental low-resource fine-tuning script for sequence classification. It compares full fine-tuning, fixed-rank LoRA, standard AdaLoRA, and a dataset-aware AdaLoRA variant.

The proposed method extends AdaLoRA in two ways:

- Dataset-aware rank selection: before training, the dataset is profiled for sample count, label imbalance, token sparsity, token-frequency skew, label entropy, and input-embedding variance. These statistics are combined into a target rank.
- Stability-aware monitoring: during training, LoRA parameter importance is logged with an exponential moving average and variance normalization so rank-adjustment behavior can be inspected for noisy or unstable updates.

## Project Files

- `train.py`: main training entry point.
- `dataset_analyzer.py`: computes dataset statistics used by the adaptive rank policy.
- `rank_policy.py`: maps dataset statistics to a bounded target LoRA rank.
- `stability_callback.py`: logs stability-aware importance scores and triggers AdaLoRA rank allocation.
- `run_experiments.py`: runs multi-method, multi-seed low-resource sweeps and writes aggregate reports.

## Methods

`train.py` supports four methods:

- `full`: updates all model parameters.
- `lora`: uses fixed-rank LoRA.
- `adalora`: uses PEFT AdaLoRA with the provided base rank as the target rank.
- `dataset_aware_adalora`: computes a dataset-aware target rank, then runs AdaLoRA with that target rank.

For AdaLoRA methods, the training schedule is estimated from `num_train_examples / batch_size * epochs` and passed to PEFT as `total_step`.

## Installation

Create an environment and install the core dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install torch transformers peft datasets evaluate numpy scikit-learn
```

The current workspace already contains a `venv`, but the commands above document a reproducible setup.

## Quick Start

Run the dataset-aware AdaLoRA method on a low-resource SST-2 subset:

```bash
python train.py \
  --model_name distilbert-base-uncased \
  --dataset_name glue \
  --dataset_config sst2 \
  --text_column sentence \
  --label_column label \
  --num_samples 1000 \
  --method dataset_aware_adalora \
  --base_rank 8 \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-5 \
  --output_dir ./outputs/dataset_aware_adalora
```

Compare against standard AdaLoRA:

```bash
python train.py \
  --method adalora \
  --base_rank 8 \
  --output_dir ./outputs/adalora
```

Compare against fixed-rank LoRA:

```bash
python train.py \
  --method lora \
  --base_rank 8 \
  --output_dir ./outputs/lora
```

## Outputs

Each run writes:

- `experiment_info.json`: method, rank settings, dataset statistics for the dataset-aware method, and final evaluation metrics.
- `stability_scores.csv`: per-step LoRA parameter importance logs for LoRA/AdaLoRA methods.
- Trainer checkpoints in the selected `output_dir`.

## Important Notes

This implementation is a research prototype. It should run for DistilBERT-style sequence classification models whose attention projection module names are `q_lin` and `v_lin`. Other model families often use names like `query`/`value` or `q_proj`/`v_proj`; update `target_modules` in `train.py` before using those models.

The current task path is classification-oriented. GLUE regression tasks such as STS-B are not supported without changing the label handling, model problem type, and metrics.

The stability-aware component currently logs smoothed importance and variance-normalized scores. It does not yet replace PEFT AdaLoRA's internal importance allocator with a custom stability-gated allocator, so treat claims about improved stability as an experimental hypothesis to validate with repeated runs.

## Experiments

Run each method over multiple seeds and low-resource budgets, for example 100, 500, 1000, and 5000 samples. Report accuracy, macro F1, convergence speed, final selected rank, and training variance across seeds.

You can run the full suggested sweep with:

```bash
python run_experiments.py \
  --methods full lora adalora dataset_aware_adalora \
  --sample_budgets 100 500 1000 5000 \
  --seeds 13 21 42 \
  --epochs 3 \
  --batch_size 8 \
  --output_root ./outputs/sweeps
```

For a quicker pilot run:

```bash
python run_experiments.py \
  --methods lora adalora dataset_aware_adalora \
  --sample_budgets 100 \
  --seeds 13 \
  --epochs 1 \
  --output_root ./outputs/pilot
```

The sweep writes:

- `results.csv`: one row per method/sample-budget/seed run.
- `summary.csv`: mean and standard deviation grouped by method and sample budget.
- `summary.md`: a Markdown report with accuracy, macro F1, runtime, throughput, selected rank, and cross-seed variance.

Use `--skip_existing` to resume a partially completed sweep without rerunning finished experiments. Use `--dry_run` to print the commands without launching training. Use `--collect_only` to regenerate `results.csv`, `summary.csv`, and `summary.md` from existing run folders.
