import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_METHODS = [
    "full",
    "lora",
    "adalora",
    "dataset_aware_adalora",
    "dataset_aware_layerwise_adalora",
]

DEFAULT_SAMPLE_BUDGETS = [100, 500, 1000, 5000]
DEFAULT_SEEDS = [13, 21, 42]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run low-resource LoRA/AdaLoRA sweeps and summarize results."
    )

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config", type=str, default="sst2")
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--sample_budgets", nargs="+", type=int, default=DEFAULT_SAMPLE_BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument(
        "--torch_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_root", type=str, default="./outputs/sweeps")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")

    return parser.parse_args()


def run_name(method, sample_budget, seed):
    return f"{method}_n{sample_budget}_seed{seed}"


def experiment_dir(output_root, method, sample_budget, seed):
    return Path(output_root) / run_name(method, sample_budget, seed)


def experiment_info_path(output_root, method, sample_budget, seed):
    return experiment_dir(output_root, method, sample_budget, seed) / "experiment_info.json"


def build_command(args, method, sample_budget, seed, output_dir):
    return [
        args.python,
        str(PROJECT_DIR / "train.py"),
        "--model_name",
        args.model_name,
        "--dataset_name",
        args.dataset_name,
        "--dataset_config",
        args.dataset_config,
        "--text_column",
        args.text_column,
        "--label_column",
        args.label_column,
        "--num_samples",
        str(sample_budget),
        "--method",
        method,
        "--base_rank",
        str(args.base_rank),
        "--torch_dtype",
        args.torch_dtype,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
    ]


def run_one(args, method, sample_budget, seed):
    output_dir = experiment_dir(args.output_root, method, sample_budget, seed)
    info_path = output_dir / "experiment_info.json"

    if args.skip_existing and info_path.exists():
        print(f"Skipping existing run: {info_path}")
        return

    command = build_command(args, method, sample_budget, seed, output_dir)

    print("Running:", " ".join(command))

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, cwd=PROJECT_DIR)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for method={method}, samples={sample_budget}, seed={seed}"
        )


def load_json(path):
    with open(path) as file:
        return json.load(file)


def as_float(value):
    if value is None:
        return None

    return float(value)


def flatten_result(output_root, method, sample_budget, seed):
    info_path = experiment_info_path(output_root, method, sample_budget, seed)

    if not info_path.exists():
        return {
            "method": method,
            "num_samples": sample_budget,
            "seed": seed,
            "status": "missing",
        }

    info = load_json(info_path)
    eval_metrics = info.get("metrics", {})
    train_metrics = info.get("train_metrics", {})

    return {
        "method": method,
        "num_samples": sample_budget,
        "seed": seed,
        "status": "ok",
        "accuracy": as_float(eval_metrics.get("eval_accuracy")),
        "macro_f1": as_float(eval_metrics.get("eval_macro_f1")),
        "eval_loss": as_float(eval_metrics.get("eval_loss")),
        "train_loss": as_float(train_metrics.get("train_loss")),
        "train_runtime": as_float(train_metrics.get("train_runtime")),
        "train_samples_per_second": as_float(train_metrics.get("train_samples_per_second")),
        "train_steps_per_second": as_float(train_metrics.get("train_steps_per_second")),
        "selected_rank": info.get("rank"),
        "init_rank": info.get("init_rank"),
        "estimated_total_steps": info.get("estimated_total_steps"),
        "output_dir": str(experiment_dir(output_root, method, sample_budget, seed)),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    clean = [value for value in values if value is not None]

    if not clean:
        return None

    return statistics.mean(clean)


def stdev(values):
    clean = [value for value in values if value is not None]

    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None

    return statistics.stdev(clean)


def round_or_blank(value, digits=4):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return round(value, digits)


def summarize(rows):
    grouped = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        key = (row["method"], row["num_samples"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []

    for (method, sample_budget), group in sorted(grouped.items()):
        accuracy_values = [row.get("accuracy") for row in group]
        f1_values = [row.get("macro_f1") for row in group]
        runtime_values = [row.get("train_runtime") for row in group]
        samples_per_second_values = [row.get("train_samples_per_second") for row in group]
        steps_per_second_values = [row.get("train_steps_per_second") for row in group]
        rank_values = [row.get("selected_rank") for row in group]

        summary_rows.append(
            {
                "method": method,
                "num_samples": sample_budget,
                "num_seeds": len(group),
                "accuracy_mean": round_or_blank(mean(accuracy_values)),
                "accuracy_std": round_or_blank(stdev(accuracy_values)),
                "macro_f1_mean": round_or_blank(mean(f1_values)),
                "macro_f1_std": round_or_blank(stdev(f1_values)),
                "train_runtime_mean": round_or_blank(mean(runtime_values), digits=2),
                "train_samples_per_second_mean": round_or_blank(
                    mean(samples_per_second_values), digits=2
                ),
                "train_steps_per_second_mean": round_or_blank(
                    mean(steps_per_second_values), digits=2
                ),
                "selected_rank_mean": round_or_blank(mean(rank_values), digits=2),
                "selected_rank_std": round_or_blank(stdev(rank_values), digits=2),
            }
        )

    return summary_rows


def markdown_table(rows, fieldnames):
    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")

    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_markdown_report(path, summary_rows, detail_rows):
    summary_fields = [
        "method",
        "num_samples",
        "num_seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "train_runtime_mean",
        "train_samples_per_second_mean",
        "train_steps_per_second_mean",
        "selected_rank_mean",
        "selected_rank_std",
    ]

    failed_or_missing = [
        row for row in detail_rows if row.get("status") != "ok"
    ]

    lines = [
        "# Experiment Summary",
        "",
        "Convergence speed is reported with training runtime, samples per second, and steps per second.",
        "",
        markdown_table(summary_rows, summary_fields),
    ]

    if failed_or_missing:
        lines.extend(
            [
                "",
                "## Missing or Failed Runs",
                "",
                markdown_table(
                    failed_or_missing,
                    ["method", "num_samples", "seed", "status"],
                ),
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def collect_results(args):
    detail_rows = []

    for method in args.methods:
        for sample_budget in args.sample_budgets:
            for seed in args.seeds:
                detail_rows.append(
                    flatten_result(args.output_root, method, sample_budget, seed)
                )

    summary_rows = summarize(detail_rows)
    output_root = Path(args.output_root)

    detail_fields = [
        "method",
        "num_samples",
        "seed",
        "status",
        "accuracy",
        "macro_f1",
        "eval_loss",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "selected_rank",
        "init_rank",
        "estimated_total_steps",
        "output_dir",
    ]

    summary_fields = [
        "method",
        "num_samples",
        "num_seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "train_runtime_mean",
        "train_samples_per_second_mean",
        "train_steps_per_second_mean",
        "selected_rank_mean",
        "selected_rank_std",
    ]

    write_csv(output_root / "results.csv", detail_rows, detail_fields)
    write_csv(output_root / "summary.csv", summary_rows, summary_fields)
    write_markdown_report(output_root / "summary.md", summary_rows, detail_rows)

    print(f"Wrote {output_root / 'results.csv'}")
    print(f"Wrote {output_root / 'summary.csv'}")
    print(f"Wrote {output_root / 'summary.md'}")


def main():
    args = parse_args()

    if args.collect_only:
        collect_results(args)
        return

    if not args.dry_run:
        os.makedirs(args.output_root, exist_ok=True)

    for method in args.methods:
        for sample_budget in args.sample_budgets:
            for seed in args.seeds:
                run_one(args, method, sample_budget, seed)

    if not args.dry_run:
        collect_results(args)


if __name__ == "__main__":
    main()
