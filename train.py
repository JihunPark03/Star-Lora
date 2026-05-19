import argparse
import json
import math
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import (
    AdaLoraConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset_analyzer import analyze_dataset
from rank_policy import compute_dataset_aware_rank
from stability_callback import AdaLoraAllocationCallback, StabilityAwareCallback


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="glue",
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sst2",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="sentence",
    )

    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "lora",
            "adalora",
            "dataset_aware_adalora",
            "dataset_aware_layerwise_adalora",
            "full",
        ],
        default="dataset_aware_adalora",
    )

    parser.add_argument(
        "--base_rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="LoRA target module names. Defaults are inferred from the model name.",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=[
            "float32",
            "float16",
            "bfloat16",
        ],
        default="float32",
        help="Model loading dtype. Qwen sequence classification uses float32 by default to avoid BF16 NaNs.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_low_resource_dataset(args):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
    )

    train_dataset = dataset["train"]

    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        raise ValueError("Dataset must have validation or test split.")

    if args.num_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset = train_dataset.select(range(args.num_samples))

    return train_dataset, eval_dataset


def get_num_labels(train_dataset, label_column):
    labels = train_dataset[label_column]
    unique_labels = set(labels)

    return len(unique_labels)


def tokenize_dataset(dataset, tokenizer, text_column):
    def tokenize_function(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=256,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
    )


def infer_target_modules(model_name):
    normalized_name = model_name.lower()

    if "qwen" in normalized_name:
        return ["q_proj", "v_proj"]

    if "distilbert" in normalized_name:
        return ["q_lin", "v_lin"]

    return ["q_proj", "v_proj"]


def resolve_torch_dtype(dtype_name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def build_lora_config(
    rank,
    target_modules,
    lora_alpha=None,
    lora_dropout=0.1,
):
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha or rank * 2,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def build_adalora_config(
    init_rank,
    target_rank,
    total_step,
    target_modules,
    lora_alpha=None,
    lora_dropout=0.1,
    tinit_fraction=0.1,
    tfinal_fraction=0.5,
    delta_fraction=0.02,
):
    tinit = max(0, int(tinit_fraction * total_step))
    tfinal = max(0, int(tfinal_fraction * total_step))
    delta_t = max(1, int(delta_fraction * total_step))

    if tinit + tfinal >= total_step:
        tinit = 0
        tfinal = max(0, total_step - 1)

    return AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=init_rank,
        target_r=target_rank,
        lora_alpha=lora_alpha or target_rank * 2,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=delta_t,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=total_step,
    )


def infer_dataset_aware_target_modules(model_name, stats):
    normalized_name = model_name.lower()

    if "qwen" not in normalized_name:
        return infer_target_modules(model_name)

    target_modules = [
        "q_proj",
        "v_proj",
    ]

    if stats.label_entropy >= 0.8 or stats.token_sparsity >= 0.6:
        target_modules.extend(
            [
                "k_proj",
                "o_proj",
            ]
        )

    if stats.token_frequency_skew >= 2.0 and stats.num_samples >= 250:
        target_modules.extend(
            [
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )

    ordered_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    return [
        module
        for module in ordered_modules
        if module in set(target_modules)
    ]


def get_layer_index(module_name):
    parts = module_name.split(".")

    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])

    return None


def module_leaf_name(module_name):
    return module_name.rsplit(".", 1)[-1]


def infer_dataset_aware_layerwise_target_modules(model, model_name, stats):
    normalized_name = model_name.lower()

    if "qwen" not in normalized_name:
        return infer_dataset_aware_target_modules(model_name, stats), None

    module_names_by_layer = {}
    candidate_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    for module_name, _module in model.named_modules():
        layer_index = get_layer_index(module_name)

        if layer_index is None:
            continue

        leaf_name = module_leaf_name(module_name)

        if leaf_name not in candidate_modules:
            continue

        module_names_by_layer.setdefault(layer_index, {})[leaf_name] = module_name

    layer_indices = sorted(module_names_by_layer)

    if len(layer_indices) == 0:
        return infer_dataset_aware_target_modules(model_name, stats), None

    complex_attention_layers = set()
    mlp_layers = set()

    if stats.label_entropy >= 0.8 or stats.token_sparsity >= 0.6:
        start_index = len(layer_indices) // 2
        complex_attention_layers = set(layer_indices[start_index:])

    if stats.token_frequency_skew >= 2.0 and stats.num_samples >= 250:
        start_index = (len(layer_indices) * 2) // 3
        mlp_layers = set(layer_indices[start_index:])

    target_modules = []
    layerwise_target_modules = {}

    for layer_index in layer_indices:
        layer_modules = ["q_proj", "v_proj"]

        if layer_index in complex_attention_layers:
            layer_modules.extend(["k_proj", "o_proj"])

        if layer_index in mlp_layers:
            layer_modules.extend(["gate_proj", "up_proj", "down_proj"])

        selected_modules = []

        for leaf_name in layer_modules:
            module_name = module_names_by_layer[layer_index].get(leaf_name)

            if module_name is None:
                continue

            target_modules.append(module_name)
            selected_modules.append(leaf_name)

        layerwise_target_modules[str(layer_index)] = selected_modules

    return target_modules, layerwise_target_modules


def build_dataset_aware_adalora_settings(args, stats):
    target_modules = (
        args.target_modules
        or infer_dataset_aware_target_modules(args.model_name, stats)
    )

    rank = compute_dataset_aware_rank(
        stats=stats,
        base_rank=args.base_rank,
        min_rank=4,
        max_rank=48,
    )

    init_rank = max(rank + 8, rank * 2)
    lora_alpha = max(rank * 4, args.base_rank * 2)

    dropout = 0.05

    if stats.num_samples < 1000:
        dropout += 0.05

    if stats.class_imbalance > 0.2 or stats.token_sparsity > 0.6:
        dropout += 0.05

    lora_dropout = min(dropout, 0.2)

    return {
        "rank": rank,
        "init_rank": init_rank,
        "target_modules": target_modules,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "tinit_fraction": 0.2,
        "tfinal_fraction": 0.85,
        "delta_fraction": 0.05,
    }


def build_dataset_aware_layerwise_adalora_settings(args, model, stats):
    settings = build_dataset_aware_adalora_settings(args, stats)

    if args.target_modules:
        settings["layerwise_target_modules"] = None
        return settings

    target_modules, layerwise_target_modules = infer_dataset_aware_layerwise_target_modules(
        model=model,
        model_name=args.model_name,
        stats=stats,
    )

    settings["target_modules"] = target_modules
    settings["layerwise_target_modules"] = layerwise_target_modules

    return settings


def build_model(args, num_labels, train_dataset, tokenizer, total_step):
    target_modules = args.target_modules or infer_target_modules(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        dtype=resolve_torch_dtype(args.torch_dtype),
    )

    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    experiment_info = {
        "method": args.method,
        "base_rank": args.base_rank,
        "target_modules": target_modules,
        "torch_dtype": args.torch_dtype,
    }

    if args.method == "full":
        experiment_info["trainable_type"] = "full_finetuning"
        return model, experiment_info

    if args.method == "lora":
        rank = args.base_rank
        config = build_lora_config(rank, target_modules)

        model = get_peft_model(model, config)

        experiment_info["rank"] = rank
        experiment_info["trainable_type"] = "lora"

        return model, experiment_info

    if args.method == "adalora":
        rank = args.base_rank
        init_rank = max(rank + 4, 4)

        config = build_adalora_config(
            init_rank=init_rank,
            target_rank=rank,
            total_step=total_step,
            target_modules=target_modules,
        )

        model = get_peft_model(model, config)

        experiment_info["rank"] = rank
        experiment_info["init_rank"] = init_rank
        experiment_info["trainable_type"] = "adalora"

        return model, experiment_info

    if args.method == "dataset_aware_adalora":
        stats = analyze_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            model=model,
            text_column=args.text_column,
            label_column=args.label_column,
        )

        settings = build_dataset_aware_adalora_settings(args, stats)
        rank = settings["rank"]
        init_rank = settings["init_rank"]
        target_modules = settings["target_modules"]

        config = build_adalora_config(
            init_rank=settings["init_rank"],
            target_rank=settings["rank"],
            total_step=total_step,
            target_modules=settings["target_modules"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            tinit_fraction=settings["tinit_fraction"],
            tfinal_fraction=settings["tfinal_fraction"],
            delta_fraction=settings["delta_fraction"],
        )

        model = get_peft_model(model, config)

        experiment_info["rank"] = rank
        experiment_info["init_rank"] = init_rank
        experiment_info["target_modules"] = target_modules
        experiment_info["lora_alpha"] = settings["lora_alpha"]
        experiment_info["lora_dropout"] = settings["lora_dropout"]
        experiment_info["adalora_schedule"] = {
            "tinit_fraction": settings["tinit_fraction"],
            "tfinal_fraction": settings["tfinal_fraction"],
            "delta_fraction": settings["delta_fraction"],
        }
        experiment_info["dataset_stats"] = stats.to_dict()
        experiment_info["trainable_type"] = "dataset_aware_adalora"

        return model, experiment_info


    if args.method == "dataset_aware_layerwise_adalora":
        stats = analyze_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            model=model,
            text_column=args.text_column,
            label_column=args.label_column,
        )

        settings = build_dataset_aware_layerwise_adalora_settings(
            args=args,
            model=model,
            stats=stats,
        )
        rank = settings["rank"]
        init_rank = settings["init_rank"]
        target_modules = settings["target_modules"]

        config = build_adalora_config(
            init_rank=init_rank,
            target_rank=rank,
            total_step=total_step,
            target_modules=target_modules,
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            tinit_fraction=settings["tinit_fraction"],
            tfinal_fraction=settings["tfinal_fraction"],
            delta_fraction=settings["delta_fraction"],
        )

        model = get_peft_model(model, config)

        experiment_info["rank"] = rank
        experiment_info["init_rank"] = init_rank
        experiment_info["target_modules"] = target_modules
        experiment_info["layerwise_target_modules"] = settings["layerwise_target_modules"]
        experiment_info["lora_alpha"] = settings["lora_alpha"]
        experiment_info["lora_dropout"] = settings["lora_dropout"]
        experiment_info["adalora_schedule"] = {
            "tinit_fraction": settings["tinit_fraction"],
            "tfinal_fraction": settings["tfinal_fraction"],
            "delta_fraction": settings["delta_fraction"],
        }
        experiment_info["dataset_stats"] = stats.to_dict()
        experiment_info["trainable_type"] = "dataset_aware_layerwise_adalora"

        return model, experiment_info

    raise ValueError(f"Unknown method: {args.method}")


def build_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )

        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )

        return {
            "accuracy": accuracy["accuracy"],
            "macro_f1": f1["f1"],
        }

    return compute_metrics


def save_experiment_info(output_dir, experiment_info):
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(
        output_dir,
        "experiment_info.json",
    )

    with open(path, "w") as file:
        json.dump(
            experiment_info,
            file,
            indent=2,
        )


def estimate_total_steps(train_dataset, batch_size, epochs):
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)

    return max(1, int(steps_per_epoch * epochs))


def main():
    args = parse_args()
    set_seed(args.seed)
    is_adalora_method = args.method in [
        "adalora",
        "dataset_aware_adalora",
        "dataset_aware_layerwise_adalora",
    ]

    train_dataset, eval_dataset = load_low_resource_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_step = estimate_total_steps(
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    num_labels = get_num_labels(
        train_dataset=train_dataset,
        label_column=args.label_column,
    )

    model, experiment_info = build_model(
        args=args,
        num_labels=num_labels,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        total_step=total_step,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenized_train_dataset = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
    )

    tokenized_eval_dataset = tokenize_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []

    if args.method in [
        "lora",
        "adalora",
        "dataset_aware_adalora",
        "dataset_aware_layerwise_adalora",
    ]:
        callbacks.append(StabilityAwareCallback(output_dir=args.output_dir))

    if is_adalora_method:
        callbacks.append(AdaLoraAllocationCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
    )

    train_result = trainer.train()

    metrics = trainer.evaluate()

    experiment_info["metrics"] = metrics
    experiment_info["train_metrics"] = train_result.metrics
    experiment_info["dataset_name"] = args.dataset_name
    experiment_info["dataset_config"] = args.dataset_config
    experiment_info["num_samples"] = args.num_samples
    experiment_info["model_name"] = args.model_name
    experiment_info["seed"] = args.seed
    experiment_info["estimated_total_steps"] = total_step

    save_experiment_info(
        output_dir=args.output_dir,
        experiment_info=experiment_info,
    )

    print(json.dumps(experiment_info, indent=2))


if __name__ == "__main__":
    main()
