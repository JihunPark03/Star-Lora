import argparse
import json
import math
import os

import evaluate
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
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


def build_lora_config(rank):
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )


def build_adalora_config(init_rank, target_rank, total_step):
    tinit = max(0, int(0.1 * total_step))
    tfinal = max(0, int(0.5 * total_step))
    delta_t = max(1, int(0.02 * total_step))

    if tinit + tfinal >= total_step:
        tinit = 0
        tfinal = max(0, total_step - 1)

    return AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=init_rank,
        target_r=target_rank,
        lora_alpha=target_rank * 2,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        tinit=tinit,
        tfinal=tfinal,
        deltaT=delta_t,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=total_step,
    )


def build_model(args, num_labels, train_dataset, tokenizer, total_step):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    experiment_info = {
        "method": args.method,
        "base_rank": args.base_rank,
    }

    if args.method == "full":
        experiment_info["trainable_type"] = "full_finetuning"
        return model, experiment_info

    if args.method == "lora":
        rank = args.base_rank
        config = build_lora_config(rank)

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

        rank = compute_dataset_aware_rank(
            stats=stats,
            base_rank=args.base_rank,
            min_rank=2,
            max_rank=32,
        )

        init_rank = max(rank + 4, 4)

        config = build_adalora_config(
            init_rank=init_rank,
            target_rank=rank,
            total_step=total_step,
        )

        model = get_peft_model(model, config)

        experiment_info["rank"] = rank
        experiment_info["init_rank"] = init_rank
        experiment_info["dataset_stats"] = stats.to_dict()
        experiment_info["trainable_type"] = "dataset_aware_adalora"

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

    train_dataset, eval_dataset = load_low_resource_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
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
    ]:
        callbacks.append(StabilityAwareCallback(output_dir=args.output_dir))

    if args.method in [
        "adalora",
        "dataset_aware_adalora",
    ]:
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
