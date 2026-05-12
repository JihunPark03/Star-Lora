import math
from collections import Counter

import numpy as np
import torch


class DatasetStats:
    def __init__(
        self,
        num_samples,
        class_imbalance,
        token_sparsity,
        token_frequency_skew,
        label_entropy,
        embedding_variance,
    ):
        self.num_samples = num_samples
        self.class_imbalance = class_imbalance
        self.token_sparsity = token_sparsity
        self.token_frequency_skew = token_frequency_skew
        self.label_entropy = label_entropy
        self.embedding_variance = embedding_variance

    def to_dict(self):
        return {
            "num_samples": self.num_samples,
            "class_imbalance": self.class_imbalance,
            "token_sparsity": self.token_sparsity,
            "token_frequency_skew": self.token_frequency_skew,
            "label_entropy": self.label_entropy,
            "embedding_variance": self.embedding_variance,
        }


def compute_label_entropy(labels):
    counter = Counter(labels)
    total = len(labels)

    entropy = 0.0

    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12)

    max_entropy = math.log(len(counter) + 1e-12)

    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy


def compute_class_imbalance(labels):
    counter = Counter(labels)
    counts = np.array(list(counter.values()), dtype=np.float32)

    max_count = counts.max()
    min_count = counts.min()

    return float(1.0 - (min_count / (max_count + 1e-12)))


def compute_token_features(dataset, tokenizer, text_column, max_samples=1000):
    token_counter = Counter()
    used_samples = min(len(dataset), max_samples)

    for i in range(used_samples):
        text = dataset[i][text_column]

        token_ids = tokenizer(
            text,
            truncation=True,
            max_length=256,
            add_special_tokens=False,
        )["input_ids"]

        token_counter.update(token_ids)

    counts = np.array(list(token_counter.values()), dtype=np.float32)

    if len(counts) == 0:
        return 0.0, 0.0

    token_sparsity = float(np.mean(counts <= 1))
    token_frequency_skew = float(np.std(counts) / (np.mean(counts) + 1e-12))

    return token_sparsity, token_frequency_skew


def compute_embedding_variance(
    dataset,
    tokenizer,
    model,
    text_column,
    max_samples=256,
):
    device = next(model.parameters()).device
    model.eval()

    vectors = []
    used_samples = min(len(dataset), max_samples)

    with torch.no_grad():
        for i in range(used_samples):
            text = dataset[i][text_column]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            input_ids = inputs["input_ids"].to(device)
            embeddings = model.get_input_embeddings()(input_ids)

            pooled_embedding = embeddings.mean(dim=1).squeeze(0)
            vectors.append(pooled_embedding.detach().float().cpu().numpy())

    if len(vectors) == 0:
        return 0.0

    vectors = np.stack(vectors)
    variance = np.mean(np.var(vectors, axis=0))

    return float(variance)


def analyze_dataset(
    dataset,
    tokenizer,
    model,
    text_column,
    label_column,
):
    labels = [dataset[i][label_column] for i in range(len(dataset))]

    class_imbalance = compute_class_imbalance(labels)
    label_entropy = compute_label_entropy(labels)

    token_sparsity, token_frequency_skew = compute_token_features(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column,
    )

    embedding_variance = compute_embedding_variance(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        text_column=text_column,
    )

    return DatasetStats(
        num_samples=len(dataset),
        class_imbalance=class_imbalance,
        token_sparsity=token_sparsity,
        token_frequency_skew=token_frequency_skew,
        label_entropy=label_entropy,
        embedding_variance=embedding_variance,
    )
