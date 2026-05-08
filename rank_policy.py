import math


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def size_scaling(num_samples, reference_size=5000):
    """
    f(N)

    Small dataset  -> smaller rank
    Large dataset  -> larger rank
    """
    value = math.log(1 + num_samples) / math.log(1 + reference_size)
    return clamp(value, 0.3, 1.5)


def distribution_scaling(
    class_imbalance,
    token_sparsity,
    token_frequency_skew,
):
    """
    g_dist

    If the dataset is imbalanced or sparse,
    rank should be reduced to avoid overfitting.
    """
    normalized_skew = min(token_frequency_skew, 1.0)

    penalty = (
        0.5 * class_imbalance
        + 0.3 * token_sparsity
        + 0.2 * normalized_skew
    )

    scale = 1.0 - 0.5 * penalty

    return clamp(scale, 0.5, 1.0)


def complexity_scaling(
    label_entropy,
    embedding_variance,
):
    """
    g_comp

    If the task looks complex,
    rank should be increased.
    """
    normalized_variance = min(embedding_variance, 1.0)

    scale = (
        1.0
        + 0.5 * label_entropy
        + 0.3 * normalized_variance
    )

    return clamp(scale, 1.0, 2.0)


def compute_dataset_aware_rank(
    stats,
    base_rank=8,
    min_rank=2,
    max_rank=32,
):
    f_size = size_scaling(stats.num_samples)

    g_dist = distribution_scaling(
        class_imbalance=stats.class_imbalance,
        token_sparsity=stats.token_sparsity,
        token_frequency_skew=stats.token_frequency_skew,
    )

    g_comp = complexity_scaling(
        label_entropy=stats.label_entropy,
        embedding_variance=stats.embedding_variance,
    )

    raw_rank = base_rank * f_size * g_dist * g_comp
    rank = int(round(raw_rank))

    return clamp(rank, min_rank, max_rank)
