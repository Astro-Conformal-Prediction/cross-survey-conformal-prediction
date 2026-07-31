import numpy as np

eps = 1e-8

def normalized_l2(translated, actual, eps=eps):
    """
   per sample ||actual - translated|| / (||actual|| + eps). 
    """
    translated = np.asarray(translated)
    actual = np.asarray(actual)
    num = np.linalg.norm(actual - translated, axis=1)
    denom = np.linalg.norm(actual, axis=1)
    return num / (denom + eps)


def cosine_residual(translated, actual, eps=eps):
    """
    Per-sample cosine residual between translated and actual embeddings. 
    """
    translated = np.asarray(translated)
    actual = np.asarray(actual)

    dot_product = np.sum(translated * actual, axis=1)
    translated_norm = np.linalg.norm(translated, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    return 1 - dot_product / ((translated_norm * actual_norm) + eps)


def residual_summary(
    translated,
    actual,
    prefix=""):
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    l2 = normalized_l2(translated, actual)
    cosine = cosine_residual(translated, actual)

    return {
        f"{prefix}mean_l2": float(np.mean(l2)),
        f"{prefix}median_l2": float(np.median(l2)),
        f"{prefix}p90_l2": float(np.percentile(l2, 90)),
        f"{prefix}mean_cosine_residual": float(np.mean(cosine))
    }


def pooled_residuals(
    residuals,
    prefix="" ):
    residuals = np.asarray(residuals, dtype=np.float32)

    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    return {
        f"{prefix}mean_l2": float(np.mean(residuals)),
        f"{prefix}median_l2": float(np.median(residuals)),
        f"{prefix}p90_l2": float(np.percentile(residuals, 90))
    }
