"""
Evaluation metrics for classification, language modeling, and generation.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def calculate_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Calculate classification metrics: Accuracy, Macro/Micro Precision,
    Recall, F1-Score, and Confusion Matrix.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if labels is None:
        unique_labels = sorted(list(set(y_true_arr) | set(y_pred_arr)))
    else:
        unique_labels = labels

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for yt, yp in zip(y_true_arr, y_pred_arr):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    total_samples = len(y_true_arr)
    accuracy = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels": unique_labels,
        "confusion_matrix": cm.tolist()
    }


def calculate_perplexity(loss_or_neg_log_prob: float, is_loss: bool = True) -> float:
    """
    Calculate Perplexity from Cross-Entropy loss or negative log probability.
    Perplexity = exp(loss).
    """
    try:
        val = loss_or_neg_log_prob if is_loss else -loss_or_neg_log_prob
        return float(math.exp(val))
    except OverflowError:
        return float("inf")


def calculate_bleu_score(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    max_n: int = 4
) -> float:
    """
    Compute BLEU score with modified n-gram precision and brevity penalty.
    """
    if not hypothesis or not reference:
        return 0.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / hyp_len)) if hyp_len > 0 else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        if hyp_len < n:
            break
        hyp_ngrams = [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)]
        ref_ngrams = [tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)]

        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_matches = sum(min(count, ref_counts[ng]) for ng, count in hyp_counts.items())
        total_ngrams = len(hyp_ngrams)

        p_n = clipped_matches / total_ngrams if total_ngrams > 0 else 1e-9
        precisions.append(max(p_n, 1e-9))

    if not precisions:
        return 0.0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return float(bp * geometric_mean)
