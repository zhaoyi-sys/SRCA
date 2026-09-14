from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


UNKNOWN_LABEL = "__unknown__"


def euclidean_knn_distance(query: np.ndarray, train: np.ndarray, k: int = 1, leave_one_out: bool = False) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")
    q = query.astype(np.float64)
    t = train.astype(np.float64)
    q2 = np.sum(q * q, axis=1, keepdims=True)
    t2 = np.sum(t * t, axis=1, keepdims=True).T
    dist2 = np.maximum(q2 + t2 - 2.0 * (q @ t.T), 0.0)
    if leave_one_out:
        if len(query) != len(train):
            raise ValueError("leave_one_out=True requires query and train to have the same length.")
        np.fill_diagonal(dist2, np.inf)
    k_eff = min(int(k), dist2.shape[1] - int(leave_one_out))
    if k_eff <= 0:
        return np.full(len(query), np.inf, dtype=np.float32)
    nearest2 = np.partition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(np.sqrt(nearest2), axis=1).astype(np.float32)


def true_class_thresholds(
    train_score: np.ndarray,
    y_train: np.ndarray,
    percentile: float,
    num_classes: int,
) -> np.ndarray:
    fallback_threshold = float(np.percentile(train_score, float(percentile)))
    thresholds = np.full(int(num_classes), fallback_threshold, dtype=np.float32)
    y_train = y_train.astype(np.int64)
    for class_idx in range(int(num_classes)):
        mask = y_train == class_idx
        if np.any(mask):
            thresholds[class_idx] = float(np.percentile(train_score[mask], float(percentile)))
    return thresholds


def evaluate_open_set(
    labels: Sequence[str],
    known_classes: Sequence[str],
    known_test_idx: np.ndarray,
    unknown_idx: np.ndarray,
    pred_known: np.ndarray,
    unknown_score: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    eval_idx = np.concatenate([known_test_idx, unknown_idx])
    pred_known_label = np.asarray([known_classes[int(i)] for i in pred_known.tolist()], dtype=object)
    threshold_used = thresholds[pred_known.astype(np.int64)]
    pred_unknown = unknown_score > threshold_used
    pred_open = np.where(pred_unknown, UNKNOWN_LABEL, pred_known_label)
    true_label = np.asarray([labels[int(i)] for i in eval_idx.tolist()], dtype=object)
    true_open = np.concatenate(
        [
            np.asarray([labels[int(i)] for i in known_test_idx.tolist()], dtype=object),
            np.full(len(unknown_idx), UNKNOWN_LABEL, dtype=object),
        ]
    )
    known_slice = slice(0, len(known_test_idx))
    unknown_slice = slice(len(known_test_idx), len(eval_idx))
    binary_true = (true_open == UNKNOWN_LABEL).astype(np.int64)
    binary_pred = (pred_open == UNKNOWN_LABEL).astype(np.int64)
    try:
        auroc = float(roc_auc_score(binary_true, unknown_score))
    except ValueError:
        auroc = math.nan
    unknown_precision, _, _, _ = precision_recall_fscore_support(
        binary_true,
        binary_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    metrics: Dict[str, object] = {
        "known_pre_reject_acc": float(np.mean(pred_known_label[known_slice] == true_open[known_slice])) if len(known_test_idx) else math.nan,
        "known_acc": float(accuracy_score(true_open[known_slice], pred_open[known_slice])) if len(known_test_idx) else math.nan,
        "known_f1": float(f1_score(true_open, pred_open, labels=list(known_classes), average="macro", zero_division=0)),
        "unknown_recall": float(np.mean(pred_open[unknown_slice] == UNKNOWN_LABEL)) if len(unknown_idx) else math.nan,
        "unknown_precision": float(unknown_precision),
        "unknown_f1": float(f1_score(true_open, pred_open, labels=[UNKNOWN_LABEL], average="macro", zero_division=0)),
        "open_macro_f1": float(f1_score(true_open, pred_open, labels=list(known_classes) + [UNKNOWN_LABEL], average="macro", zero_division=0)),
        "auroc": auroc,
    }
    pred_df = pd.DataFrame(
        {
            "sample_index": eval_idx.astype(int),
            "true_label": true_label,
            "true_open_label": true_open,
            "pred_known_label": pred_known_label,
            "pred_open_label": pred_open,
            "unknown_score": unknown_score.astype(float),
            "threshold": threshold_used.astype(float),
            "score_minus_threshold": (unknown_score - threshold_used).astype(float),
            "is_unknown_true": binary_true.astype(int),
            "is_unknown_pred": binary_pred.astype(int),
        }
    )
    return metrics, pred_df
