from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .datasets import sort_label_key


def split_feature_matrix(streams: Dict[str, np.ndarray]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for x in streams.values():
        x = x.astype(np.float32)
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        min_v = np.min(x, axis=1)
        max_v = np.max(x, axis=1)
        ptp = max_v - min_v
        rms = np.sqrt(np.mean(x * x, axis=1))
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=1)
        parts.append(np.stack([mean, std, min_v, max_v, ptp, rms, q25, q50, q75], axis=1).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _block_query_from_order(order: np.ndarray, n_query: int, split_offset: int) -> np.ndarray:
    start = (int(split_offset) % 5) * n_query
    start = min(start, len(order) - n_query)
    return order[start : start + n_query]


def split_leave_one_unknown(
    streams: Dict[str, np.ndarray],
    by_class: Dict[str, List[int]],
    unknown_label: str,
    train_fraction: float,
    seed: int,
    split_strategy: str,
    split_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    split_feats = split_feature_matrix(streams)
    train_idx: List[int] = []
    known_test_idx: List[int] = []
    unknown_idx = np.asarray(by_class[unknown_label], dtype=np.int64)
    known_classes: List[str] = []
    class_stats: Dict[str, Dict[str, object]] = {}

    for class_name in sorted(by_class.keys(), key=sort_label_key):
        idxs = np.asarray(by_class[class_name], dtype=np.int64)
        if class_name == unknown_label:
            class_stats[class_name] = {
                "role": "unknown_test_only",
                "total": int(len(idxs)),
                "train": 0,
                "test": int(len(idxs)),
            }
            continue

        n_train = max(2, min(len(idxs) - 1, int(round(len(idxs) * float(train_fraction)))))
        n_test = len(idxs) - n_train
        if split_strategy == "random":
            local_train = rng.choice(np.arange(len(idxs)), size=n_train, replace=False)
        elif split_strategy == "index_block":
            local_query = _block_query_from_order(np.arange(len(idxs), dtype=np.int64), n_test, split_offset)
            mask = np.ones(len(idxs), dtype=bool)
            mask[local_query] = False
            local_train = np.flatnonzero(mask)
        elif split_strategy == "index_stride":
            fold_count = max(2, int(round(1.0 / max(1e-6, 1.0 - float(train_fraction)))))
            local_query = np.flatnonzero((np.arange(len(idxs)) % fold_count) == (int(split_offset) % fold_count))
            if len(local_query) > n_test:
                local_query = local_query[:n_test]
            elif len(local_query) < n_test:
                remaining = np.setdiff1d(np.arange(len(idxs)), local_query, assume_unique=True)
                local_query = np.concatenate([local_query, remaining[: n_test - len(local_query)]])
            mask = np.ones(len(idxs), dtype=bool)
            mask[local_query] = False
            local_train = np.flatnonzero(mask)
        else:
            feats = split_feats[idxs].astype(np.float32)
            feats = (feats - np.mean(feats, axis=0, keepdims=True)) / (np.std(feats, axis=0, keepdims=True) + 1e-6)
            centroid = np.mean(feats, axis=0, keepdims=True)
            dist = np.linalg.norm(feats - centroid, axis=1)
            order = np.argsort(dist, kind="mergesort")
            if split_strategy == "centroid_block":
                local_query = _block_query_from_order(order, n_test, split_offset)
                mask = np.ones(len(idxs), dtype=bool)
                mask[local_query] = False
                local_train = np.flatnonzero(mask)
            elif split_strategy == "nn_block":
                pair_dist = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
                np.fill_diagonal(pair_dist, np.inf)
                nn_order = np.argsort(np.min(pair_dist, axis=1), kind="mergesort")
                local_query = _block_query_from_order(nn_order, n_test, split_offset)
                mask = np.ones(len(idxs), dtype=bool)
                mask[local_query] = False
                local_train = np.flatnonzero(mask)
            else:
                raise ValueError(f"Unknown split_strategy={split_strategy!r}")

        mask = np.zeros(len(idxs), dtype=bool)
        mask[local_train] = True
        train_idx.extend(int(v) for v in idxs[mask].tolist())
        known_test_idx.extend(int(v) for v in idxs[~mask].tolist())
        known_classes.append(class_name)
        class_stats[class_name] = {
            "role": "known",
            "total": int(len(idxs)),
            "train": int(np.sum(mask)),
            "test": int(np.sum(~mask)),
            "split_strategy": split_strategy,
            "split_offset": int(split_offset),
        }

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(known_test_idx, dtype=np.int64),
        unknown_idx,
        known_classes,
        class_stats,
    )

