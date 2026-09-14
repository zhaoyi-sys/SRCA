from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .datasets import sort_label_key


def split_leave_one_unknown(
    by_class: Dict[str, List[int]],
    unknown_label: str,
    train_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
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
        local_train = rng.choice(np.arange(len(idxs)), size=n_train, replace=False)
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
            "split": "random",
        }

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(known_test_idx, dtype=np.int64),
        unknown_idx,
        known_classes,
        class_stats,
    )
