from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def import_minirocket(minirocket_dir: Optional[Path] = None):
    candidate_dirs: List[Path] = []
    if minirocket_dir is not None:
        candidate_dirs.append(minirocket_dir.resolve())
    candidate_dirs.append(Path(__file__).resolve().parents[1])
    for directory in candidate_dirs:
        if directory.exists() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module = importlib.import_module("minirocket")
    return module.fit, module.transform


def fit_transform_minirocket_streams(
    streams: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    num_features_per_stream: int,
    max_dilations_per_kernel: int,
    minirocket_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    fit, transform = import_minirocket(minirocket_dir)
    train_parts: List[np.ndarray] = []
    eval_parts: List[np.ndarray] = []
    stream_dims: Dict[str, int] = {}
    for stream_name, values in streams.items():
        x_train = values[train_idx].astype(np.float32)
        x_eval = values[eval_idx].astype(np.float32)
        params = fit(
            x_train,
            num_features=int(num_features_per_stream),
            max_dilations_per_kernel=int(max_dilations_per_kernel),
        )
        f_train = transform(x_train, params).astype(np.float32)
        f_eval = transform(x_eval, params).astype(np.float32)
        train_parts.append(f_train)
        eval_parts.append(f_eval)
        stream_dims[stream_name] = int(f_train.shape[1])
    return np.concatenate(train_parts, axis=1), np.concatenate(eval_parts, axis=1), stream_dims
