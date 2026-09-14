from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class ClosedSetResult:
    scaler: Optional[object]
    classifier: RidgeClassifier
    x_train_scaled: np.ndarray
    x_eval_scaled: np.ndarray
    eval_pred: np.ndarray


def _scale_features(
    x_train_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
    scaler_name = str(scaler_name).strip().lower()
    if scaler_name == "none":
        return x_train_raw.astype(np.float32, copy=False), x_eval_raw.astype(np.float32, copy=False), None
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown feature scaler: {scaler_name!r}")
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_eval = scaler.transform(x_eval_raw).astype(np.float32)
    return x_train, x_eval, scaler


def fit_minmax_ridge(
    x_train_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    use_minmax: bool = True,
    scaler_name: Optional[str] = None,
) -> ClosedSetResult:
    if scaler_name is None:
        scaler_name = "minmax" if use_minmax else "none"
    x_train, x_eval, scaler = _scale_features(x_train_raw, x_eval_raw, scaler_name=scaler_name)
    classifier = RidgeClassifier(class_weight="balanced", random_state=int(seed))
    classifier.fit(x_train, y_train)
    eval_pred = classifier.predict(x_eval).astype(np.int64)
    return ClosedSetResult(
        scaler=scaler,
        classifier=classifier,
        x_train_scaled=x_train,
        x_eval_scaled=x_eval,
        eval_pred=eval_pred,
    )


def minmax_scale_for_rejection(
    x_train_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    use_minmax: bool = True,
    scaler_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
    if scaler_name is None:
        scaler_name = "minmax" if use_minmax else "none"
    return _scale_features(x_train_raw, x_eval_raw, scaler_name=scaler_name)
