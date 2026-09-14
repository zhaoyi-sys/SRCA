from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .closed_set import fit_minmax_ridge, minmax_scale_for_rejection
from .datasets import load_dataset, sort_label_key
from .minirocket_features import fit_transform_minirocket_streams
from .rejection import euclidean_knn_distance, evaluate_open_set, true_class_thresholds
from .sphor import SpHORConfig, encode_normalized_z, encode_raw_h, seed_everything, train_sphor_linear
from .splits import split_leave_one_unknown


@dataclass
class MethodConfig:
    num_features_per_stream: int = 672
    max_dilations_per_kernel: int = 32
    train_fraction: float = 0.8
    threshold_percentiles: Tuple[float, ...] = (90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)
    closed_use_minmax: bool = True
    reject_use_minmax: bool = True
    closed_feature_scaler: str = "minmax"
    reject_feature_scaler: str = "minmax"
    reject_representation: str = "sphor_raw_h"
    reject_k: int = 1
    reject_ks: Tuple[int, ...] = ()
    sphor: SpHORConfig = SpHORConfig()


def _safe_label_for_filename(label: str) -> str:
    return str(label).replace("/", "_").replace("\\", "_").replace(" ", "_")


def run_fold(
    streams: Dict[str, np.ndarray],
    labels: Sequence[str],
    by_class: Dict[str, List[int]],
    unknown_label: str,
    seed: int,
    config: MethodConfig,
    device: torch.device,
    minirocket_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], List[pd.DataFrame], pd.DataFrame, Dict[str, object], Optional[Dict[str, object]]]:
    seed_everything(int(seed))
    train_idx, known_test_idx, unknown_idx, known_classes, class_stats = split_leave_one_unknown(
        by_class=by_class,
        unknown_label=unknown_label,
        train_fraction=float(config.train_fraction),
        seed=int(seed),
    )
    eval_idx = np.concatenate([known_test_idx, unknown_idx])
    class_to_int = {name: idx for idx, name in enumerate(known_classes)}
    y_train = np.asarray([class_to_int[labels[int(i)]] for i in train_idx.tolist()], dtype=np.int64)

    f_train_raw, f_eval_raw, stream_dims = fit_transform_minirocket_streams(
        streams=streams,
        train_idx=train_idx,
        eval_idx=eval_idx,
        num_features_per_stream=int(config.num_features_per_stream),
        max_dilations_per_kernel=int(config.max_dilations_per_kernel),
        minirocket_dir=minirocket_dir,
    )
    closed_feature_scaler = str(config.closed_feature_scaler)
    reject_feature_scaler = str(config.reject_feature_scaler)
    if closed_feature_scaler == "minmax" and not bool(config.closed_use_minmax):
        closed_feature_scaler = "none"
    if reject_feature_scaler == "minmax" and not bool(config.reject_use_minmax):
        reject_feature_scaler = "none"

    closed = fit_minmax_ridge(
        f_train_raw,
        f_eval_raw,
        y_train=y_train,
        seed=int(seed),
        use_minmax=bool(config.closed_use_minmax),
        scaler_name=closed_feature_scaler,
    )
    x_train_reject, x_eval_reject, _reject_scaler = minmax_scale_for_rejection(
        f_train_raw,
        f_eval_raw,
        use_minmax=bool(config.reject_use_minmax),
        scaler_name=reject_feature_scaler,
    )
    reject_representation = str(config.reject_representation).strip().lower()
    valid_representations = {"ridge_confidence", "minirocket_feature", "sphor_z", "sphor_raw_h"}
    if reject_representation not in valid_representations:
        raise ValueError(f"Unknown reject representation: {reject_representation!r}")
    history = pd.DataFrame()
    knn_reference: Optional[Dict[str, object]] = None
    if reject_representation in {"sphor_z", "sphor_raw_h"}:
        sphor_model, history = train_sphor_linear(
            x_train=x_train_reject,
            y_train=y_train,
            num_classes=len(known_classes),
            config=config.sphor,
            device=device,
            seed=int(seed),
        )
        if reject_representation == "sphor_z":
            reject_train = encode_normalized_z(
                sphor_model,
                x_train_reject,
                device=device,
                batch_size=int(config.sphor.eval_batch_size),
            )
            reject_eval = encode_normalized_z(
                sphor_model,
                x_eval_reject,
                device=device,
                batch_size=int(config.sphor.eval_batch_size),
            )
            reject_space = "normalized_z"
        else:
            reject_train = encode_raw_h(
                sphor_model,
                x_train_reject,
                device=device,
                batch_size=int(config.sphor.eval_batch_size),
            )
            reject_eval = encode_raw_h(
                sphor_model,
                x_eval_reject,
                device=device,
                batch_size=int(config.sphor.eval_batch_size),
            )
            reject_space = "raw_h"
        knn_reference = {
            "train_vectors_128": reject_train.astype(np.float32),
            "y_train_groundtruth": y_train.astype(np.int64),
            "train_sample_index": train_idx.astype(np.int64),
            "known_classes": np.asarray(known_classes, dtype=object),
            "reject_space": reject_space,
            "threshold_policy": "true_class_groundtruth",
        }
    elif reject_representation == "minirocket_feature":
        reject_train = x_train_reject
        reject_eval = x_eval_reject
        reject_space = "minirocket_feature"
    else:
        train_decision = np.asarray(closed.classifier.decision_function(closed.x_train_scaled), dtype=np.float32)
        eval_decision = np.asarray(closed.classifier.decision_function(closed.x_eval_scaled), dtype=np.float32)
        if train_decision.ndim == 1:
            train_confidence = np.abs(train_decision)
            eval_confidence = np.abs(eval_decision)
        else:
            train_confidence = np.max(train_decision, axis=1)
            eval_confidence = np.max(eval_decision, axis=1)
        reject_train = -train_confidence.astype(np.float32)
        reject_eval = -eval_confidence.astype(np.float32)
        reject_space = "negative_ridge_max_score"

    rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    reject_ks = tuple(int(k) for k in config.reject_ks) if config.reject_ks else (int(config.reject_k),)
    for reject_k in reject_ks:
        if reject_representation == "ridge_confidence":
            train_score = reject_train
            eval_score = reject_eval
            reject_distance = "negative_max_score"
            effective_k = 0
        else:
            train_score = euclidean_knn_distance(reject_train, reject_train, k=reject_k, leave_one_out=True)
            eval_score = euclidean_knn_distance(reject_eval, reject_train, k=reject_k, leave_one_out=False)
            reject_distance = f"euclidean_{reject_k}nn"
            effective_k = int(reject_k)
        for q in config.threshold_percentiles:
            thresholds = true_class_thresholds(
                train_score=train_score,
                y_train=y_train,
                percentile=float(q),
                num_classes=len(known_classes),
            )
            metrics, pred_df = evaluate_open_set(
                labels=labels,
                known_classes=known_classes,
                known_test_idx=known_test_idx,
                unknown_idx=unknown_idx,
                pred_known=closed.eval_pred,
                unknown_score=eval_score,
                thresholds=thresholds,
            )
            metrics.update(
                {
                    "seed": int(seed),
                    "heldout_unknown_label": str(unknown_label),
                    "threshold_percentile": float(q),
                    "threshold_group": "true_class",
                    "split": "random",
                    "train_count": int(len(train_idx)),
                    "known_test_count": int(len(known_test_idx)),
                    "unknown_test_count": int(len(unknown_idx)),
                    "feature_dim": int(f_train_raw.shape[1]),
                    "num_features_per_stream": int(config.num_features_per_stream),
                    "stream_dims": json.dumps(stream_dims, ensure_ascii=False),
                    "closed_set_classifier": "ridge",
                    "closed_feature_scaler": closed_feature_scaler,
                    "reject_feature_scaler": reject_feature_scaler,
                    "reject_representation": reject_representation,
                    "reject_space": reject_space,
                    "reject_distance": reject_distance,
                    "reject_k": int(effective_k),
                    "threshold_mean": float(np.mean(thresholds)),
                    "threshold_min": float(np.min(thresholds)),
                    "threshold_max": float(np.max(thresholds)),
                }
            )
            pred_df.insert(0, "threshold_group", "true_class")
            pred_df.insert(0, "threshold_percentile", float(q))
            pred_df.insert(0, "reject_k", int(effective_k))
            pred_df.insert(0, "heldout_unknown_label", str(unknown_label))
            pred_df.insert(0, "seed", int(seed))
            rows.append(metrics)
            pred_parts.append(pred_df)

    fold_meta = {
        "unknown_label": str(unknown_label),
        "known_classes": list(known_classes),
        "class_stats": class_stats,
        "stream_dims": stream_dims,
        "threshold_policy": "training_groundtruth_true_class",
        "split": "random",
    }
    return rows, pred_parts, history, fold_meta, knn_reference


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["reject_k", "threshold_percentile"] if "reject_k" in rows.columns else ["threshold_percentile"]
    if "threshold_group" in rows.columns and rows["threshold_group"].nunique(dropna=False) > 1:
        group_cols = ["threshold_group"] + group_cols
    metric_cols = ["known_pre_reject_acc", "known_acc", "known_f1", "unknown_recall", "unknown_precision", "unknown_f1", "open_macro_f1", "auroc"]
    return rows.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()


def run_dataset(
    dataset: str,
    dataset_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    config: MethodConfig,
    device: torch.device,
    label_file: str = "labels_corrected.npy",
    channel: int = 0,
    merge_async_motor: bool = False,
    input_mode: str = "ip",
    only_heldout_label: Optional[str] = None,
    minirocket_dir: Optional[Path] = None,
) -> None:
    streams, labels, meta, by_class = load_dataset(
        dataset=dataset,
        dataset_dir=dataset_dir,
        label_file=label_file,
        channel=channel,
        merge_async_motor=merge_async_motor,
        input_mode=input_mode,
    )
    sample_records = meta.pop("sample_records", None)
    heldout_labels = sorted(by_class.keys(), key=sort_label_key)
    if only_heldout_label is not None:
        target = str(only_heldout_label).strip().lower()
        if target not in by_class:
            raise ValueError(f"Unknown label {target!r}; available labels are {heldout_labels}")
        heldout_labels = [target]

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_index_df = pd.DataFrame(sample_records) if sample_records is not None else pd.DataFrame()
    if not sample_index_df.empty:
        sample_index_df.to_csv(output_dir / "sample_index.csv", index=False)
    all_rows: List[Dict[str, object]] = []
    all_preds: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, object]] = []
    total = len(seeds) * len(heldout_labels)
    done = 0
    print(f"Loaded {dataset}: samples={len(labels)}, classes={len(by_class)}, streams={list(streams.keys())}, device={device}")
    for seed in seeds:
        for unknown_label in heldout_labels:
            done += 1
            print(f"[{done}/{total}] seed={seed} heldout={unknown_label}")
            rows, preds, history, fold_meta, knn_reference = run_fold(
                streams=streams,
                labels=labels,
                by_class=by_class,
                unknown_label=unknown_label,
                seed=int(seed),
                config=config,
                device=device,
                minirocket_dir=minirocket_dir,
            )
            all_rows.extend(rows)
            all_preds.extend(preds)
            history.to_csv(output_dir / f"history_seed{seed}_unknown_{unknown_label}.csv", index=False)
            if knn_reference is not None:
                reference_path = output_dir / f"knn_reference_seed{seed}_unknown_{_safe_label_for_filename(unknown_label)}.npz"
                np.savez_compressed(
                    reference_path,
                    train_vectors_128=knn_reference["train_vectors_128"],
                    y_train_groundtruth=knn_reference["y_train_groundtruth"],
                    train_sample_index=knn_reference["train_sample_index"],
                    known_classes=knn_reference["known_classes"],
                    reject_space=np.asarray([knn_reference["reject_space"]], dtype=object),
                    threshold_policy=np.asarray([knn_reference["threshold_policy"]], dtype=object),
                )
                fold_meta["knn_reference_path"] = str(reference_path)
            meta_rows.append({"seed": int(seed), **fold_meta})
            best = max(rows, key=lambda r: float(r["unknown_f1"]))
            print(
                "  best_by_unknown "
                f"q={best['threshold_percentile']:.0f} known_f1={best['known_f1']:.4f} "
                f"unknown_f1={best['unknown_f1']:.4f} open_macro_f1={best['open_macro_f1']:.4f} "
                f"auroc={best['auroc']:.4f}"
            )

    rows_df = pd.DataFrame(all_rows)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not preds_df.empty and not sample_index_df.empty:
        preds_df = preds_df.merge(sample_index_df, on="sample_index", how="left", suffixes=("", "_source"))
    summary_df = summarize(rows_df)
    rows_df.to_csv(output_dir / "leave_one_unknown_summary.csv", index=False)
    preds_df.to_csv(output_dir / "predictions.csv", index=False)
    summary_df.to_csv(output_dir / "setting_summary.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        closed_scaler_name = "MinMaxScaler" if config.closed_use_minmax else "no scaler"
        reject_scaler_name = "MinMaxScaler" if config.reject_use_minmax else "no scaler"
        json.dump(
            {
                "method": (
                    f"MiniROCKET -> {closed_scaler_name} -> RidgeClassifier; "
                    f"rejection={config.reject_representation}; reject_scaler={reject_scaler_name}; "
                    "groundtruth class thresholds"
                ),
                "dataset_meta": meta,
                "input_mode": input_mode,
                "config": asdict(config),
                "seeds": [int(v) for v in seeds],
                "heldout_labels": heldout_labels,
                "threshold_policy": "training_groundtruth_true_class",
                "knn_reference": "For SpHOR rejection, each fold saves train_vectors_128 and y_train_groundtruth in knn_reference_seed*_unknown_*.npz.",
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {output_dir}")
