from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nilm_clean_open.datasets import load_dataset, make_class_index, sort_label_key
from nilm_clean_open.minirocket_features import import_minirocket
from nilm_clean_open.rejection import UNKNOWN_LABEL, euclidean_knn_distance, evaluate_open_set, true_class_thresholds
from nilm_clean_open.sphor import LinearSpHORNet, SpHORConfig, encode_raw_h, seed_everything, train_sphor_linear
from nilm_clean_open.splits import split_leave_one_unknown


DEFAULT_LILAC_DIR = Path(r"E:\研究生阶段\NILM\工业负荷识别\数据集\lilac\agr")
DEFAULT_ILSED_DIR = Path(r"E:\研究生阶段\NILM\工业负荷识别\数据集\316数据集")


@dataclass
class SRCAConfig:
    dataset: str
    dataset_dir: Path
    output_dir: Path
    label_file: str = "labels_corrected.npy"
    input_mode: str = "ip"
    channel: int = 0
    merge_async_motor: bool = False
    seed: int = 42
    train_fraction: float = 0.8
    split_strategy: str = "random"
    split_offset: int = 1
    unknown_label: Optional[str] = None
    num_features_per_stream: int = 672
    max_dilations_per_kernel: int = 32
    closed_feature_scaler: str = "minmax"
    reject_feature_scaler: str = "minmax"
    embedding_dim: int = 128
    sphor_epochs: int = 100
    sphor_lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    eval_batch_size: int = 1024
    temperature: float = 0.10
    ortho_weight: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 1.0
    grad_clip: float = 5.0
    reject_k: int = 3
    threshold_percentile: float = 93.0
    device: str = "auto"
    minirocket_dir: Optional[Path] = None


def resolve_dataset_name(name: str) -> str:
    text = str(name).strip().lower()
    if text in {"ilsed", "mine"}:
        return "single_device_mat"
    if text == "lilac":
        return "lilac"
    raise ValueError("dataset must be one of: lilac, ilsed, mine.")


def default_dataset_dir(name: str) -> Path:
    text = str(name).strip().lower()
    if text == "lilac":
        return DEFAULT_LILAC_DIR
    if text in {"ilsed", "mine"}:
        return DEFAULT_ILSED_DIR
    raise ValueError(name)


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def scaler_from_name(name: str):
    text = str(name).strip().lower()
    if text == "none":
        return None
    if text == "minmax":
        return MinMaxScaler()
    if text == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler: {name!r}")


def fit_transform_scaler(x_train: np.ndarray, scaler_name: str) -> Tuple[np.ndarray, Any]:
    scaler = scaler_from_name(scaler_name)
    if scaler is None:
        return x_train.astype(np.float32, copy=False), None
    return scaler.fit_transform(x_train).astype(np.float32), scaler


def transform_scaler(x: np.ndarray, scaler: Any) -> np.ndarray:
    if scaler is None:
        return x.astype(np.float32, copy=False)
    return scaler.transform(x).astype(np.float32)


def fit_minirocket_reference(
    streams: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    num_features_per_stream: int,
    max_dilations_per_kernel: int,
    minirocket_dir: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, int]]:
    fit, transform = import_minirocket(minirocket_dir)
    parts: List[np.ndarray] = []
    params_by_stream: Dict[str, Any] = {}
    stream_dims: Dict[str, int] = {}
    for stream_name, values in streams.items():
        x_train = values[train_idx].astype(np.float32)
        params = fit(
            x_train,
            num_features=int(num_features_per_stream),
            max_dilations_per_kernel=int(max_dilations_per_kernel),
        )
        features = transform(x_train, params).astype(np.float32)
        parts.append(features)
        params_by_stream[stream_name] = params
        stream_dims[stream_name] = int(features.shape[1])
    return np.concatenate(parts, axis=1).astype(np.float32), params_by_stream, stream_dims


def transform_minirocket(
    streams: Dict[str, np.ndarray],
    sample_idx: np.ndarray,
    params_by_stream: Dict[str, Any],
    minirocket_dir: Optional[Path],
) -> np.ndarray:
    _fit, transform = import_minirocket(minirocket_dir)
    parts: List[np.ndarray] = []
    for stream_name, params in params_by_stream.items():
        values = streams[stream_name][sample_idx].astype(np.float32)
        parts.append(transform(values, params).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def make_sphor_config(config: SRCAConfig) -> SpHORConfig:
    return SpHORConfig(
        embedding_dim=int(config.embedding_dim),
        epochs=int(config.sphor_epochs),
        lr=float(config.sphor_lr),
        weight_decay=float(config.weight_decay),
        batch_size=int(config.batch_size),
        eval_batch_size=int(config.eval_batch_size),
        temperature=float(config.temperature),
        ortho_weight=float(config.ortho_weight),
        label_smoothing=float(config.label_smoothing),
        mixup_alpha=float(config.mixup_alpha),
        grad_clip=float(config.grad_clip),
    )


def artifact_dir(output_dir: Path, dataset_label: str, unknown_label: str, seed: int) -> Path:
    safe_unknown = str(unknown_label).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return Path(output_dir) / f"{dataset_label}_unknown_{safe_unknown}_seed{int(seed)}"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def write_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_config_dataset(config: SRCAConfig):
    dataset_name = resolve_dataset_name(config.dataset)
    streams, labels, meta, by_class = load_dataset(
        dataset=dataset_name,
        dataset_dir=Path(config.dataset_dir),
        label_file=str(config.label_file),
        channel=int(config.channel),
        merge_async_motor=bool(config.merge_async_motor),
        input_mode=str(config.input_mode),
    )
    return streams, np.asarray(labels, dtype=object), meta, by_class


def train_one_unknown(config: SRCAConfig, unknown_label: str) -> Path:
    seed_everything(int(config.seed))
    device = resolve_device(config.device)
    streams, labels, meta, by_class_original = load_config_dataset(config)
    if unknown_label not in by_class_original:
        raise KeyError(f"Unknown label {unknown_label!r} is not in dataset labels: {sorted(by_class_original)}")

    open_labels = np.asarray([UNKNOWN_LABEL if label == unknown_label else label for label in labels], dtype=object)
    by_class = make_class_index(open_labels)
    train_idx, known_test_idx, unknown_idx, known_classes, class_stats = split_leave_one_unknown(
        streams={key: streams[key] for key in streams if key in {"i", "p", "v"}},
        by_class=by_class,
        unknown_label=UNKNOWN_LABEL,
        train_fraction=float(config.train_fraction),
        seed=int(config.seed),
        split_strategy=str(config.split_strategy),
        split_offset=int(config.split_offset),
    )
    class_to_int = {name: idx for idx, name in enumerate(known_classes)}
    y_train = np.asarray([class_to_int[open_labels[int(i)]] for i in train_idx], dtype=np.int64)

    f_train_raw, minirocket_params, stream_dims = fit_minirocket_reference(
        streams=streams,
        train_idx=train_idx,
        num_features_per_stream=int(config.num_features_per_stream),
        max_dilations_per_kernel=int(config.max_dilations_per_kernel),
        minirocket_dir=config.minirocket_dir,
    )
    x_closed_train, closed_scaler = fit_transform_scaler(f_train_raw, config.closed_feature_scaler)
    ridge = RidgeClassifier(class_weight="balanced", random_state=int(config.seed))
    ridge.fit(x_closed_train, y_train)

    x_reject_train, reject_scaler = fit_transform_scaler(f_train_raw, config.reject_feature_scaler)
    sphor_config = make_sphor_config(config)
    sphor_model, history = train_sphor_linear(
        x_train=x_reject_train,
        y_train=y_train,
        num_classes=len(known_classes),
        config=sphor_config,
        device=device,
        seed=int(config.seed),
    )
    h_train_128 = encode_raw_h(
        sphor_model,
        x_reject_train,
        device=device,
        batch_size=int(config.eval_batch_size),
    ).astype(np.float32)
    train_score = euclidean_knn_distance(h_train_128, h_train_128, k=int(config.reject_k), leave_one_out=True)
    thresholds = true_class_thresholds(
        train_score=train_score,
        y_train=y_train,
        percentile=float(config.threshold_percentile),
        num_classes=len(known_classes),
    )

    out_dir = artifact_dir(Path(config.output_dir), str(config.dataset).lower(), unknown_label, int(config.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "knn_reference_train_h128.npz",
        h_train_128=h_train_128.astype(np.float32),
        y_train_groundtruth=y_train.astype(np.int64),
        train_sample_index=train_idx.astype(np.int64),
        known_classes=np.asarray(known_classes, dtype=object),
        thresholds=thresholds.astype(np.float32),
        train_knn_score=train_score.astype(np.float32),
    )
    write_pickle(
        out_dir / "model_artifacts.pkl",
        {
            "minirocket_params": minirocket_params,
            "closed_scaler": closed_scaler,
            "reject_scaler": reject_scaler,
            "ridge": ridge,
            "sphor_state_dict": {k: v.detach().cpu() for k, v in sphor_model.state_dict().items()},
            "sphor_input_dim": int(x_reject_train.shape[1]),
            "sphor_embedding_dim": int(config.embedding_dim),
            "num_known_classes": int(len(known_classes)),
        },
    )
    history.to_csv(out_dir / "sphor_training_history.csv", index=False, encoding="utf-8-sig")
    split_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(labels), dtype=np.int64),
            "original_label": labels,
            "open_label": open_labels,
            "role": "unused",
        }
    )
    split_df.loc[train_idx, "role"] = "train_known"
    split_df.loc[known_test_idx, "role"] = "test_known"
    split_df.loc[unknown_idx, "role"] = "test_unknown"
    split_df.to_csv(out_dir / "split.csv", index=False, encoding="utf-8-sig")
    write_json(
        out_dir / "train_manifest.json",
        {
            "dataset_meta": meta,
            "config": asdict(config),
            "unknown_label": unknown_label,
            "known_classes": list(known_classes),
            "class_to_int": class_to_int,
            "class_stats": class_stats,
            "stream_dims": stream_dims,
            "threshold_policy": "true_class_groundtruth_training_labels",
            "test_reference_policy": "store only training h_train_128 vectors and y_train_groundtruth for kNN reference",
            "no_training_prediction_for_thresholds": True,
        },
    )
    return out_dir


def load_trained_sphor(artifacts: Dict[str, Any], device: torch.device) -> LinearSpHORNet:
    model = LinearSpHORNet(
        input_dim=int(artifacts["sphor_input_dim"]),
        embedding_dim=int(artifacts["sphor_embedding_dim"]),
        num_classes=int(artifacts["num_known_classes"]),
    ).to(device)
    model.load_state_dict(artifacts["sphor_state_dict"])
    model.eval()
    return model


def test_artifact(artifact_path: Path, device_text: str = "auto", minirocket_dir: Optional[Path] = None) -> Dict[str, Any]:
    artifact_path = Path(artifact_path)
    manifest = json.loads((artifact_path / "train_manifest.json").read_text(encoding="utf-8"))
    cfg_raw = manifest["config"]
    cfg_raw["dataset_dir"] = Path(cfg_raw["dataset_dir"])
    cfg_raw["output_dir"] = Path(cfg_raw["output_dir"])
    if cfg_raw.get("minirocket_dir") is not None:
        cfg_raw["minirocket_dir"] = Path(cfg_raw["minirocket_dir"])
    config = SRCAConfig(**cfg_raw)
    if minirocket_dir is not None:
        config.minirocket_dir = minirocket_dir

    device = resolve_device(device_text if device_text != "from_config" else config.device)
    streams, labels, _meta, _by_class = load_config_dataset(config)
    split_df = pd.read_csv(artifact_path / "split.csv", encoding="utf-8-sig")
    known_test_idx = split_df.loc[split_df["role"].eq("test_known"), "sample_index"].to_numpy(dtype=np.int64)
    unknown_idx = split_df.loc[split_df["role"].eq("test_unknown"), "sample_index"].to_numpy(dtype=np.int64)
    eval_idx = np.concatenate([known_test_idx, unknown_idx])

    reference = np.load(artifact_path / "knn_reference_train_h128.npz", allow_pickle=True)
    h_train_128 = reference["h_train_128"].astype(np.float32)
    y_train_groundtruth = reference["y_train_groundtruth"].astype(np.int64)
    thresholds = reference["thresholds"].astype(np.float32)
    known_classes = [str(v) for v in reference["known_classes"].tolist()]

    artifacts = read_pickle(artifact_path / "model_artifacts.pkl")
    f_eval_raw = transform_minirocket(streams, eval_idx, artifacts["minirocket_params"], config.minirocket_dir)
    x_closed_eval = transform_scaler(f_eval_raw, artifacts["closed_scaler"])
    pred_known = artifacts["ridge"].predict(x_closed_eval).astype(np.int64)
    x_reject_eval = transform_scaler(f_eval_raw, artifacts["reject_scaler"])
    sphor_model = load_trained_sphor(artifacts, device)
    h_eval_128 = encode_raw_h(
        sphor_model,
        x_reject_eval,
        device=device,
        batch_size=int(config.eval_batch_size),
    ).astype(np.float32)
    eval_score = euclidean_knn_distance(h_eval_128, h_train_128, k=int(config.reject_k), leave_one_out=False)
    metrics, pred_df = evaluate_open_set(
        labels=split_df["open_label"].astype(str).tolist(),
        known_classes=known_classes,
        known_test_idx=known_test_idx,
        unknown_idx=unknown_idx,
        pred_known=pred_known,
        unknown_score=eval_score,
        thresholds=thresholds,
    )
    pred_df.insert(0, "unknown_label", manifest["unknown_label"])
    pred_df.insert(0, "seed", int(config.seed))
    pred_df.to_csv(artifact_path / "test_predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        **metrics,
        "seed": int(config.seed),
        "unknown_label": manifest["unknown_label"],
        "reject_k": int(config.reject_k),
        "threshold_percentile": float(config.threshold_percentile),
        "threshold_policy": "true_class_groundtruth_training_labels",
        "knn_reference_vectors": int(h_train_128.shape[0]),
        "knn_reference_dim": int(h_train_128.shape[1]),
        "uses_training_groundtruth_labels": True,
        "does_not_repredict_training_classes": True,
        "groundtruth_label_checksum": int(np.sum(y_train_groundtruth)),
    }
    write_json(artifact_path / "test_metrics.json", result)
    pd.DataFrame([result]).to_csv(artifact_path / "test_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def train_dataset(config: SRCAConfig) -> List[Path]:
    streams, labels, _meta, by_class = load_config_dataset(config)
    del streams, labels
    unknown_labels = [config.unknown_label] if config.unknown_label else sorted(by_class.keys(), key=sort_label_key)
    return [train_one_unknown(config, str(label)) for label in unknown_labels]

