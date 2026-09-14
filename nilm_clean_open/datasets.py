from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat


ASYNC_SOURCE_LABELS = {"1-phase-async-motor-11.7a", "1-phase-async-motor-2.5a"}
ASYNC_TARGET_LABEL = "1-phase-async-motor"


def normalize_label(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().lower()


def make_class_index(labels: Sequence[str]) -> Dict[str, List[int]]:
    by_class: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[str(label)].append(int(idx))
    return dict(by_class)


def sort_label_key(label: str) -> object:
    text = str(label)
    return int(text) if text.isdigit() else text


def extract_signal_channel(signal: np.ndarray, channel: int = 0, name: str = "signal") -> np.ndarray:
    signal = signal.astype(np.float32)
    if signal.ndim == 2:
        return signal
    if signal.ndim != 3:
        raise ValueError(f"{name} must have shape (N,L), (N,C,L), or (N,L,C), got {signal.shape}.")
    if signal.shape[1] <= 8 and signal.shape[2] > signal.shape[1]:
        if not 0 <= channel < signal.shape[1]:
            raise ValueError(f"channel={channel} is out of range for {name} shape {signal.shape}.")
        return signal[:, channel, :].astype(np.float32)
    if signal.shape[2] <= 8 and signal.shape[1] > signal.shape[2]:
        if not 0 <= channel < signal.shape[2]:
            raise ValueError(f"channel={channel} is out of range for {name} shape {signal.shape}.")
        return signal[:, :, channel].astype(np.float32)
    raise ValueError(f"Cannot infer channel dimension for {name}; got {signal.shape}.")


def _find_label_file(dataset_dir: Path, preferred_name: str) -> Path:
    preferred = dataset_dir / preferred_name
    fallback = dataset_dir / "labels.npy"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing labels file: {preferred} or {fallback}")


def merge_async_motor_labels(labels: Sequence[str]) -> Tuple[List[str], Dict[str, object]]:
    merged = [ASYNC_TARGET_LABEL if str(label) in ASYNC_SOURCE_LABELS else str(label) for label in labels]
    return merged, {
        "label_merge": {
            "enabled": True,
            "source_labels": sorted(ASYNC_SOURCE_LABELS),
            "target_label": ASYNC_TARGET_LABEL,
            "num_classes_before_merge": int(len(set(labels))),
            "num_classes_after_merge": int(len(set(merged))),
        }
    }


def load_lilac_ip(
    dataset_dir: Path,
    label_file: str = "labels_corrected.npy",
    channel: int = 0,
    merge_async_motor: bool = False,
    input_mode: str = "ip",
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, object]]:
    if input_mode not in {"i", "p", "ip", "iv", "ivp"}:
        raise ValueError("input_mode must be one of: i, p, ip, iv, ivp.")
    current_path = dataset_dir / "current.npy"
    voltage_path = dataset_dir / "voltage.npy"
    labels_path = _find_label_file(dataset_dir, label_file)
    if not current_path.exists():
        raise FileNotFoundError(f"Missing file: {current_path}")
    if not voltage_path.exists():
        raise FileNotFoundError(f"Missing file: {voltage_path}")

    current_raw = np.load(current_path, allow_pickle=False)
    voltage_raw = np.load(voltage_path, allow_pickle=False)
    labels_raw = np.load(labels_path, allow_pickle=True)
    current = extract_signal_channel(current_raw, channel=channel, name="current.npy")
    voltage = extract_signal_channel(voltage_raw, channel=channel, name="voltage.npy")
    if current.shape != voltage.shape:
        raise ValueError(f"Current and voltage shapes differ after channel extraction: {current.shape} vs {voltage.shape}")
    labels = [normalize_label(v) for v in labels_raw]
    if current.shape[0] != len(labels):
        raise ValueError(f"Sample count mismatch: current={current.shape[0]}, labels={len(labels)}")
    if merge_async_motor:
        labels, merge_meta = merge_async_motor_labels(labels)
    else:
        merge_meta = {"label_merge": {"enabled": False}}

    streams: Dict[str, np.ndarray] = {}
    if input_mode in {"i", "ip", "iv", "ivp"}:
        streams["i"] = current.astype(np.float32)
    if input_mode in {"iv", "ivp"}:
        streams["v"] = voltage.astype(np.float32)
    if input_mode in {"p", "ip", "ivp"}:
        streams["p"] = (current * voltage).astype(np.float32)
    meta: Dict[str, object] = {
        "dataset": "lilac",
        "dataset_dir": str(dataset_dir),
        "labels_path": str(labels_path),
        "input_mode": input_mode,
        "channel": int(channel),
        "num_samples": int(current.shape[0]),
        "series_length": int(current.shape[1]),
        "streams": list(streams.keys()),
        **merge_meta,
    }
    return streams, labels, meta


def load_npy_ip(dataset_dir: Path, dataset_name: str, input_mode: str = "ip") -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, object]]:
    if input_mode not in {"i", "p", "ip", "iv", "ivp"}:
        raise ValueError("input_mode must be one of: i, p, ip, iv, ivp.")
    current_path = dataset_dir / "current.npy"
    voltage_path = dataset_dir / "voltage.npy"
    labels_path = dataset_dir / "labels.npy"
    for path in [current_path, voltage_path, labels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
    current = np.load(current_path, allow_pickle=False).astype(np.float32)
    voltage = np.load(voltage_path, allow_pickle=False).astype(np.float32)
    labels_raw = np.load(labels_path, allow_pickle=True)
    if current.shape != voltage.shape:
        raise ValueError(f"Current and voltage shapes differ: {current.shape} vs {voltage.shape}")
    labels = [normalize_label(v) for v in labels_raw]
    if current.shape[0] != len(labels):
        raise ValueError(f"Sample count mismatch: current={current.shape[0]}, labels={len(labels)}")
    streams: Dict[str, np.ndarray] = {}
    if input_mode in {"i", "ip", "iv", "ivp"}:
        streams["i"] = current.astype(np.float32)
    if input_mode in {"iv", "ivp"}:
        streams["v"] = voltage.astype(np.float32)
    if input_mode in {"p", "ip", "ivp"}:
        streams["p"] = (current * voltage).astype(np.float32)
    meta: Dict[str, object] = {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "labels_path": str(labels_path),
        "input_mode": input_mode,
        "num_samples": int(current.shape[0]),
        "series_length": int(current.shape[1]),
        "streams": list(streams.keys()),
    }
    return streams, labels, meta


def _norm_path_key(path: object) -> str:
    return os.path.normcase(os.path.normpath(str(path).strip()))


def load_single_device_mat_ip(dataset_dir: Path, input_mode: str = "ip") -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, object]]:
    if input_mode not in {"i", "p", "ip", "iv", "ivp"}:
        raise ValueError("input_mode must be one of: i, p, ip, iv, ivp.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    index_path = dataset_dir / "classification_index.csv"
    index_by_destination: Dict[str, Dict[str, str]] = {}
    if index_path.exists():
        with index_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                destination = row.get("destination", "")
                if destination:
                    index_by_destination[_norm_path_key(destination)] = dict(row)

    voltages: List[np.ndarray] = []
    currents: List[np.ndarray] = []
    labels: List[str] = []
    sample_records: List[Dict[str, object]] = []
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: sort_label_key(p.name))
    for class_dir in class_dirs:
        label = normalize_label(class_dir.name)
        for mat_path in sorted(class_dir.glob("*.mat"), key=lambda p: p.name):
            mat = loadmat(mat_path)
            if "V" not in mat or "I" not in mat:
                raise KeyError(f"{mat_path} must contain fields 'V' and 'I'.")
            voltage = np.asarray(mat["V"]).ravel().astype(np.float32)
            current = np.asarray(mat["I"]).ravel().astype(np.float32)
            if voltage.shape != current.shape:
                raise ValueError(f"V and I lengths differ in {mat_path}: {voltage.shape} vs {current.shape}")
            if voltage.ndim != 1 or len(voltage) != 128:
                raise ValueError(f"Expected V/I length 128 in {mat_path}, got {len(voltage)}.")
            if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(current)):
                raise ValueError(f"Non-finite V/I values found in {mat_path}.")

            sample_index = len(labels)
            record = {
                "sample_index": int(sample_index),
                "label": label,
                "mat_path": str(mat_path),
            }
            record.update(index_by_destination.get(_norm_path_key(mat_path), {}))
            voltages.append(voltage)
            currents.append(current)
            labels.append(label)
            sample_records.append(record)

    if not labels:
        raise FileNotFoundError(f"No .mat samples found under class subdirectories of {dataset_dir}")

    voltage_arr = np.stack(voltages, axis=0).astype(np.float32)
    current_arr = np.stack(currents, axis=0).astype(np.float32)
    streams: Dict[str, np.ndarray] = {}
    if input_mode in {"i", "ip", "iv", "ivp"}:
        streams["i"] = current_arr
    if input_mode in {"iv", "ivp"}:
        streams["v"] = voltage_arr
    if input_mode in {"p", "ip", "ivp"}:
        streams["p"] = (current_arr * voltage_arr).astype(np.float32)
    meta: Dict[str, object] = {
        "dataset": "single_device_mat",
        "dataset_dir": str(dataset_dir),
        "classification_index_path": str(index_path) if index_path.exists() else None,
        "input_mode": input_mode,
        "num_samples": int(current_arr.shape[0]),
        "series_length": int(current_arr.shape[1]),
        "streams": list(streams.keys()),
        "classes": sorted(set(labels), key=sort_label_key),
        "sample_records": sample_records,
    }
    return streams, labels, meta


def load_dataset(
    dataset: str,
    dataset_dir: Path,
    label_file: str = "labels_corrected.npy",
    channel: int = 0,
    merge_async_motor: bool = False,
    input_mode: str = "ip",
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, object], Dict[str, List[int]]]:
    if dataset == "lilac":
        streams, labels, meta = load_lilac_ip(
            dataset_dir=dataset_dir,
            label_file=label_file,
            channel=channel,
            merge_async_motor=merge_async_motor,
            input_mode=input_mode,
        )
    elif dataset in {"plaid_other", "whited_other"}:
        streams, labels, meta = load_npy_ip(dataset_dir=dataset_dir, dataset_name=dataset, input_mode=input_mode)
    elif dataset == "single_device_mat":
        streams, labels, meta = load_single_device_mat_ip(dataset_dir=dataset_dir, input_mode=input_mode)
    else:
        raise ValueError("--dataset must be lilac, plaid_other, whited_other, or single_device_mat.")
    return streams, labels, meta, make_class_index(labels)
