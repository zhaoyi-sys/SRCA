#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from srca_open_set_engineering.pipeline import DEFAULT_ILSED_DIR, DEFAULT_LILAC_DIR, SRCAConfig, test_artifact, train_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test SRCA on LILAC and/or ILSED.")
    parser.add_argument("--datasets", default="lilac,ilsed", help="Comma-separated: lilac,ilsed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_srca_open_set_engineering"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-label", default=None, help="Optional single unknown label used for every selected dataset.")
    parser.add_argument("--lilac-dir", type=Path, default=DEFAULT_LILAC_DIR)
    parser.add_argument("--ilsed-dir", type=Path, default=DEFAULT_ILSED_DIR)
    parser.add_argument("--sphor-epochs", type=int, default=100)
    parser.add_argument("--reject-k", type=int, default=3)
    parser.add_argument("--threshold-percentile", type=float, default=93.0)
    parser.add_argument("--split-strategy", choices=["random", "index_block", "index_stride", "centroid_block", "nn_block"], default="random")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for dataset in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        dataset_dir = args.lilac_dir if dataset == "lilac" else args.ilsed_dir
        config = SRCAConfig(
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=args.output_dir,
            seed=int(args.seed),
            unknown_label=args.unknown_label,
            sphor_epochs=int(args.sphor_epochs),
            reject_k=int(args.reject_k),
            threshold_percentile=float(args.threshold_percentile),
            split_strategy=args.split_strategy,
            device=args.device,
        )
        for artifact_dir in train_dataset(config):
            result = test_artifact(artifact_dir, device_text=args.device)
            result["dataset"] = dataset
            result["artifact_dir"] = str(artifact_dir)
            rows.append(result)
            print(
                f"{dataset} unknown={result['unknown_label']} "
                f"F1-macro={result['open_macro_f1']:.4f} "
                f"unknown-F1={result['unknown_f1']:.4f} AUROC={result['auroc']:.4f}"
            )
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary_all_test_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
