#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from srca_open_set_engineering.pipeline import DEFAULT_ILSED_DIR, DEFAULT_LILAC_DIR, SRCAConfig, train_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRCA open-set NILM artifacts.")
    parser.add_argument("--dataset", choices=["lilac", "ilsed", "mine"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-file", default="labels_corrected.npy")
    parser.add_argument("--input-mode", choices=["i", "p", "ip", "iv", "ivp"], default="ip")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--merge-async-motor", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-label", default=None, help="If omitted, train one artifact for each leave-one-unknown class.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-strategy", choices=["random", "index_block", "index_stride", "centroid_block", "nn_block"], default="random")
    parser.add_argument("--split-offset", type=int, default=1)
    parser.add_argument("--num-features-per-stream", type=int, default=672)
    parser.add_argument("--max-dilations-per-kernel", type=int, default=32)
    parser.add_argument("--closed-feature-scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--reject-feature-scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--sphor-epochs", type=int, default=100)
    parser.add_argument("--sphor-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ortho-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--reject-k", type=int, default=3)
    parser.add_argument("--threshold-percentile", type=float, default=93.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minirocket-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = DEFAULT_LILAC_DIR if args.dataset == "lilac" else DEFAULT_ILSED_DIR
    config = SRCAConfig(
        dataset=args.dataset,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        label_file=args.label_file,
        input_mode=args.input_mode,
        channel=args.channel,
        merge_async_motor=args.merge_async_motor,
        seed=args.seed,
        train_fraction=args.train_fraction,
        split_strategy=args.split_strategy,
        split_offset=args.split_offset,
        unknown_label=args.unknown_label,
        num_features_per_stream=args.num_features_per_stream,
        max_dilations_per_kernel=args.max_dilations_per_kernel,
        closed_feature_scaler=args.closed_feature_scaler,
        reject_feature_scaler=args.reject_feature_scaler,
        embedding_dim=args.embedding_dim,
        sphor_epochs=args.sphor_epochs,
        sphor_lr=args.sphor_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        temperature=args.temperature,
        ortho_weight=args.ortho_weight,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        grad_clip=args.grad_clip,
        reject_k=args.reject_k,
        threshold_percentile=args.threshold_percentile,
        device=args.device,
        minirocket_dir=args.minirocket_dir,
    )
    artifact_dirs = train_dataset(config)
    for path in artifact_dirs:
        print(path)


if __name__ == "__main__":
    main()
