#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from srca_open_set_engineering.pipeline import test_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained SRCA open-set NILM artifact.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minirocket-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = test_artifact(args.artifact_dir, device_text=args.device, minirocket_dir=args.minirocket_dir)
    print(
        "unknown={unknown_label} F1^(K+1)={unknown_f1:.4f} F1-macro={open_macro_f1:.4f} AUROC={auroc:.4f}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
