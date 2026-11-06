#!/usr/bin/env python3
"""Download the released CNN Clean and CNN Broadcast weights from the Hugging Face Hub.

Result:
    weights/cnn_clean/model.h5             + config_snapshot.yaml
    weights/cnn_broadcast/model.h5         + config_snapshot.yaml

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --output-dir weights --revision v1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "DaveLoay/bamm-cnn-detectors"
MODELS = ("cnn_clean", "cnn_broadcast")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help=f"default: {DEFAULT_REPO}")
    parser.add_argument("--revision", default=None, help="Branch, tag or commit (default: main)")
    parser.add_argument("--output-dir", default="weights", help="default: weights/")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.output_dir)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=out,
        allow_patterns=[f"{m}/*" for m in MODELS],
    )
    for m in MODELS:
        for name in ("model.h5", "config_snapshot.yaml"):
            path = out / m / name
            if not path.exists():
                raise SystemExit(f"Missing after download: {path}")
        print(f"{m}: {out / m / 'model.h5'}")


if __name__ == "__main__":
    main()
