"""Feature cache path helpers shared by Stages 2-4."""

from __future__ import annotations

from pathlib import Path


def feature_output_name(features_name: str, dataset_name: str) -> str:
    """Name of the feature cache directory for one dataset + feature config.

    The cache is keyed on both, because the same feature config extracted from
    two different datasets produces two different caches.
    """
    return f"{dataset_name}__{features_name}"


def feature_manifest_path(
    features_name: str,
    dataset_name: str,
    features_dir: str | Path = "data/features",
) -> Path:
    """Return the feature manifest path written by Stage 2."""
    name = feature_output_name(features_name, dataset_name)
    return Path(features_dir) / name / "feature_manifest.json"


def resolve_feature_manifest(
    features_name: str,
    dataset_name: str,
    features_dir: str | Path = "data/features",
) -> Path:
    """Return the Stage 2 feature manifest, or raise if it has not been extracted yet."""
    path = feature_manifest_path(features_name, dataset_name, features_dir=features_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {path}. Run Stage 2 (extract_features) first."
        )
    return path
