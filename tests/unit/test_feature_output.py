"""Tests for the feature cache path helpers."""

from pathlib import Path

import pytest

from src.features.feature_output import (
    feature_manifest_path,
    feature_output_name,
    resolve_feature_manifest,
)


def test_output_name_is_keyed_on_dataset_and_features():
    assert feature_output_name("mel_8k_128", "synthetic_broadcast") == (
        "synthetic_broadcast__mel_8k_128"
    )


def test_same_features_on_two_datasets_do_not_collide():
    a = feature_manifest_path("mel_8k_128", "fma_medium_sonics_suno_v35")
    b = feature_manifest_path("mel_8k_128", "synthetic_broadcast")
    assert a != b


def test_manifest_path_layout(tmp_path):
    path = feature_manifest_path("mel_8k_128", "ds", features_dir=tmp_path)
    assert path == Path(tmp_path) / "ds__mel_8k_128" / "feature_manifest.json"


def test_resolve_returns_existing_manifest(tmp_path):
    path = feature_manifest_path("mel_8k_128", "ds", features_dir=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert resolve_feature_manifest("mel_8k_128", "ds", features_dir=tmp_path) == path


def test_resolve_raises_when_not_extracted(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run Stage 2"):
        resolve_feature_manifest("mel_8k_128", "ds", features_dir=tmp_path)
