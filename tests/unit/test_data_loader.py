# tests/unit/test_data_loader.py
"""Tests for tf.data pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.training.data_loader import (
    build_dataset,
    build_train_val_datasets,
    load_feature_manifest,
)


def _constant_entries(tmp_path, *, n_train: int = 2, n_val: int = 2):
    n_frames = 16
    n_freq = 8
    entries = []
    for i in range(n_train + n_val):
        npy_path = tmp_path / f"constant_{i}.npy"
        np.save(npy_path, np.ones((n_frames, n_freq), dtype=np.float32))
        split = "train" if i < n_train else "val"
        entries.append({
            "feature_path": str(npy_path),
            "source_audio": f"/datasets/constant_{i}.mp3",
            "label": i % 2,
            "label_name": "real" if i % 2 == 0 else "fake",
            "split": split,
            "n_frames": n_frames,
            "n_freq_bins": n_freq,
            "duration_sec": 1.0,
        })
    return entries


def _write_manifest(path, entries):
    manifest = {
        "feature_config": "test_features",
        "dataset": "test_dataset",
        "entries": entries,
        "summary": {"total": len(entries), "errors": 0},
    }
    with open(path, "w") as f:
        json.dump(manifest, f)


@pytest.fixture
def feature_dir(tmp_path):
    """Create synthetic .npy spectrograms and a feature manifest."""
    n_freq = 743
    n_files = 6
    entries = []

    for i in range(n_files):
        n_frames = np.random.randint(200, 500)
        spec = np.random.randn(n_frames, n_freq).astype(np.float32)
        npy_path = tmp_path / f"spec_{i}.npy"
        np.save(npy_path, spec)

        entries.append({
            "feature_path": str(npy_path),
            "source_audio": f"/datasets/fake_{i}.mp3",
            "label": i % 2,
            "label_name": "real" if i % 2 == 0 else "fake",
            "split": "train" if i < 4 else "val",
            "n_frames": n_frames,
            "n_freq_bins": n_freq,
            "duration_sec": round(n_frames * 512 / 44100, 2),
        })

    manifest = {
        "feature_config": "test_features",
        "dataset": "test_dataset",
        "entries": entries,
        "summary": {"total": n_files, "errors": 0},
    }

    manifest_path = tmp_path / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path, entries


def test_load_feature_manifest(feature_dir):
    manifest_path, expected_entries = feature_dir
    manifest, entries = load_feature_manifest(manifest_path)

    assert manifest["feature_config"] == "test_features"
    assert len(entries) == len(expected_entries)


def test_load_feature_manifest_not_found():
    with pytest.raises(FileNotFoundError):
        load_feature_manifest("/nonexistent/manifest.json")


def test_build_dataset_shapes(feature_dir):
    manifest_path, entries = feature_dir
    train_entries = [e for e in entries if e["split"] == "train"]

    crop_frames = 67
    samples_per_file = 3
    batch_size = 2

    ds = build_dataset(
        train_entries,
        crop_frames=crop_frames,
        samples_per_file=samples_per_file,
        batch_size=batch_size,
        shuffle_buffer=10,
        seed=42,
        is_training=False,
    )

    batch = next(iter(ds))
    specs, labels = batch

    assert specs.shape[1] == crop_frames  # time (crop width)
    assert specs.shape[2] == 743          # n_freq
    assert specs.shape[3] == 1          # channel
    assert labels.shape[0] == specs.shape[0]


def test_build_dataset_labels(feature_dir):
    manifest_path, entries = feature_dir
    train_entries = [e for e in entries if e["split"] == "train"]

    ds = build_dataset(
        train_entries,
        crop_frames=67,
        samples_per_file=2,
        batch_size=100,
        seed=42,
        is_training=False,
    )

    all_labels = []
    for _, labels in ds:
        all_labels.extend(labels.numpy().tolist())

    assert set(all_labels).issubset({0, 1})


def test_build_dataset_pads_short_spectrogram(tmp_path):
    n_frames = 30
    n_freq = 16
    crop_frames = 67
    spec = np.ones((n_frames, n_freq), dtype=np.float32)
    npy_path = tmp_path / "short.npy"
    np.save(npy_path, spec)

    entries = [{
        "feature_path": str(npy_path),
        "source_audio": "/datasets/short.mp4",
        "label": 0,
        "label_name": "real",
        "split": "train",
        "n_frames": n_frames,
        "n_freq_bins": n_freq,
        "duration_sec": 4.0,
    }]

    ds = build_dataset(
        entries,
        crop_frames=crop_frames,
        samples_per_file=1,
        batch_size=1,
        seed=42,
        is_training=False,
    )

    specs, labels = next(iter(ds))

    assert specs.shape == (1, crop_frames, n_freq, 1)
    np.testing.assert_array_equal(specs.numpy()[0, :n_frames, :, 0], 1.0)
    np.testing.assert_array_equal(specs.numpy()[0, n_frames:, :, 0], 0.0)
    assert labels.numpy().tolist() == [0]


def test_build_train_val_rejects_non_binary_labels(feature_dir):
    manifest_path, entries = feature_dir

    entries[0]["label"] = 2
    entries[0]["label_name"] = "speech"

    manifest = {
        "feature_config": "test_features",
        "dataset": "test_dataset",
        "entries": entries,
        "summary": {"total": len(entries), "errors": 0},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="non-binary labels"):
        build_train_val_datasets(
            feature_manifest_path=manifest_path,
            crop_frames=67,
            samples_per_file=2,
            batch_size=2,
        )

