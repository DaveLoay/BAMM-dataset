"""End-to-end test of the trainer on a tiny synthetic feature cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from src.config import (
    AudioConfig,
    BackboneConfig,
    CropConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeaturesConfig,
    HeadConfig,
    MelConfig,
    ModelConfig,
    NormalizationConfig,
    OutputConfig,
    SplitConfig,
    STFTConfig,
    TrainingConfig,
    load_features_from_snapshot,
)
from src.training.trainer import BEST_MODEL_FILENAME, LAST_MODEL_FILENAME, train


def _make_cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny_exp",
        description="Test experiment",
        tags={"task": "test"},
        dataset=DatasetConfig(
            name="tiny_ds",
            description="Test dataset",
            paths={"real": "/data/real", "fake": "/data/fake"},
            format="mp3",
            split=SplitConfig(strategy="stratified", ratios=[0.7, 0.2, 0.1], seed=42),
        ),
        features=FeaturesConfig(
            name="tiny_mel",
            type="mel_spectrogram_db",
            audio=AudioConfig(sample_rate=8000, mono=True),
            stft=STFTConfig(n_fft=256, win_length=256, hop_length=128),
            mel=MelConfig(n_mels=16, fmin=20.0, fmax=4000.0, power=2.0),
            normalization=NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
            storage="full",
            output_format="npy",
        ),
        model=ModelConfig(
            name="tiny_cnn",
            type="spectrogram_cnn",
            backbone=BackboneConfig(
                filters=[4],
                kernels=[3],
                strides=[2],
                use_maxpool=True,
                activation="relu",
                batch_norm=False,
                pooling="average",
                padding="same",
            ),
            head=HeadConfig(dense_units=8, activation="relu", dropout=0.0),
            output=OutputConfig(
                type="binary", activation="sigmoid", loss="binary_crossentropy", threshold=0.5
            ),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            optimizer="adam",
            seed=42,
            crop=CropConfig(window_sec=0.5, strategy="random", samples_per_file=2),
            shuffle_buffer_multiplier=2,
            callbacks={},
        ),
        evaluation=EvaluationConfig(metrics=["accuracy"], plots=[]),
        config_path=tmp_path / "configs" / "experiments" / "tiny.yaml",
    )


def _write_feature_cache(tmp_path: Path, n_freq: int) -> Path:
    rng = np.random.default_rng(0)
    entries = []
    for split, n in (("train", 6), ("val", 4)):
        for i in range(n):
            label = i % 2
            path = tmp_path / "features" / split / f"{i}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            n_frames = 40 + i
            np.save(path, rng.normal(label, 1.0, size=(n_frames, n_freq)).astype(np.float32))
            entries.append({
                "feature_path": str(path),
                "label": label,
                "label_name": "fake" if label else "real",
                "split": split,
                "n_frames": n_frames,
            })
    manifest = tmp_path / "features" / "feature_manifest.json"
    manifest.write_text(json.dumps({"entries": entries}))
    return manifest


def test_train_writes_loadable_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = _make_cfg(tmp_path)
    manifest = _write_feature_cache(tmp_path, cfg.features.n_freq_bins)

    result = train(cfg, feature_manifest_path=manifest, output_dir=tmp_path / "artifacts")

    out = tmp_path / "artifacts" / cfg.name
    assert result["epochs_run"] == 1
    for name in (BEST_MODEL_FILENAME, LAST_MODEL_FILENAME, "config_snapshot.yaml",
                 "training_history.json", "model_summary.txt"):
        assert (out / name).exists(), name

    model = tf.keras.models.load_model(out / BEST_MODEL_FILENAME, compile=False)
    assert tuple(model.input_shape[1:]) == cfg.input_shape


def test_snapshot_round_trips_feature_config(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = _make_cfg(tmp_path)
    manifest = _write_feature_cache(tmp_path, cfg.features.n_freq_bins)
    train(cfg, feature_manifest_path=manifest, output_dir=tmp_path / "artifacts")

    snapshot = tmp_path / "artifacts" / cfg.name / "config_snapshot.yaml"
    features_cfg, window_sec = load_features_from_snapshot(snapshot)
    assert features_cfg == cfg.features
    assert window_sec == cfg.training.crop.window_sec
    assert yaml.safe_load(snapshot.read_text())["name"] == cfg.name
