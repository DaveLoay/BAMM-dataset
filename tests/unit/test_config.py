"""Tests for config loading, using the two paper experiment configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import SplitConfig, load_experiment, load_features_from_snapshot

REPO = Path(__file__).resolve().parents[2]
CNN_CLEAN = REPO / "configs" / "experiments" / "cnn_clean.yaml"
CNN_BROADCAST = REPO / "configs" / "experiments" / "cnn_broadcast.yaml"


@pytest.mark.parametrize("path", [CNN_CLEAN, CNN_BROADCAST], ids=["clean", "broadcast"])
def test_paper_configs_load_with_the_paper_features(path):
    cfg = load_experiment(path)
    f = cfg.features
    # Paper, Sec. 4.1: mono 8 kHz, STFT 2048 / hop 1024, 128 mels in 20 Hz - 4 kHz,
    # dB, fixed standardization (mean -4, std 3), 5 s windows.
    assert (f.audio.sample_rate, f.audio.mono) == (8000, True)
    assert (f.stft.n_fft, f.stft.win_length, f.stft.hop_length) == (2048, 2048, 1024)
    assert (f.mel.n_mels, f.mel.fmin, f.mel.fmax, f.mel.power) == (128, 20.0, 4000.0, 2.0)
    assert (f.normalization.strategy, f.normalization.mean, f.normalization.std) == (
        "fixed", -4.0, 3.0,
    )
    assert cfg.training.crop.window_sec == 5
    assert cfg.input_shape == (38, 128, 1)
    assert cfg.model.backbone.filters == [16, 32, 64, 128, 256, 512]


def test_the_two_models_differ_only_in_training_data():
    clean = load_experiment(CNN_CLEAN).to_flat_dict()
    broadcast = load_experiment(CNN_BROADCAST).to_flat_dict()
    differing = {k for k in clean if clean[k] != broadcast.get(k)}
    allowed = {"name", "description", "training.crop.samples_per_file"}
    assert {k for k in differing if not k.startswith("dataset.")} <= allowed
    assert clean["dataset.name"] == "fma_medium_sonics_suno_v35"
    assert broadcast["dataset.name"] == "synthetic_broadcast"


def test_snapshot_round_trip(tmp_path):
    cfg = load_experiment(CNN_CLEAN)
    snapshot = tmp_path / "config_snapshot.yaml"
    snapshot.write_text(yaml.dump(cfg.to_flat_dict(), sort_keys=False))
    features_cfg, window_sec = load_features_from_snapshot(snapshot)
    assert features_cfg == cfg.features
    assert window_sec == 5


def test_missing_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_features_from_snapshot(tmp_path / "nope.yaml")


def test_missing_referenced_config_raises(tmp_path):
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    raw = yaml.safe_load(CNN_CLEAN.read_text())
    raw["dataset"] = "configs/datasets/does_not_exist.yaml"
    path = tmp_path / "configs" / "experiments" / "exp.yaml"
    path.write_text(yaml.dump(raw))
    with pytest.raises(FileNotFoundError):
        load_experiment(path)


@pytest.mark.parametrize("ratios", [[0.7, 0.3], [0.5, 0.3, 0.3]])
def test_invalid_split_ratios_raise(ratios):
    with pytest.raises(ValueError):
        SplitConfig(strategy="stratified", ratios=ratios, seed=1)
