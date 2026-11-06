"""Tests for Stage 2 feature extraction, with audio decoding faked out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pipelines.extract_features as ef
from src.data.manifest import write_manifest
from src.data.scanner import AudioFile
from src.data.splitter import DataSplit

REPO = Path(__file__).resolve().parents[2]
CONFIG = str(REPO / "configs" / "experiments" / "cnn_clean.yaml")


def _fake_audio(monkeypatch):
    def fake_load_audio(path, audio_cfg):
        seed = sum(map(ord, str(path)))
        return np.random.default_rng(seed).normal(size=8000).astype(np.float32), 1.0

    monkeypatch.setattr(ef, "load_audio", fake_load_audio)


def _manifest(tmp_path: Path) -> Path:
    def af(name, label):
        return AudioFile(Path(f"/audio/{'fake' if label else 'real'}/{name}.mp3"), label,
                         "fake" if label else "real")

    split = DataSplit(
        train=[af("a", 0), af("b", 1), af("c", 0)],
        val=[af("d", 1)],
        test=[af("e", 0), af("f", 1)],
    )
    return write_manifest(split, tmp_path / "manifest.json", dataset_name="tiny")


def _run(monkeypatch, tmp_path, workers=1) -> dict:
    _fake_audio(monkeypatch)
    out = tmp_path / f"features_{workers}"
    ef.main(["--config", CONFIG, "--manifest", str(_manifest(tmp_path)),
             "--output-dir", str(out), "--workers", str(workers)])
    path = out / "fma_medium_sonics_suno_v35__mel_8k_128" / "feature_manifest.json"
    return json.loads(path.read_text())


def test_one_entry_per_file_in_manifest_order(monkeypatch, tmp_path):
    manifest = _run(monkeypatch, tmp_path)
    names = [Path(e["source_audio"]).stem for e in manifest["entries"]]
    assert names == ["a", "b", "c", "d", "e", "f"]
    assert manifest["summary"]["by_split"] == {"train": 3, "val": 1, "test": 2}
    assert manifest["summary"]["errors"] == 0


def test_saved_spectrograms_have_the_model_frequency_axis(monkeypatch, tmp_path):
    manifest = _run(monkeypatch, tmp_path)
    for e in manifest["entries"]:
        spec = np.load(e["feature_path"])
        assert spec.shape == (e["n_frames"], 128)
        assert e["label"] == (1 if e["label_name"] == "fake" else 0)


def test_failed_files_are_reported_not_fatal(monkeypatch, tmp_path):
    _fake_audio(monkeypatch)
    real_loader = ef.load_audio

    def flaky(path, audio_cfg):
        if Path(path).stem == "b":
            raise RuntimeError("cannot decode")
        return real_loader(path, audio_cfg)

    monkeypatch.setattr(ef, "load_audio", flaky)
    out = tmp_path / "features"
    ef.main(["--config", CONFIG, "--manifest", str(_manifest(tmp_path)),
             "--output-dir", str(out)])
    manifest = json.loads(
        (out / "fma_medium_sonics_suno_v35__mel_8k_128" / "feature_manifest.json").read_text()
    )
    assert manifest["summary"]["errors"] == 1
    assert [e["path"] for e in manifest["errors"]] == ["/audio/fake/b.mp3"]
