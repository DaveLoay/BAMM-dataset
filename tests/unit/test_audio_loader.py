# tests/unit/test_audio_loader.py
"""Tests for audio loader."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.config import AudioConfig
from src.features import audio_loader
from src.features.audio_loader import load_audio


@pytest.fixture
def mono_wav(tmp_path):
    """Create a short mono WAV file (1 second at 44100 Hz)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test_mono.wav"
    sf.write(path, y, sr)
    return path, sr, duration


@pytest.fixture
def stereo_wav(tmp_path):
    """Create a short stereo WAV file (1 second at 44100 Hz)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    stereo = np.stack([left, right], axis=-1)
    path = tmp_path / "test_stereo.wav"
    sf.write(path, stereo, sr)
    return path, sr, duration


def test_load_mono_audio(mono_wav):
    path, sr, expected_dur = mono_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, duration = load_audio(path, cfg)

    assert y.ndim == 1
    assert y.dtype == np.float32
    assert abs(duration - expected_dur) < 0.05
    assert len(y) == sr  # 1 second


def test_load_stereo_as_mono(stereo_wav):
    path, sr, _ = stereo_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, _ = load_audio(path, cfg)

    assert y.ndim == 1


def test_resample(mono_wav):
    path, original_sr, _ = mono_wav
    target_sr = 22050
    cfg = AudioConfig(sample_rate=target_sr, mono=True)

    y, duration = load_audio(path, cfg)

    assert len(y) == target_sr  # 1 second at new SR
    assert abs(duration - 1.0) < 0.05


def test_file_not_found():
    cfg = AudioConfig(sample_rate=44100, mono=True)
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/audio.wav", cfg)


# ── per-track duration cap (Step 1.2 / C2 / D2) ──────────────────────────


def _ffmpeg_cmd_for(monkeypatch, **kwargs):
    """Run load_audio with ffmpeg/librosa stubbed; return the ffmpeg argv."""
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **_):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(audio_loader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        audio_loader.librosa, "load", lambda *a, **k: (np.zeros(80, dtype=np.float32), 8000)
    )
    monkeypatch.setattr(audio_loader.Path, "exists", lambda self: True)

    audio_loader.load_audio(
        "/audio/track.mp3", AudioConfig(sample_rate=8000, mono=True), **kwargs
    )
    return captured["cmd"]


def test_no_cap_by_default_leaves_ffmpeg_unchanged(monkeypatch):
    cmd = _ffmpeg_cmd_for(monkeypatch)
    assert "-t" not in cmd

