"""Tests for log-mel spectrogram computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    AudioConfig,
    FeaturesConfig,
    MelConfig,
    NormalizationConfig,
    STFTConfig,
)
from src.features.spectrogram import compute_spectrogram


def _cfg(normalization: NormalizationConfig | None = None, **overrides) -> FeaturesConfig:
    """The paper's feature config: 8 kHz, STFT 2048/1024, 128 mels in 20-4000 Hz."""
    kwargs = dict(
        name="mel_8k_128",
        type="mel_spectrogram_db",
        audio=AudioConfig(sample_rate=8000, mono=True),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=1024),
        mel=MelConfig(n_mels=128, fmin=20.0, fmax=4000.0, power=2.0),
        normalization=normalization or NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
        storage="full",
        output_format="npy",
    )
    kwargs.update(overrides)
    return FeaturesConfig(**kwargs)


@pytest.fixture
def sine_wave() -> np.ndarray:
    """2 s of a 440 Hz sine at 8 kHz."""
    t = np.linspace(0, 2.0, 16000, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_output_shape_is_time_major(sine_wave):
    spec = compute_spectrogram(sine_wave, _cfg())
    # librosa centres frames: 1 + n_samples // hop
    assert spec.shape == (1 + 16000 // 1024, 128)
    assert spec.dtype == np.float32
    assert spec.flags["C_CONTIGUOUS"]


def test_frames_for_five_seconds_matches_model_input():
    # The released models take 38 x 128 inputs.
    assert _cfg().frames_for_duration(5.0) == 38
    assert _cfg().n_freq_bins == 128


def test_fixed_normalization(sine_wave):
    raw = compute_spectrogram(sine_wave, _cfg(NormalizationConfig(strategy="none")))
    fixed = compute_spectrogram(sine_wave, _cfg())
    np.testing.assert_allclose(fixed, (raw + 4.0) / 3.0, rtol=1e-5, atol=1e-5)


def test_db_reference_is_the_maximum(sine_wave):
    raw = compute_spectrogram(sine_wave, _cfg(NormalizationConfig(strategy="none")))
    assert raw.max() == pytest.approx(0.0, abs=1e-4)


def test_per_sample_normalization(sine_wave):
    spec = compute_spectrogram(sine_wave, _cfg(NormalizationConfig(strategy="per_sample")))
    assert abs(float(spec.mean())) < 1e-4
    assert float(spec.std()) == pytest.approx(1.0, abs=1e-3)


def test_unknown_normalization_raises(sine_wave):
    with pytest.raises(ValueError, match="Unknown normalization"):
        compute_spectrogram(sine_wave, _cfg(NormalizationConfig(strategy="bogus")))


def test_unknown_feature_type_raises(sine_wave):
    with pytest.raises(ValueError, match="Unknown feature type"):
        compute_spectrogram(sine_wave, _cfg(type="stft_magnitude_db"))


def test_mel_config_is_required(sine_wave):
    with pytest.raises(ValueError, match="requires 'mel' config"):
        compute_spectrogram(sine_wave, _cfg(mel=None))
