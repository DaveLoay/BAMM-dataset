# src/features/spectrogram.py
"""
Compute the log-mel spectrogram used by the CNN Clean / CNN Broadcast models.

Pipeline (paper, Sec. 4.1): power mel spectrogram (STFT 2048 / hop 1024,
128 mel bands, 20 Hz - 4 kHz) -> dB (ref = max) -> fixed standardization
((S - mean) / std, with mean = -4.0 and std = 3.0).
"""

from __future__ import annotations

import librosa
import numpy as np

from src.config import FeaturesConfig


def compute_spectrogram(
    y: np.ndarray,
    features_cfg: FeaturesConfig,
) -> np.ndarray:
    """Compute a full-length, time-major feature map from a waveform.

    Args:
        y: 1-D float32 waveform at features_cfg.audio.sample_rate.
        features_cfg: Features config (STFT, mel and normalization parameters).

    Returns:
        2-D float32 array of shape (n_frames, n_mels), time-major.
    """
    if features_cfg.type != "mel_spectrogram_db":
        raise ValueError(f"Unknown feature type: {features_cfg.type}")

    S_db = _compute_mel_spectrogram_db(y, features_cfg)
    S_db = _normalize(S_db, features_cfg)
    return np.ascontiguousarray(S_db.T.astype(np.float32))


def _compute_mel_spectrogram_db(y: np.ndarray, features_cfg: FeaturesConfig) -> np.ndarray:
    """Power mel spectrogram -> dB."""
    if features_cfg.mel is None:
        raise ValueError("Feature type 'mel_spectrogram_db' requires 'mel' config.")

    stft = features_cfg.stft
    mel = features_cfg.mel
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=features_cfg.audio.sample_rate,
        n_fft=stft.n_fft,
        hop_length=stft.hop_length,
        win_length=stft.win_length,
        n_mels=mel.n_mels,
        fmin=mel.fmin,
        fmax=mel.fmax,
        power=mel.power,
    )
    return librosa.power_to_db(S_mel, ref=np.max)


def _normalize(S: np.ndarray, features_cfg: FeaturesConfig) -> np.ndarray:
    """Apply the configured normalization strategy to a spectrogram."""
    norm = features_cfg.normalization

    if norm.strategy == "fixed":
        return (S - norm.mean) / norm.std

    if norm.strategy == "per_sample":
        mean = S.mean()
        std = S.std() + 1e-8
        return (S - mean) / std

    if norm.strategy == "none":
        return S

    raise ValueError(f"Unknown normalization strategy: {norm.strategy}")
