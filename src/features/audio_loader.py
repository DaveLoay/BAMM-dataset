# src/features/audio_loader.py
"""
Load and preprocess audio files.

Every input (mp3, wav, mp4, ...) is first transcoded with FFmpeg to a mono MP3
at the target sample rate (8 kHz) and 5 kbps per kHz (40 kbps at 8 kHz), then
decoded with librosa. This puts all sources through the same low-bitrate codec
the models were trained with.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from src.config import AudioConfig

logger = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    audio_cfg: AudioConfig,
    bitrate_kbps: int | None = None,
) -> tuple[np.ndarray, float]:
    """Load an audio file, enforce mono, and resample to target SR.

    A pre-conversion step using FFmpeg is applied to standardize codification
    and normalize inputs into a mono MP3 file at target sample rate and bitrate.

    Args:
        path: Path to the audio file.
        audio_cfg: Audio preprocessing config (sample_rate, mono).
        bitrate_kbps: Target MP3 bitrate in kbps. Defaults to 5x the sample
            rate in kHz (e.g. 40 kbps at 8000 Hz) if not given.

    Returns:
        Tuple of (waveform as float32 1-D array, duration in seconds).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If librosa or FFmpeg fails to process the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if bitrate_kbps is None:
        bitrate_kbps = int((audio_cfg.sample_rate / 1000.0) * 5)

    # Create a temporary mp3 file
    fd, temp_mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        # Run FFmpeg to normalize the file: mono, target SR, target bitrate.
        # -ac 1 mixes down to mono before encoding, matching how the synthetic
        # broadcast training set was built (scripts/build_synthetic_broadcast_dataset.py).
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                   # Overwrite output
            "-i", str(path),        # Input file
            "-vn",                  # No video
            "-ac", "1",             # Mono
            "-ar", str(audio_cfg.sample_rate),  # Target sample rate
            "-b:a", f"{bitrate_kbps}k",  # Target bitrate
        ]
        ffmpeg_cmd += [
            "-loglevel", "error",   # Silence most output
            temp_mp3_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed for {path}: {result.stderr}")

        try:
            y, sr = librosa.load(
                temp_mp3_path,
                sr=audio_cfg.sample_rate,
                mono=audio_cfg.mono,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio {path} via temporary MP3: {exc}") from exc

        y = y.astype(np.float32)
        duration_sec = len(y) / sr

        return y, duration_sec
    finally:
        # Clean up the temporary MP3 file securely
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
