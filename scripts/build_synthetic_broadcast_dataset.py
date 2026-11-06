#!/usr/bin/env python3
"""Build the synthetic broadcast training set used for CNN Broadcast (paper, Sec. 4.1.2).

Example usage (the settings used for the paper):
    python scripts/build_synthetic_broadcast_dataset.py \
        --human-music-dir /path/to/datasets/fma_medium \
        --ai-music-dir /path/to/datasets/sonics_suno_v35 \
        --speech-dir /path/to/datasets/LibriSpeech/train-clean-360 \
        --output-dir /path/to/datasets/synthetic_broadcast \
        --mix-ratio 0.7 --snr-min-db -30 --snr-max-db 30 \
        --output-sample-rate 8000 --output-bitrate 40k \
        --workers 10

Given three input folders (Human music, AI music, Speech), this script creates
an output dataset with two class folders:
- Human/
- AI/

For each class, it enforces a mixed/clean ratio (default 70/30):
- Mixed sample: music + speech at random SNR
- Clean sample: original music unchanged

For mixed samples, speech is always present across the full music duration.
If one speech file is too short, more random speech files are concatenated
until the full music length is covered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AudioConfig  # noqa: E402
from src.data.scanner import scan_audio_dir  # noqa: E402
from src.features.audio_loader import load_audio  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _stable_seed_for_path(base_seed: int, path: Path) -> int:
    """Derive a deterministic per-file seed independent of worker scheduling."""
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    path_part = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((base_seed + path_part) % (2**32 - 1))


def _mix_waveforms_at_snr(signal: np.ndarray, speech: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal and speech in time domain with RMS-based SNR control."""
    eps = 1e-8

    rms_signal = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + eps))
    rms_speech = float(np.sqrt(np.mean(np.square(speech), dtype=np.float64) + eps))

    if rms_speech <= eps:
        mixed = signal.copy()
    else:
        snr_linear = float(10.0 ** (snr_db / 20.0))
        target_signal_rms = rms_speech * snr_linear
        gain_signal = target_signal_rms / max(rms_signal, eps)
        mixed = speech + (gain_signal * signal)

    peak = float(np.max(np.abs(mixed))) if mixed.size > 0 else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    return mixed.astype(np.float32)


@dataclass(frozen=True)
class Task:
    class_name: str  # Human or AI
    music_root: Path
    music_path: Path
    is_mixed: bool


@dataclass(frozen=True)
class Options:
    output_dir: Path
    speech_files: list[Path]
    audio_cfg: AudioConfig
    snr_min_db: float
    snr_max_db: float
    bitrate: str
    ffmpeg_bin: str
    dry_run: bool
    run_seed: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic broadcast dataset by mixing Human/AI music with "
            "speech (and keeping a clean share) from three input folders."
        )
    )
    parser.add_argument(
        "--human-music-dir",
        type=str,
        required=True,
        help="Input folder containing human music files",
    )
    parser.add_argument(
        "--ai-music-dir",
        type=str,
        required=True,
        help="Input folder containing AI music files",
    )
    parser.add_argument(
        "--speech-dir",
        type=str,
        required=True,
        help="Input folder containing speech files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output folder where Human/ and AI/ class folders will be created",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.7,
        help="Per-class fraction of samples to mix with speech (default: 0.7)",
    )
    parser.add_argument(
        "--snr-min-db",
        type=float,
        default=-30.0,
        help="Minimum SNR for mixed samples in dB (default: -30)",
    )
    parser.add_argument(
        "--snr-max-db",
        type=float,
        default=30.0,
        help="Maximum SNR for mixed samples in dB (default: 30)",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=8000,
        help="Output sample rate in Hz (default: 8000)",
    )
    parser.add_argument(
        "--output-bitrate",
        type=str,
        default="40k",
        help="Output MP3 bitrate (default: 40k)",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="all",
        help=(
            "Input extension filter for all folders (e.g. mp3, flac, wav). "
            "Use 'all' for all supported formats."
        ),
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional cap per class for quick experiments",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for synthesis/export (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic assignment and synthesis",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output-dir if it exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Plan and report only; do not write audio files",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="synthetic_broadcast_report",
        help="Prefix used for CSV/JSON report files",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.mix_ratio <= 1.0):
        raise ValueError("mix-ratio must be in [0.0, 1.0]")
    if args.snr_min_db > args.snr_max_db:
        raise ValueError("snr-min-db must be <= snr-max-db")
    if args.output_sample_rate <= 0:
        raise ValueError("output-sample-rate must be > 0")
    if args.workers <= 0:
        raise ValueError("workers must be >= 1")
    if args.max_files_per_class is not None and args.max_files_per_class <= 0:
        raise ValueError("max-files-per-class must be > 0 when provided")


def _resolve_audio_paths(
    human_root: Path,
    ai_root: Path,
    speech_root: Path,
    format_filter: str | None,
    max_files_per_class: int | None,
) -> tuple[list[Path], list[Path], list[Path]]:
    human_files = scan_audio_dir(human_root, format=format_filter)
    ai_files = scan_audio_dir(ai_root, format=format_filter)
    speech_files = scan_audio_dir(speech_root, format=format_filter)

    if max_files_per_class is not None:
        human_files = human_files[:max_files_per_class]
        ai_files = ai_files[:max_files_per_class]

    if not human_files:
        raise RuntimeError("No human music files found")
    if not ai_files:
        raise RuntimeError("No AI music files found")
    if not speech_files:
        raise RuntimeError("No speech files found")

    return human_files, ai_files, speech_files


def _assign_mixed_flags(n_files: int, mix_ratio: float, rng: np.random.Generator) -> list[bool]:
    n_mixed = int(round(n_files * mix_ratio))
    n_mixed = min(max(n_mixed, 0), n_files)
    flags = np.array([True] * n_mixed + [False] * (n_files - n_mixed), dtype=bool)
    rng.shuffle(flags)
    return flags.tolist()


def _compose_full_speech_cover(
    target_len: int,
    speech_files: list[Path],
    audio_cfg: AudioConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32), 0

    chunks: list[np.ndarray] = []
    used = 0
    remaining = target_len
    empty_guard = 0

    while remaining > 0:
        speech_path = speech_files[int(rng.integers(0, len(speech_files)))]
        speech_wave, _ = load_audio(speech_path, audio_cfg)

        if speech_wave.size == 0:
            empty_guard += 1
            if empty_guard > 1000:
                raise RuntimeError("Speech pool appears to contain only empty audio")
            continue

        if speech_wave.shape[0] >= remaining:
            chunks.append(speech_wave[:remaining])
            used += 1
            remaining = 0
            break

        chunks.append(speech_wave)
        used += 1
        remaining -= speech_wave.shape[0]

    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False), used


def _output_path_for_task(task: Task, output_dir: Path) -> Path:
    class_dir = output_dir / task.class_name
    relative = task.music_path.resolve().relative_to(task.music_root.resolve())
    return (class_dir / relative).with_suffix(".mp3")


def _encode_mp3(
    waveform: np.ndarray,
    sample_rate: int,
    output_path: Path,
    bitrate: str,
    ffmpeg_bin: str,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        sf.write(tmp_wav, waveform, sample_rate, subtype="PCM_16")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(tmp_wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_text = (result.stderr or result.stdout).strip()
            raise RuntimeError(error_text or "ffmpeg encoding failed")
    finally:
        tmp_wav.unlink(missing_ok=True)


def _process_task(task: Task, opts: Options) -> dict[str, object]:
    output_path = _output_path_for_task(task, opts.output_dir)
    result: dict[str, object] = {
        "class_name": task.class_name,
        "music_path": str(task.music_path),
        "output_path": str(output_path),
        "mode": "mixed" if task.is_mixed else "clean",
        "snr_db": None,
        "speech_segments_used": 0,
        "duration_sec": 0.0,
        "status": "unknown",
        "error": "",
    }

    try:
        local_seed = _stable_seed_for_path(opts.run_seed, task.music_path)
        rng = np.random.default_rng(local_seed)

        y_music, duration_sec = load_audio(task.music_path, opts.audio_cfg)
        result["duration_sec"] = round(float(duration_sec), 6)

        if task.is_mixed:
            y_speech_cover, n_segments = _compose_full_speech_cover(
                target_len=y_music.shape[0],
                speech_files=opts.speech_files,
                audio_cfg=opts.audio_cfg,
                rng=rng,
            )
            snr_db = float(rng.uniform(opts.snr_min_db, opts.snr_max_db))
            y_out = _mix_waveforms_at_snr(signal=y_music, speech=y_speech_cover, snr_db=snr_db)
            result["snr_db"] = round(snr_db, 6)
            result["speech_segments_used"] = int(n_segments)
        else:
            y_out = y_music.astype(np.float32, copy=False)

        if not opts.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _encode_mp3(
                waveform=y_out,
                sample_rate=opts.audio_cfg.sample_rate,
                output_path=output_path,
                bitrate=opts.bitrate,
                ffmpeg_bin=opts.ffmpeg_bin,
            )
            result["status"] = "written"
        else:
            result["status"] = "dry_run"

        return result

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return result


def _write_reports(
    results: list[dict[str, object]],
    output_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"

    rows = sorted(results, key=lambda r: (str(r["class_name"]), str(r["music_path"])))
    fieldnames = [
        "class_name",
        "music_path",
        "output_path",
        "mode",
        "snr_db",
        "speech_segments_used",
        "duration_sec",
        "status",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "total": len(rows),
        "status_counts": {},
        "by_class": {},
    }

    status_counts: dict[str, int] = {}
    class_stats: dict[str, dict[str, int]] = {}

    for row in rows:
        status = str(row["status"])
        class_name = str(row["class_name"])
        mode = str(row["mode"])

        status_counts[status] = status_counts.get(status, 0) + 1

        if class_name not in class_stats:
            class_stats[class_name] = {
                "total": 0,
                "mixed": 0,
                "clean": 0,
                "written": 0,
                "failed": 0,
                "dry_run": 0,
            }

        class_stats[class_name]["total"] += 1
        if mode == "mixed":
            class_stats[class_name]["mixed"] += 1
        else:
            class_stats[class_name]["clean"] += 1

        if status == "written":
            class_stats[class_name]["written"] += 1
        elif status == "failed":
            class_stats[class_name]["failed"] += 1
        elif status == "dry_run":
            class_stats[class_name]["dry_run"] += 1

    for class_name, stats in class_stats.items():
        total = stats["total"]
        stats["mixed_ratio"] = round(stats["mixed"] / total, 6) if total > 0 else 0.0
        stats["clean_ratio"] = round(stats["clean"] / total, 6) if total > 0 else 0.0

    summary["status_counts"] = status_counts
    summary["by_class"] = class_stats

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": rows}, f, indent=2)

    return csv_path, json_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

    human_root = Path(args.human_music_dir).expanduser().resolve()
    ai_root = Path(args.ai_music_dir).expanduser().resolve()
    speech_root = Path(args.speech_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for p, name in [
        (human_root, "human-music-dir"),
        (ai_root, "ai-music-dir"),
        (speech_root, "speech-dir"),
    ]:
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"{name} does not exist or is not a directory: {p}")

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"output-dir already exists: {output_dir}. Use --overwrite to replace it."
            )

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None and not args.dry_run:
        raise RuntimeError("ffmpeg was not found in PATH. Install ffmpeg or use --dry-run.")

    format_filter = None if args.input_format.strip().lower() == "all" else args.input_format
    human_files, ai_files, speech_files = _resolve_audio_paths(
        human_root=human_root,
        ai_root=ai_root,
        speech_root=speech_root,
        format_filter=format_filter,
        max_files_per_class=args.max_files_per_class,
    )

    if args.seed is None:
        run_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    else:
        run_seed = int(args.seed)
    planning_rng = np.random.default_rng(run_seed)

    human_flags = _assign_mixed_flags(len(human_files), args.mix_ratio, planning_rng)
    ai_flags = _assign_mixed_flags(len(ai_files), args.mix_ratio, planning_rng)

    tasks: list[Task] = []
    tasks.extend(
        Task(class_name="Human", music_root=human_root, music_path=path, is_mixed=flag)
        for path, flag in zip(human_files, human_flags, strict=True)
    )
    tasks.extend(
        Task(class_name="AI", music_root=ai_root, music_path=path, is_mixed=flag)
        for path, flag in zip(ai_files, ai_flags, strict=True)
    )

    opts = Options(
        output_dir=output_dir,
        speech_files=speech_files,
        audio_cfg=AudioConfig(sample_rate=int(args.output_sample_rate), mono=True),
        snr_min_db=float(args.snr_min_db),
        snr_max_db=float(args.snr_max_db),
        bitrate=str(args.output_bitrate),
        ffmpeg_bin=ffmpeg_bin or "ffmpeg",
        dry_run=bool(args.dry_run),
        run_seed=run_seed,
    )

    logger.info("=" * 60)
    logger.info("BUILD MONITORING MIXTURE DATASET")
    logger.info("=" * 60)
    logger.info("Human music files: %d", len(human_files))
    logger.info("AI music files: %d", len(ai_files))
    logger.info("Speech files: %d", len(speech_files))
    logger.info("Target mix ratio per class: %.4f", args.mix_ratio)
    logger.info("SNR range: [%.2f, %.2f] dB", args.snr_min_db, args.snr_max_db)
    logger.info("Output sample rate: %d", args.output_sample_rate)
    logger.info("Output bitrate: %s", args.output_bitrate)
    logger.info("Run seed: %d", run_seed)
    logger.info("Dry-run: %s", args.dry_run)
    logger.info("Total tasks: %d", len(tasks))

    results: list[dict[str, object]] = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Synthesis", unit="file"):
            results.append(_process_task(task, opts))
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(_process_task, task, opts) for task in tasks]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Synthesis",
                unit="file",
            ):
                results.append(fut.result())

    csv_path, json_path = _write_reports(results, output_dir, args.report_prefix)

    written = sum(1 for r in results if r["status"] == "written")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry_run_count = sum(1 for r in results if r["status"] == "dry_run")

    logger.info("=" * 60)
    logger.info("Dataset build complete")
    logger.info("Written: %d", written)
    logger.info("Dry-run rows: %d", dry_run_count)
    logger.info("Failed: %d", failed)
    logger.info("CSV report: %s", csv_path)
    logger.info("JSON report: %s", json_path)


if __name__ == "__main__":
    main()
