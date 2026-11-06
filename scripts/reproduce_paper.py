#!/usr/bin/env python3
"""Reproduce Table 1, Figure 3 and Figure 4 of the paper with the released weights.

Scenarios (paper, Sec. 4.2):
  CFM  Clean Foreground Music   AI-OpenBMAT clean references (476 human / 476 AI tracks)
  STB  Synthetic TV Broadcast   AI-OpenBMAT mixtures (3,294 clips, 1,231 with AI music)
  RTB  Real TV Broadcast        BAMM, all clips (2,489 human / 3,009 AI)

Every clip is scored the same way: the log-mel spectrogram is cut into
non-overlapping 5 s windows (the last partial window is dropped, clips shorter
than 5 s are zero-padded), each window is scored by the CNN, and the clip score
is the mean of its window scores. A clip is called AI when its score is >= 0.5.

BAMM is also scored per segment, using the Deep Music Detector (DMD) labels in
metadata/bamm_dmd_segments.csv (8,653 segments: "music" = foreground music,
"bgmusic" = background music, "speech"). Each segment is cut out of the decoded
clip audio, its spectrogram is recomputed, and it is scored like a clip.
Figure 3 shows ROC curves over all segments, foreground segments and
background segments; Figure 4 shows the score histograms of the last two.

Usage:
    python scripts/download_weights.py
    python scripts/reproduce_paper.py \
        --openbmat-root /path/to/AI-OpenBMAT \
        --bamm-root /path/to/BAMM \
        --output-dir results/paper --workers 8

    # CFM needs the clean reference tracks, which are not in the public
    # AI-OpenBMAT release; pass them with --cfm-root if you have them.

    # Re-draw the tables and figures from saved scores, without inference:
    python scripts/reproduce_paper.py --output-dir results/paper --plots-only

Expected layouts:
    AI-OpenBMAT/{human,ai}/*.wav + *.json             (STB, as unzipped from Zenodo;
                                                       AI-OpenBMAT/data/{human,ai} also works)
    <cfm-root>/{human,ai}/*.wav                       (CFM, default AI-OpenBMAT/raw_materials)
    BAMM/metadata.csv + BAMM/audio/{human,ai}/*.mp4   (RTB)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import matplotlib

# TensorFlow must be imported before scikit-learn / matplotlib: loading the
# OpenMP runtime of scipy first can make TF's oneDNN segfault on some systems.
import tensorflow as tf  # noqa: I001

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FeaturesConfig, load_features_from_snapshot  # noqa: E402
from src.features.audio_loader import load_audio  # noqa: E402
from src.features.spectrogram import compute_spectrogram  # noqa: E402
from src.features.windowing import strided_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reproduce_paper")

MODELS = {"CNN Clean": "cnn_clean", "CNN Broadcast": "cnn_broadcast"}
SCENARIOS = {
    "CFM": "Clean Foreground Music",
    "STB": "Synthetic TV Broadcast",
    "RTB": "Real TV Broadcast",
}
SUBSETS = {
    "all": "All clips",
    "segments": "All segments",
    "music": "Foreground music",
    "bgmusic": "Background music",
}
THRESHOLD = 0.5
SCORES_FILENAME = "scores.csv"
COLORS = {"CNN Clean": "#4C9BE8", "CNN Broadcast": "#E8553E"}
HIST_COLORS = {0: "#9CC9F5", 1: "#F08A7A"}


# ── Dataset listings ──────────────────────────────────────────────────────────


def _wavs(folder: Path) -> list[Path]:
    """Sorted .wav files of a folder, ignoring macOS "._" resource-fork files."""
    return sorted(p for p in folder.glob("*.wav") if not p.name.startswith("._"))


def stb_dir(openbmat_root: Path) -> Path:
    """Folder holding human/ and ai/: the Zenodo layout, or a data/ subfolder."""
    for candidate in (openbmat_root, openbmat_root / "data"):
        if (candidate / "human").is_dir() and (candidate / "ai").is_dir():
            return candidate
    raise SystemExit(f"No human/ and ai/ folders found in {openbmat_root} or its data/ subfolder")


def list_cfm(cfm_root: Path) -> list[dict]:
    items = []
    for label_name, label in (("human", 0), ("ai", 1)):
        for path in _wavs(cfm_root / label_name):
            items.append({"id": f"{label_name}/{path.stem}", "path": path, "label": label})
    return items


def list_stb(openbmat_root: Path) -> list[dict]:
    """AI-OpenBMAT mixtures, labelled as in the paper.

    Clips in human/ are negatives. A clip in ai/ is a positive when its
    reference annotation contains music; the ones annotated only as "no-music"
    (416 clips) carry no AI music and count as negatives. Clips in ai/ are
    scored on the windows that overlap their annotated segments.
    """
    items = []
    root = stb_dir(openbmat_root)
    for label_name, label in (("human", 0), ("ai", 1)):
        for path in _wavs(root / label_name):
            item = {"id": f"{label_name}/{path.stem}", "path": path, "label": label}
            if label == 1:
                ann = path.with_suffix(".json")
                if not ann.exists():
                    continue
                segments = json.loads(ann.read_text())["reference_structure"]["segments"]
                item["segments"] = [(float(s["start"]), float(s["end"])) for s in segments]
                item["label"] = int(any(s["label"] != "no-music" for s in segments))
            items.append(item)
    return items


def list_rtb(bamm_root: Path, dmd_path: Path | None) -> list[dict]:
    dmd: dict[str, list[dict]] = {}
    if dmd_path is not None and dmd_path.exists():
        with open(dmd_path, newline="") as f:
            for row in csv.DictReader(f):
                dmd.setdefault(row["id"], []).append(row)
    elif dmd_path is not None:
        logger.warning("DMD segment file not found (%s): Figures 3-4 will only show all clips.",
                       dmd_path)

    items = []
    with open(bamm_root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            items.append({
                "id": row["id"],
                "path": bamm_root / row["file"],
                "label": 1 if row["label"] == "ai" else 0,
                "dmd": [
                    (r["dmd_class"], float(r["start_s"]), float(r["end_s"]))
                    for r in dmd.get(row["id"], [])
                ],
            })
    return items


# ── Feature extraction (runs in worker processes) ─────────────────────────────


def extract_item(path: Path, segments: list, features_cfg: FeaturesConfig):
    """Decode one clip and return its spectrogram plus one spectrogram per DMD segment."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        y, duration = load_audio(path, features_cfg.audio)
        spec = compute_spectrogram(y, features_cfg)
        sr = features_cfg.audio.sample_rate
        seg_specs = []
        for _, start_s, end_s in segments:
            piece = y[int(start_s * sr):int(end_s * sr)]
            seg_specs.append(compute_spectrogram(piece, features_cfg) if len(piece) else None)
        return str(path), spec, seg_specs, duration, None
    except Exception as exc:  # report and continue: one bad file must not stop the run
        return str(path), None, None, None, str(exc)


# ── Scoring ───────────────────────────────────────────────────────────────────


def window_starts(n_frames: int, crop_frames: int) -> list[int]:
    """Start frames of the windows returned by strided_windows(hop = crop)."""
    if n_frames < crop_frames:
        return [0]
    return list(range(0, n_frames - crop_frames + 1, crop_frames))


def predict_windows(models: dict, spec: np.ndarray, crop_frames: int) -> dict[str, np.ndarray]:
    # model.predict() is what produced the paper's numbers. A direct model(x) call is
    # faster but runs different kernels and can move a score by ~1e-3.
    windows = strided_windows(spec, crop_frames)
    return {
        name: m.predict(windows, batch_size=64, verbose=0).reshape(-1).astype(float)
        for name, m in models.items()
    }


def score_item(scenario: str, item: dict, spec: np.ndarray, seg_specs: list,
               duration: float, models: dict, cfg: dict) -> list[dict]:
    """Return one score row per model for the clip, and per model and DMD segment."""
    crop, hop, sr = cfg["crop_frames"], cfg["hop_length"], cfg["sample_rate"]
    probs = predict_windows(models, spec, crop)
    rows = []

    keep = np.ones(len(next(iter(probs.values()))), dtype=bool)
    if scenario == "STB" and "segments" in item:
        # Keep the windows that overlap the annotated part of the mixture.
        starts = window_starts(spec.shape[0], crop)
        for i, s in enumerate(starts):
            w0 = s * hop / sr
            w1 = min(duration, (s + crop) * hop / sr)
            keep[i] = any(min(w1, b) - max(w0, a) > 0 for a, b in item["segments"])
        if not keep.any():
            return rows

    for name, p in probs.items():
        rows.append({"scenario": scenario, "subset": "all", "id": item["id"],
                     "label": item["label"], "model": name, "score": float(np.mean(p[keep]))})

    for k, ((dmd_class, _, _), seg) in enumerate(zip(item.get("dmd", []), seg_specs or [])):
        if seg is None:
            continue
        for name, p in predict_windows(models, seg, crop).items():
            for subset in ("segments", dmd_class):
                if subset not in SUBSETS:
                    continue
                rows.append({"scenario": scenario, "subset": subset, "id": f"{item['id']}#{k}",
                             "label": item["label"], "model": name, "score": float(np.mean(p))})
    return rows


def run_inference(args: argparse.Namespace, out_dir: Path) -> None:
    models, feature_cfgs, windows = {}, [], []
    for name, sub in MODELS.items():
        d = Path(args.weights_dir) / sub
        models[name] = tf.keras.models.load_model(d / "model.h5", compile=False)
        fcfg, window_sec = load_features_from_snapshot(d / "config_snapshot.yaml")
        feature_cfgs.append(fcfg)
        windows.append(window_sec)
    if any(asdict(c) != asdict(feature_cfgs[0]) for c in feature_cfgs) or len(set(windows)) != 1:
        raise ValueError("Both models must share one feature config to be scored together.")
    features_cfg = feature_cfgs[0]
    cfg = {
        "crop_frames": features_cfg.frames_for_duration(windows[0]),
        "hop_length": features_cfg.stft.hop_length,
        "sample_rate": features_cfg.audio.sample_rate,
    }
    logger.info("Window: %.1f s = %d frames", windows[0], cfg["crop_frames"])

    listings = {}
    if "CFM" in args.scenarios:
        cfm_root = Path(args.cfm_root) if args.cfm_root else (
            Path(args.openbmat_root) / "raw_materials" if args.openbmat_root else None
        )
        if cfm_root is not None and (cfm_root / "human").is_dir() and (cfm_root / "ai").is_dir():
            listings["CFM"] = list_cfm(cfm_root)
        else:
            logger.warning("Skipping CFM: no clean reference tracks found (pass --cfm-root with "
                           "human/ and ai/ folders). They are not part of the public AI-OpenBMAT.")
    if "STB" in args.scenarios:
        if not args.openbmat_root:
            raise SystemExit("--openbmat-root is required for the STB scenario")
        listings["STB"] = list_stb(Path(args.openbmat_root))
    if "RTB" in args.scenarios:
        if not args.bamm_root:
            raise SystemExit("--bamm-root is required for the RTB scenario")
        listings["RTB"] = list_rtb(Path(args.bamm_root), Path(args.dmd_segments))

    scores_path = out_dir / SCORES_FILENAME
    errors = []
    fields = ["scenario", "subset", "id", "label", "model", "score"]
    with open(scores_path, "w", newline="") as f, \
            ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario, items in listings.items():
            if args.limit:
                items = [i for i in items if i["label"] == 0][: args.limit] + \
                        [i for i in items if i["label"] == 1][: args.limit]
            logger.info("%s (%s): %d clips", scenario, SCENARIOS[scenario], len(items))
            results = ex.map(extract_item, [i["path"] for i in items],
                             [i.get("dmd", []) for i in items],
                             [features_cfg] * len(items), chunksize=4)
            for item, (path, spec, seg_specs, duration, err) in tqdm(
                zip(items, results), total=len(items), desc=scenario
            ):
                if err is not None:
                    errors.append({"scenario": scenario, "path": path, "error": err})
                    continue
                writer.writerows(
                    score_item(scenario, item, spec, seg_specs, duration, models, cfg)
                )
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2))
        logger.warning("%d clips failed to load (see errors.json)", len(errors))
    logger.info("Scores: %s", scores_path)


# ── Tables and figures ────────────────────────────────────────────────────────


def load_scores(out_dir: Path) -> list[dict]:
    with open(out_dir / SCORES_FILENAME, newline="") as f:
        return [{**r, "label": int(r["label"]), "score": float(r["score"])}
                for r in csv.DictReader(f)]


def select(rows, scenario, model, subset="all"):
    sel = [r for r in rows if r["scenario"] == scenario and r["model"] == model
           and r["subset"] == subset]
    return np.array([r["label"] for r in sel]), np.array([r["score"] for r in sel])


def write_table(rows: list[dict], out_dir: Path) -> list[dict]:
    table = []
    for scenario in SCENARIOS:
        for subset in SUBSETS:
            for model in MODELS:
                y, s = select(rows, scenario, model, subset)
                if len(y) == 0 or len(set(y)) < 2:
                    continue
                pred = (s >= THRESHOLD).astype(int)
                table.append({
                    "scenario": scenario, "subset": subset, "model": model,
                    "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                    "accuracy": accuracy_score(y, pred),
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "f1": f1_score(y, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y, s),
                })
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    lines = ["| Scenario | CNN Clean F1 | CNN Clean ROC | CNN Broadcast F1 | CNN Broadcast ROC |",
             "|---|---|---|---|---|"]
    for scenario in SCENARIOS:
        cells = []
        for model in MODELS:
            m = next((t for t in table if t["scenario"] == scenario and t["model"] == model
                      and t["subset"] == "all"), None)
            cells += [f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}"] if m else ["-", "-"]
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    seg_lines = [
        "| BAMM subset | n | CNN Clean F1 | CNN Clean ROC "
        "| CNN Broadcast F1 | CNN Broadcast ROC |",
        "|---|---|---|---|---|---|",
    ]
    for subset in ("all", "segments", "music", "bgmusic"):
        cells, n = [], ""
        for model in MODELS:
            m = next((t for t in table if t["scenario"] == "RTB" and t["model"] == model
                      and t["subset"] == subset), None)
            if m:
                n = str(m["n_human"] + m["n_ai"])
            cells += [f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}"] if m else ["-", "-"]
        seg_lines.append(f"| {SUBSETS[subset]} | {n} | " + " | ".join(cells) + " |")

    text = ("Table 1: clip level, threshold 0.5\n\n" + "\n".join(lines) +
            "\n\nBAMM by DMD segment (Figures 3-4)\n\n" + "\n".join(seg_lines) + "\n")
    (out_dir / "table1.md").write_text(text)
    print("\n" + text)
    return table


def plot_roc(rows: list[dict], out_dir: Path) -> None:
    subsets = [s for s in ("segments", "music", "bgmusic")
               if len(select(rows, "RTB", "CNN Clean", s)[0])]
    if not subsets:
        return
    fig, axes = plt.subplots(1, len(subsets), figsize=(5 * len(subsets), 4.6), squeeze=False)
    for ax, subset in zip(axes[0], subsets):
        for model in MODELS:
            y, s = select(rows, "RTB", model, subset)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=COLORS[model], lw=1.6,
                    label=f"{model} (AUC={roc_auc_score(y, s):.3f})")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1, label="Random")
        ax.set_title(SUBSETS[subset])
        ax.set_xlabel("False Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0][0].set_ylabel("True Positive Rate")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure3_roc_bamm.{ext}", dpi=200)
    plt.close(fig)


def plot_histograms(rows: list[dict], out_dir: Path) -> None:
    subsets = [s for s in ("music", "bgmusic") if len(select(rows, "RTB", "CNN Clean", s)[0])]
    if not subsets:
        return
    names = {"music": "Music", "bgmusic": "Background music"}
    fig, axes = plt.subplots(len(subsets), len(MODELS), figsize=(9, 3.6 * len(subsets)),
                             squeeze=False)
    bins = np.linspace(0, 1, 21)
    for i, subset in enumerate(subsets):
        for j, model in enumerate(MODELS):
            ax = axes[i][j]
            y, s = select(rows, "RTB", model, subset)
            for label, name in ((0, "Human"), (1, "AI")):
                ax.hist(s[y == label], bins=bins, color=HIST_COLORS[label], alpha=0.75,
                        label=name)
            ax.set_title(model)
            if j == 0:
                ax.set_ylabel(names[subset])
            if i == len(subsets) - 1:
                ax.set_xlabel("Probability (AI)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure4_histograms_bamm.{ext}", dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights-dir", default="weights",
                   help="Folder with cnn_clean/ and cnn_broadcast/ (default: weights)")
    p.add_argument("--openbmat-root", default=None,
                   help="AI-OpenBMAT folder with human/ and ai/ (STB)")
    p.add_argument("--cfm-root", default=None,
                   help="Clean reference tracks with human/ and ai/ (CFM); "
                        "default: <openbmat-root>/raw_materials")
    p.add_argument("--bamm-root", default=None, help="BAMM root, with metadata.csv (RTB)")
    p.add_argument("--dmd-segments",
                   default=str(PROJECT_ROOT / "metadata" / "bamm_dmd_segments.csv"),
                   help="DMD foreground/background segments for BAMM")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--output-dir", default="results/paper")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="Feature-extraction processes")
    p.add_argument("--limit", type=int, default=None,
                   help="Max clips per class and scenario, for a quick test run")
    p.add_argument("--plots-only", action="store_true",
                   help=f"Skip inference and rebuild outputs from an existing {SCORES_FILENAME}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        run_inference(args, out_dir)
    rows = load_scores(out_dir)
    write_table(rows, out_dir)
    plot_roc(rows, out_dir)
    plot_histograms(rows, out_dir)
    logger.info("Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
