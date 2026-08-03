#!/usr/bin/env python3
"""Run ETAP on speed-ranked EC clips using fixed-duration event windows.

This is an end-to-end experiment driver for clips produced by
``slice_ec_by_difficulty.py``.  At every DDFT ground-truth timestamp it builds
an event stack from exactly the preceding N milliseconds, runs online ETAP,
computes tracking metrics, and plots duration/accuracy/speed relationships.

Representations are generated lazily and are not stored on disk.  Predictions
are saved per trial so interrupted experiments can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.representations import EventRepresentationFactory


DIFFICULTIES = ("slow", "medium", "fast")
RESULT_FIELDS = (
    "sequence",
    "difficulty",
    "duration_ms",
    "duration_s",
    "motion_score",
    "translation_speed_mps",
    "angular_speed_degps",
    "accuracy_1px",
    "accuracy_3px",
    "accuracy_5px",
    "accuracy_10px",
    "mean_endpoint_error_px",
    "median_endpoint_error_px",
    "relative_feature_age_5px",
    "tracking_error_5px",
    "mean_events_per_window",
    "median_events_per_window",
    "num_timestamps",
    "num_tracks",
    "prediction_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-duration event windows on EC difficulty clips."
    )
    parser.add_argument("--clips-root", type=Path, default=Path("data/ec_difficulty"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ec_duration_experiment"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exe/inference_online/feature_tracking.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--durations-ms",
        nargs="+",
        type=float,
        default=[5, 10, 20, 50, 100, 200, 220, 240],
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--num-stacks", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Reuse saved trial predictions.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from results.csv; no GPU/checkpoint required.",
    )
    return parser.parse_args()


def discover_clips(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    if args.sequences is None:
        sequences = sorted(path.name for path in args.clips_root.iterdir() if path.is_dir())
    else:
        sequences = args.sequences
    clips = []
    for sequence in sequences:
        for difficulty in args.difficulties:
            clip_dir = args.clips_root / sequence / difficulty
            if clip_dir.is_dir():
                clips.append((sequence, difficulty, clip_dir))
            elif args.sequences is not None:
                raise FileNotFoundError(clip_dir)
    if not clips:
        raise FileNotFoundError(f"no difficulty clips found under {args.clips_root}")
    return clips


def load_events(path: Path) -> np.ndarray:
    events = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if events.shape[1] < 4:
        raise ValueError(f"{path}: expected timestamp x y polarity")
    events = events[:, :4]
    events = events[np.argsort(events[:, 0])]
    return events


def load_tracks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[1] < 4:
        raise ValueError(f"{path}: expected id timestamp x y")
    rows = rows[:, :4]
    times = np.unique(rows[:, 1])
    track_ids = np.unique(rows[:, 0]).astype(np.int64)
    id_to_col = {track_id: col for col, track_id in enumerate(track_ids)}
    time_to_row = {timestamp: row for row, timestamp in enumerate(times)}
    tracks = np.full((len(times), len(track_ids), 2), np.nan, dtype=np.float32)
    for track_id, timestamp, x, y in rows:
        tracks[time_to_row[timestamp], id_to_col[int(track_id)]] = (x, y)

    # ETAP queries every point at frame zero.  Remove tracks unavailable there.
    valid_at_start = np.all(np.isfinite(tracks[0]), axis=1)
    tracks = tracks[:, valid_at_start]
    track_ids = track_ids[valid_at_start]
    if len(track_ids) == 0:
        raise ValueError(f"{path}: no tracks are defined at the first timestamp")
    return times, track_ids, tracks


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, comments="#").reshape(-1)
    if len(values) < 4:
        raise ValueError(f"{path}: expected at least fx fy cx cy")
    fx, fy, cx, cy = values[:4]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    distortion = values[4:]
    return camera_matrix, distortion


def load_clip_motion(clip_dir: Path) -> dict[str, float]:
    metadata_path = clip_dir / "clip.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is required; recreate clips with slice_ec_by_difficulty.py"
        )
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    return {
        "motion_score": float(metadata["mean_motion_score"]),
        "translation_speed_mps": float(metadata["mean_translation_speed_mps"]),
        "angular_speed_degps": float(metadata["mean_angular_speed_degps"]),
    }


class FixedDurationRepresentationBuilder:
    def __init__(
        self,
        events: np.ndarray,
        duration_s: float,
        height: int,
        width: int,
        num_stacks: int,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> None:
        self.events = events
        self.timestamps = events[:, 0]
        self.duration_s = duration_s
        self.height = height
        self.width = width
        self.converter = EventRepresentationFactory.create(
            {
                "representation_name": "event_stack",
                "num_stacks": num_stacks,
                "interpolation": "bilinear",
                "channel_overlap": True,
                "centered_channels": False,
                "image_shape": (height, width),
            }
        )
        self.maps = None
        if len(distortion) > 0 and not np.allclose(distortion, 0):
            self.maps = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                camera_matrix,
                (width, height),
                cv2.CV_32FC1,
            )

    def build(self, timestamp: float) -> tuple[np.ndarray, int]:
        end = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        start = int(np.searchsorted(self.timestamps, timestamp - self.duration_s, side="left"))
        selected = self.events[start:end]
        count = len(selected)
        if count == 0:
            representation = np.zeros(
                (self.converter.num_stacks, self.height, self.width), dtype=np.float32
            )
        else:
            # Converter order is y, x, timestamp, polarity.
            converted_events = np.stack(
                (selected[:, 2], selected[:, 1], selected[:, 0], selected[:, 3]), axis=1
            )
            in_bounds = (
                (converted_events[:, 0] >= 0)
                & (converted_events[:, 0] < self.height)
                & (converted_events[:, 1] >= 0)
                & (converted_events[:, 1] < self.width)
            )
            converted_events = converted_events[in_bounds]
            representation = self.converter(converted_events).astype(np.float32)
        if self.maps is not None:
            map_x, map_y = self.maps
            representation = np.stack(
                [cv2.remap(channel, map_x, map_y, cv2.INTER_CUBIC) for channel in representation]
            )
        return representation, count


def normalize_voxels(voxels: "torch.Tensor") -> "torch.Tensor":
    import torch

    mask = voxels != 0
    counts = mask.sum(dim=(0, 2, 3), keepdim=True).clamp_min(1)
    mean = voxels.sum(dim=(0, 2, 3), keepdim=True) / counts
    variance = ((voxels - mean) ** 2 * mask).sum(dim=(0, 2, 3), keepdim=True) / counts
    normalized = (voxels - mean) / torch.sqrt(variance + 1e-8)
    return torch.where(mask, normalized, voxels)


def load_model(args: argparse.Namespace) -> tuple[object, dict[str, object]]:
    import torch

    from src.model.etap.model import Etap

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    checkpoint = args.checkpoint or Path(config["common"]["ckp_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model_config = dict(config["model"])
    model_config["model_resolution"] = (512, 512)
    if int(model_config.get("num_in_channels", args.num_stacks)) != args.num_stacks:
        raise ValueError("--num-stacks must match model.num_in_channels in the config")
    model = Etap(**model_config)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model = model.to(torch.device(args.device)).eval()
    return model, model_config


def usable_length(total: int, window_len: int) -> int:
    step = window_len // 2
    return window_len + ((total - window_len) // step) * step if total >= window_len else 0


def run_trial(
    args: argparse.Namespace,
    model: object,
    model_config: dict[str, object],
    clip_dir: Path,
    duration_ms: float,
    prediction_path: Path,
) -> None:
    import torch

    events = load_events(clip_dir / "events.txt")
    times, track_ids, gt_tracks = load_tracks(clip_dir / "tracks.gt.txt")
    window_len = int(model_config.get("window_len", 8))
    step = window_len // 2
    length = usable_length(len(times), window_len)
    if length == 0:
        raise ValueError(f"{clip_dir}: fewer than {window_len} tracking timestamps")
    times, gt_tracks = times[:length], gt_tracks[:length]
    camera_matrix, distortion = load_calibration(clip_dir / "calib.txt")
    builder = FixedDurationRepresentationBuilder(
        events,
        duration_ms / 1000.0,
        args.height,
        args.width,
        args.num_stacks,
        camera_matrix,
        distortion,
    )

    device = torch.device(args.device)
    queries_xy = torch.from_numpy(gt_tracks[0]).float().to(device)
    queries_t = torch.zeros((len(track_ids), 1), dtype=torch.float32, device=device)
    queries = torch.cat((queries_t, queries_xy), dim=1)[None]
    model.init_video_online_processing()
    previous_chunk: list[np.ndarray] | None = None
    event_counts = np.zeros(length, dtype=np.int64)
    result = None
    for start in range(0, length - window_len + 1, step):
        if previous_chunk is None:
            chunk = []
            for index in range(start, start + window_len):
                representation, count = builder.build(float(times[index]))
                chunk.append(representation)
                event_counts[index] = count
        else:
            chunk = previous_chunk[step:]
            for index in range(start + window_len - step, start + window_len):
                representation, count = builder.build(float(times[index]))
                chunk.append(representation)
                event_counts[index] = count
        previous_chunk = chunk
        voxels = torch.from_numpy(np.stack(chunk)).float().to(device)
        voxels = normalize_voxels(voxels)
        with torch.inference_mode():
            result = model(video=voxels[None], queries=queries, is_online=True, iters=args.iters)

    if result is None:
        raise RuntimeError("no inference windows were generated")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        coords=result["coords_predicted"][0].detach().cpu().numpy()[:length],
        visibility=result["vis_predicted"][0].detach().cpu().numpy()[:length],
        timestamps=times,
        track_ids=track_ids,
        gt_tracks=gt_tracks,
        event_counts=event_counts,
    )


def prediction_rows(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = data["coords"], data["gt_tracks"]
    times, track_ids = data["timestamps"], data["track_ids"]
    pred_rows, gt_rows = [], []
    for column, track_id in enumerate(track_ids):
        finite = np.all(np.isfinite(gt[:, column]), axis=1)
        for index in np.flatnonzero(finite):
            pred_rows.append((track_id, times[index], pred[index, column, 0], pred[index, column, 1]))
            gt_rows.append((track_id, times[index], gt[index, column, 0], gt[index, column, 1]))
    return np.asarray(pred_rows), np.asarray(gt_rows)


def evaluate_prediction(path: Path) -> dict[str, float]:
    from src.utils.track_utils import compute_tracking_errors

    with np.load(path) as data:
        pred = data["coords"]
        gt = data["gt_tracks"]
        valid = np.all(np.isfinite(gt), axis=-1)
        valid[0] = False  # Query locations have zero error by construction.
        errors = np.linalg.norm(pred - gt, axis=-1)[valid]
        if len(errors) == 0:
            raise ValueError(f"{path}: no valid points to evaluate")
        pred_rows, gt_rows = prediction_rows(data)
        feature_age, tracking_error = compute_tracking_errors(
            pred_rows, gt_rows, asynchronous=False, error_threshold=5
        )
        event_counts = data["event_counts"]
        return {
            "accuracy_1px": float(np.mean(errors <= 1)),
            "accuracy_3px": float(np.mean(errors <= 3)),
            "accuracy_5px": float(np.mean(errors <= 5)),
            "accuracy_10px": float(np.mean(errors <= 10)),
            "mean_endpoint_error_px": float(np.mean(errors)),
            "median_endpoint_error_px": float(np.median(errors)),
            "relative_feature_age_5px": float(np.mean(feature_age)) if len(feature_age) else 0.0,
            "tracking_error_5px": float(np.mean(tracking_error)) if len(tracking_error) else math.nan,
            "mean_events_per_window": float(np.mean(event_counts)),
            "median_events_per_window": float(np.median(event_counts)),
            "num_timestamps": int(len(data["timestamps"])),
            "num_tracks": int(len(data["track_ids"])),
        }


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(RESULT_FIELDS) - {"sequence", "difficulty", "prediction_path"}
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])
    return rows


def aggregate(rows: list[dict[str, object]], field: str, duration: float, difficulty: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row["difficulty"] == difficulty and float(row["duration_ms"]) == duration]
    )


def plot_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    durations = sorted({float(row["duration_ms"]) for row in rows})
    colors = {"slow": "tab:blue", "medium": "tab:orange", "fast": "tab:red"}
    present_difficulties = [
        difficulty for difficulty in DIFFICULTIES
        if any(row["difficulty"] == difficulty for row in rows)
    ]

    fig, axis = plt.subplots(figsize=(8, 5))
    for difficulty in present_difficulties:
        means, stds = [], []
        for duration in durations:
            values = aggregate(rows, "accuracy_5px", duration, difficulty)
            means.append(float(np.mean(values)) if len(values) else np.nan)
            stds.append(float(np.std(values)) if len(values) else np.nan)
        axis.errorbar(durations, means, yerr=stds, marker="o", capsize=3, label=difficulty, color=colors[difficulty])
    axis.set_xscale("log")
    axis.set_xlabel("Event-window duration [ms]")
    axis.set_ylabel("Tracking accuracy @ 5 px")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_duration.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for index, duration in enumerate(durations):
        subset = [row for row in rows if float(row["duration_ms"]) == duration]
        axis.scatter(
            [float(row["motion_score"]) for row in subset],
            [float(row["accuracy_5px"]) for row in subset],
            s=28,
            alpha=0.8,
            color=cmap(index / max(1, len(durations) - 1)),
            label=f"{duration:g} ms",
        )
    axis.set_xlabel("Mean motion score")
    axis.set_ylabel("Tracking accuracy @ 5 px")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_motion_speed.png", dpi=180)
    plt.close(fig)

    fig, axes_grid = plt.subplots(
        1,
        len(present_difficulties),
        figsize=(4.7 * len(present_difficulties), 4),
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    for axis, difficulty in zip(axes, present_difficulties):
        sequences = sorted({str(row["sequence"]) for row in rows if row["difficulty"] == difficulty})
        matrix = np.full((len(sequences), len(durations)), np.nan)
        for i, sequence in enumerate(sequences):
            for j, duration in enumerate(durations):
                values = [
                    float(row["accuracy_5px"])
                    for row in rows
                    if row["sequence"] == sequence
                    and row["difficulty"] == difficulty
                    and float(row["duration_ms"]) == duration
                ]
                if values:
                    matrix[i, j] = np.mean(values)
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
        axis.set_title(difficulty)
        axis.set_xticks(range(len(durations)), [f"{d:g}" for d in durations], rotation=45)
        axis.set_yticks(range(len(sequences)), sequences, fontsize=7)
        axis.set_xlabel("Duration [ms]")
    fig.colorbar(image, ax=axes, label="Accuracy @ 5 px", fraction=0.025)
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.20, wspace=0.25)
    fig.savefig(output_dir / "accuracy_duration_difficulty_heatmap.png", dpi=180)
    plt.close(fig)

    # Find the duration with the best accuracy for every sequence/difficulty.
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["sequence"]), str(row["difficulty"])), []).append(row)
    optimal = [max(group, key=lambda row: float(row["accuracy_5px"])) for group in groups.values()]
    speeds = np.asarray([float(row["motion_score"]) for row in optimal])
    best_durations = np.asarray([float(row["duration_ms"]) for row in optimal])
    can_correlate = (
        len(optimal) > 1
        and float(np.std(speeds)) > 0
        and float(np.std(best_durations)) > 0
    )
    correlation = float(np.corrcoef(speeds, best_durations)[0, 1]) if can_correlate else math.nan
    slope, intercept = np.polyfit(speeds, best_durations, 1) if len(optimal) > 1 else (math.nan, math.nan)

    fig, axis = plt.subplots(figsize=(8, 5))
    for difficulty in present_difficulties:
        subset = [row for row in optimal if row["difficulty"] == difficulty]
        axis.scatter(
            [float(row["motion_score"]) for row in subset],
            [float(row["duration_ms"]) for row in subset],
            label=difficulty,
            color=colors[difficulty],
            s=42,
        )
    if len(optimal) > 1:
        x_line = np.linspace(np.min(speeds), np.max(speeds), 100)
        axis.plot(x_line, intercept + slope * x_line, "k--", label=f"fit slope={slope:.2f}")
    axis.set_xlabel("Mean motion score")
    axis.set_ylabel("Best duration [ms]")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_duration_vs_motion_speed.png", dpi=180)
    plt.close(fig)

    # Interaction regression: accuracy ~ log(duration) + speed + interaction.
    y = np.asarray([float(row["accuracy_5px"]) for row in rows])
    log_duration = np.log(np.asarray([float(row["duration_ms"]) for row in rows]))
    speed = np.asarray([float(row["motion_score"]) for row in rows])
    design = np.column_stack((np.ones(len(rows)), log_duration, speed, log_duration * speed))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    summary = {
        "optimal_duration_motion_score_pearson_r": correlation,
        "optimal_duration_linear_slope_ms_per_score": float(slope),
        "accuracy_regression": {
            "intercept": float(coefficients[0]),
            "log_duration": float(coefficients[1]),
            "motion_score": float(coefficients[2]),
            "log_duration_x_motion_score": float(coefficients[3]),
        },
        "interpretation": (
            "A negative optimal-duration slope/correlation supports the hypothesis that "
            "faster motion prefers shorter event windows. Regression coefficients are "
            "descriptive, not causal significance tests."
        ),
    }
    with (output_dir / "relationship_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    if args.plot_only:
        if not results_path.is_file():
            raise FileNotFoundError(results_path)
        rows = read_results(results_path)
        plot_results(rows, args.output_dir)
        print(f"Regenerated plots in {args.output_dir}")
        return

    if any(duration <= 0 for duration in args.durations_ms):
        raise ValueError("all durations must be positive")
    clips = discover_clips(args)
    model, model_config = load_model(args)
    rows: list[dict[str, object]] = []
    total_trials = len(clips) * len(args.durations_ms)
    trial_index = 0
    for sequence, difficulty, clip_dir in clips:
        motion = load_clip_motion(clip_dir)
        for duration_ms in sorted(set(args.durations_ms)):
            trial_index += 1
            duration_label = f"{duration_ms:g}ms".replace(".", "p")
            prediction_path = args.output_dir / "predictions" / sequence / difficulty / f"{duration_label}.npz"
            print(
                f"[{trial_index}/{total_trials}] {sequence}/{difficulty}, "
                f"duration={duration_ms:g} ms"
            )
            if not (args.resume and prediction_path.is_file()):
                run_trial(args, model, model_config, clip_dir, duration_ms, prediction_path)
            metrics = evaluate_prediction(prediction_path)
            row: dict[str, object] = {
                "sequence": sequence,
                "difficulty": difficulty,
                "duration_ms": float(duration_ms),
                "duration_s": float(duration_ms) / 1000.0,
                **motion,
                **metrics,
                "prediction_path": str(prediction_path),
            }
            rows.append(row)
            write_results(results_path, rows)

    plot_results(rows, args.output_dir)
    print(f"Results: {results_path}")
    print(f"Plots:   {args.output_dir}")


if __name__ == "__main__":
    main()
