#!/usr/bin/env python3
"""Evaluate ETAP on EDS using fixed-duration event windows.

Only the event range required by the bundled DDFT tracking timestamps is read
from each large HDF5 file. Results use image-plane GT speed (px/s), matching the
EC duration experiment's most meaningful analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ec_duration_experiment import (
    build_frame_results,
    evaluate_prediction,
    load_model,
    load_tracks,
    plot_pixel_speed_results,
    run_trial,
    write_frame_results,
)


DEFAULT_SEQUENCES = (
    "01_peanuts_light",
    "02_rocket_earth_light",
    "08_peanuts_running",
    "14_ziggy_in_the_arena",
)

RESULT_FIELDS = (
    "sequence",
    "duration_ms",
    "timestamp_origin_s",
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
    "events_file",
    "prediction_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-duration tracking experiment on EDS."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/eds"))
    parser.add_argument(
        "--gt-tracks-root", type=Path, default=Path("config/misc/eds/gt_tracks")
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("config/misc/eds/calib.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/eds_duration_experiment")
    )
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
        default=list(range(20, 310, 10)),
    )
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--events-file",
        choices=("auto", "events.h5", "events_corrected.h5"),
        default="auto",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-stacks", type=int, default=10)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def h5_searchsorted(dataset: object, value: float, side: str = "left") -> int:
    """Binary search an ordered 1-D HDF5 dataset without loading it fully."""
    low, high = 0, len(dataset)
    while low < high:
        middle = (low + high) // 2
        current = float(dataset[middle])
        if current < value or (side == "right" and current == value):
            low = middle + 1
        else:
            high = middle
    return low


def resolve_events_file(sequence_dir: Path, requested: str) -> tuple[Path, bool]:
    if requested != "auto":
        path = sequence_dir / requested
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, requested == "events_corrected.h5"
    corrected = sequence_dir / "events_corrected.h5"
    if corrected.is_file():
        return corrected, True
    raw = sequence_dir / "events.h5"
    if raw.is_file():
        return raw, False
    raise FileNotFoundError(f"neither events_corrected.h5 nor events.h5 found in {sequence_dir}")


def load_required_events(
    path: Path, first_track_time_s: float, last_track_time_s: float, max_duration_ms: float
) -> np.ndarray:
    """Load only [first GT - max duration, last GT] from an EDS event file."""
    try:
        import hdf5plugin  # noqa: F401 - registers optional compression filters
    except ImportError:
        pass
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to read EDS events; install requirements.txt") from error

    start_us = first_track_time_s * 1e6 - max_duration_ms * 1e3
    end_us = last_track_time_s * 1e6
    with h5py.File(path, "r") as handle:
        required = ("t", "x", "y", "p")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"{path}: missing HDF5 datasets {missing}")
        timestamps = handle["t"]
        start = h5_searchsorted(timestamps, start_us, "left")
        end = h5_searchsorted(timestamps, end_us, "left")
        if end <= start:
            first_h5, last_h5 = float(timestamps[0]), float(timestamps[-1])
            raise ValueError(
                f"{path}: no events overlap GT range; H5 t=[{first_h5}, {last_h5}] us, "
                f"requested=[{start_us}, {end_us}] us"
            )
        # Rebase epoch timestamps so float32 retains sub-microsecond precision.
        origin_us = first_track_time_s * 1e6
        t = (
            (np.asarray(timestamps[start:end], dtype=np.float64) - origin_us) * 1e-6
        ).astype(np.float32)
        x = np.asarray(handle["x"][start:end], dtype=np.float32)
        y = np.asarray(handle["y"][start:end], dtype=np.float32)
        p = np.asarray(handle["p"][start:end], dtype=np.float32)
    events = np.column_stack((t, x, y, p))
    print(
        f"  Loaded {len(events):,} events ({start:,}:{end:,}) covering "
        f"{events[-1, 0] - events[0, 0]:.3f}s ({events.nbytes / 2**20:.1f} MiB)"
    )
    return events


def calibration_for_events(
    path: Path, already_corrected: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with path.open() as handle:
        calibration = yaml.safe_load(handle)
    event_intrinsics = calibration["cam1"]["intrinsics"]
    rgb_intrinsics = calibration["cam0"]["intrinsics"]

    def matrix(values: list[float]) -> np.ndarray:
        fx, fy, cx, cy = values
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    event_matrix = matrix(event_intrinsics)
    if already_corrected:
        return matrix(rgb_intrinsics), np.zeros(4), None
    distortion = np.asarray(calibration["cam1"]["distortion_coeffs"], dtype=np.float64)
    # Match prepare_event_representations.py: undistort cam1, then map to cam0 intrinsics.
    homography = matrix(rgb_intrinsics) @ np.linalg.inv(event_matrix)
    return event_matrix, distortion, homography


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    non_numeric = {"sequence", "events_file", "prediction_path"}
    for row in rows:
        for field in set(row) - non_numeric:
            row[field] = float(row[field])
    return rows


def read_frame_results(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    non_numeric = {"sequence", "clip_difficulty", "pixel_speed_class"}
    for row in rows:
        for field in set(row) - non_numeric:
            row[field] = float(row[field])
    return rows


def plot_sequence_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    sequences = sorted({str(row["sequence"]) for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for sequence in sequences:
        subset = sorted(
            (row for row in rows if row["sequence"] == sequence),
            key=lambda row: float(row["duration_ms"]),
        )
        durations = [float(row["duration_ms"]) for row in subset]
        axes[0].plot(durations, [float(row["accuracy_5px"]) for row in subset], marker="o", ms=3, label=sequence)
        axes[1].plot(durations, [float(row["mean_endpoint_error_px"]) for row in subset], marker="o", ms=3, label=sequence)
    axes[0].set_ylabel("Tracking accuracy @ 5 px")
    axes[1].set_ylabel("Mean endpoint error [px]")
    axes[1].set_xlabel("Event-window duration [ms]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "eds_metrics_vs_duration_by_sequence.png", dpi=180)
    plt.close(fig)


def create_frame_analysis(
    results: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    compatible = [
        {
            "sequence": row["sequence"],
            "difficulty": "full_gt",
            "duration_ms": row["duration_ms"],
            "prediction_path": row["prediction_path"],
        }
        for row in results
    ]
    frame_rows = build_frame_results(compatible)
    write_frame_results(output_dir / "frame_results.csv", frame_rows)
    plot_pixel_speed_results(frame_rows, output_dir)
    return frame_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    if args.plot_only:
        results = read_results(results_path)
        plot_sequence_results(results, args.output_dir)
        frame_path = args.output_dir / "frame_results.csv"
        if frame_path.is_file():
            plot_pixel_speed_results(read_frame_results(frame_path), args.output_dir)
        else:
            create_frame_analysis(results, args.output_dir)
        print(f"Regenerated EDS plots in {args.output_dir}")
        return

    if any(duration <= 0 for duration in args.durations_ms):
        raise ValueError("all durations must be positive")
    if not args.calibration.is_file():
        raise FileNotFoundError(args.calibration)
    trials = len(args.sequences) * len(set(args.durations_ms))
    model, model_config = load_model(args)
    results: list[dict[str, object]] = []
    trial_index = 0
    for sequence in args.sequences:
        sequence_dir = args.data_root / sequence
        tracks_path = args.gt_tracks_root / f"{sequence}.gt.txt"
        if not tracks_path.is_file():
            raise FileNotFoundError(tracks_path)
        times, track_ids, gt_tracks = load_tracks(tracks_path)
        timestamp_origin_s = float(times[0])
        relative_times = times - timestamp_origin_s
        events_path, corrected = resolve_events_file(sequence_dir, args.events_file)
        camera_matrix, distortion, homography = calibration_for_events(
            args.calibration, corrected
        )
        trial_data = None
        for duration_ms in sorted(set(args.durations_ms)):
            trial_index += 1
            label = f"{duration_ms:g}ms".replace(".", "p")
            prediction_path = args.output_dir / "predictions" / sequence / f"{label}.npz"
            print(f"[{trial_index}/{trials}] {sequence}, duration={duration_ms:g} ms")
            if not (args.resume and prediction_path.is_file()):
                if trial_data is None:
                    trial_data = load_required_events(
                        events_path,
                        float(times[0]),
                        float(times[-1]),
                        max(args.durations_ms),
                    )
                run_trial(
                    args,
                    model,
                    model_config,
                    trial_data,
                    relative_times,
                    track_ids,
                    gt_tracks,
                    camera_matrix,
                    distortion,
                    duration_ms,
                    prediction_path,
                    homography,
                )
            metrics = evaluate_prediction(prediction_path)
            results.append(
                {
                    "sequence": sequence,
                    "duration_ms": float(duration_ms),
                    "timestamp_origin_s": timestamp_origin_s,
                    **metrics,
                    "events_file": str(events_path),
                    "prediction_path": str(prediction_path),
                }
            )
            write_results(results_path, results)

    plot_sequence_results(results, args.output_dir)
    create_frame_analysis(results, args.output_dir)
    print(f"Results: {results_path}")
    print(f"Plots:   {args.output_dir}")


if __name__ == "__main__":
    main()
