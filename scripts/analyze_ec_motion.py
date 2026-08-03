#!/usr/bin/env python3
"""Analyze EC camera motion and create a speed-aware slicing manifest.

The EC ground-truth format is:
    timestamp px py pz qx qy qz qw

For each sequence this script writes sample-level motion estimates, fixed-size
analysis windows, and a plot.  It also writes combined CSV files and a summary
plot for all sequences.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


DEFAULT_SEQUENCES = (
    "boxes_translation",
    "shapes_6dof",
    "shapes_rotation",
    "shapes_translation",
    "boxes_rotation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EC ground-truth speed and create a slicing manifest."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ec"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ec_motion"))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--smooth-seconds",
        type=float,
        default=0.10,
        help="Savitzky-Golay smoothing span in seconds (default: 0.10).",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=0.10,
        help="Window size used for the slicing manifest, in seconds.",
    )
    parser.add_argument(
        "--window-stride",
        type=float,
        default=None,
        help="Manifest stride in seconds (default: same as --window-duration).",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Translation speed in m/s corresponding to motion score 1.",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=180.0,
        help="Angular speed in deg/s corresponding to motion score 1.",
    )
    parser.add_argument(
        "--base-duration",
        type=float,
        default=0.10,
        help="Recommended event duration at motion score 1, in seconds.",
    )
    parser.add_argument("--min-duration", type=float, default=0.01)
    parser.add_argument("--max-duration", type=float, default=0.50)
    parser.add_argument(
        "--score-floor",
        type=float,
        default=0.05,
        help="Floor used in inverse-speed duration calculation.",
    )
    return parser.parse_args()


def load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"{path}: expected at least 8 columns, got {data.shape[1]}")
    data = data[np.all(np.isfinite(data[:, :8]), axis=1)]
    data = data[np.argsort(data[:, 0])]
    # Duplicated timestamps make numerical differentiation undefined.
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(unique_idx)]
    if len(data) < 3:
        raise ValueError(f"{path}: need at least 3 distinct ground-truth samples")
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def odd_smoothing_window(t: np.ndarray, seconds: float) -> int:
    if seconds <= 0 or len(t) < 5:
        return 0
    median_dt = float(np.median(np.diff(t)))
    window = max(5, int(round(seconds / median_dt)))
    if window % 2 == 0:
        window += 1
    if window > len(t):
        window = len(t) if len(t) % 2 == 1 else len(t) - 1
    return window if window >= 5 else 0


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 0:
        return values
    return savgol_filter(values, window_length=window, polyorder=min(3, window - 2), axis=0)


def estimate_motion(
    t: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    smooth_seconds: float,
    translation_scale: float,
    rotation_scale: float,
) -> dict[str, np.ndarray]:
    window = odd_smoothing_window(t, smooth_seconds)
    position_smooth = smooth(position, window)
    velocity = np.gradient(position_smooth, t, axis=0)
    translation_speed = np.linalg.norm(velocity, axis=1)

    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    # q and -q encode the same orientation, hence abs(dot).
    dots = np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=1))
    angles_rad = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    angular_interval = np.degrees(angles_rad / np.diff(t))
    angular_speed = np.interp(t, 0.5 * (t[:-1] + t[1:]), angular_interval)
    angular_speed = smooth(angular_speed, window)
    angular_speed = np.maximum(angular_speed, 0.0)

    motion_score = np.hypot(
        translation_speed / translation_scale,
        angular_speed / rotation_scale,
    )
    return {
        "timestamp": t,
        "time_s": t - t[0],
        "translation_speed_mps": translation_speed,
        "angular_speed_degps": angular_speed,
        "motion_score": motion_score,
    }


def make_windows(
    sequence: str,
    motion: dict[str, np.ndarray],
    window_duration: float,
    stride: float,
    base_duration: float,
    min_duration: float,
    max_duration: float,
    score_floor: float,
) -> list[dict[str, float | str]]:
    t_abs = motion["timestamp"]
    first, last = float(t_abs[0]), float(t_abs[-1])
    if last - first < window_duration:
        starts = np.array([first])
    else:
        starts = np.arange(first, last - window_duration + 1e-12, stride)
    rows: list[dict[str, float | str]] = []
    for start in starts:
        end = min(start + window_duration, last)
        mask = (t_abs >= start) & (t_abs < end)
        if end == last:
            mask |= t_abs == last
        if not np.any(mask):
            continue
        trans = float(np.median(motion["translation_speed_mps"][mask]))
        angular = float(np.median(motion["angular_speed_degps"][mask]))
        score = float(np.median(motion["motion_score"][mask]))
        recommended = float(
            np.clip(base_duration / max(score, score_floor), min_duration, max_duration)
        )
        rows.append(
            {
                "sequence": sequence,
                "start_timestamp": float(start),
                "end_timestamp": float(end),
                "start_time_s": float(start - first),
                "end_time_s": float(end - first),
                "translation_speed_mps": trans,
                "angular_speed_degps": angular,
                "motion_score": score,
                "recommended_duration_s": recommended,
            }
        )
    return rows


def assign_speed_classes(rows: list[dict[str, float | str]]) -> None:
    scores = np.asarray([float(row["motion_score"]) for row in rows])
    low_cut, high_cut = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    for row in rows:
        score = float(row["motion_score"])
        row["speed_class"] = "slow" if score <= low_cut else "medium" if score <= high_cut else "fast"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence(sequence: str, motion: dict[str, np.ndarray], path: Path) -> None:
    time_s = motion["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, motion["translation_speed_mps"], linewidth=0.9)
    axes[0].set_ylabel("Translation [m/s]")
    axes[1].plot(time_s, motion["angular_speed_degps"], linewidth=0.9, color="tab:orange")
    axes[1].set_ylabel("Rotation [deg/s]")
    axes[2].plot(time_s, motion["motion_score"], linewidth=0.9, color="tab:green")
    axes[2].set_ylabel("Motion score")
    axes[2].set_xlabel("Time since sequence start [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(sequence)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(all_motion: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    keys = ("translation_speed_mps", "angular_speed_degps", "motion_score")
    labels = ("Translation [m/s]", "Rotation [deg/s]", "Motion score")
    for sequence, motion in all_motion.items():
        for axis, key in zip(axes, keys):
            axis.plot(motion["time_s"], motion[key], linewidth=0.8, label=sequence)
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time since sequence start [s]")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_duration <= 0 or args.smooth_seconds < 0:
        raise ValueError("window duration must be positive and smoothing must be non-negative")
    stride = args.window_stride or args.window_duration
    positive = {
        "window stride": stride,
        "translation scale": args.translation_scale,
        "rotation scale": args.rotation_scale,
        "base duration": args.base_duration,
        "minimum duration": args.min_duration,
        "maximum duration": args.max_duration,
        "score floor": args.score_floor,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.min_duration > args.max_duration:
        raise ValueError("--min-duration cannot exceed --max-duration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_motion: dict[str, dict[str, np.ndarray]] = {}
    all_samples: list[dict[str, object]] = []
    all_windows: list[dict[str, float | str]] = []

    for sequence in args.sequences:
        gt_path = args.data_root / sequence / "groundtruth.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing ground truth: {gt_path}")
        t, position, quaternion = load_groundtruth(gt_path)
        motion = estimate_motion(
            t,
            position,
            quaternion,
            args.smooth_seconds,
            args.translation_scale,
            args.rotation_scale,
        )
        all_motion[sequence] = motion
        sample_rows = [
            {"sequence": sequence, **{key: float(value[i]) for key, value in motion.items()}}
            for i in range(len(t))
        ]
        all_samples.extend(sample_rows)
        write_csv(args.output_dir / f"{sequence}_motion.csv", sample_rows)
        plot_sequence(sequence, motion, args.output_dir / f"{sequence}_motion.png")
        all_windows.extend(
            make_windows(
                sequence,
                motion,
                args.window_duration,
                stride,
                args.base_duration,
                args.min_duration,
                args.max_duration,
                args.score_floor,
            )
        )
        print(f"Processed {sequence}: {len(t)} poses, {t[-1] - t[0]:.2f} s")

    assign_speed_classes(all_windows)
    write_csv(args.output_dir / "all_sequences_motion.csv", all_samples)
    write_csv(args.output_dir / "slice_manifest.csv", all_windows)
    plot_summary(all_motion, args.output_dir / "all_sequences_motion.png")
    print(f"Wrote plots and CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
