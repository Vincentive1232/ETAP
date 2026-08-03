#!/usr/bin/env python3
"""Plot per-point image-plane speeds from feature-track ground truth."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot every GT point's pixel speed, separately by sequence."
    )
    parser.add_argument(
        "--gt-root", type=Path, default=Path("config/misc/ec/gt_tracks")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gt_pixel_speeds")
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined multi-sequence figure.",
    )
    return parser.parse_args()


def sequence_name(path: Path) -> str:
    name = path.name
    if name.endswith(".gt.txt"):
        name = name[: -len(".gt.txt")]
    # gt_tracks_full names may include the selected frame range.
    return re.sub(r"_\d+_\d+$", "", name)


def discover_files(root: Path, requested: list[str] | None) -> list[Path]:
    paths = sorted(root.glob("*.gt.txt"))
    if requested is not None:
        wanted = set(requested)
        paths = [path for path in paths if sequence_name(path) in wanted]
        missing = wanted - {sequence_name(path) for path in paths}
        if missing:
            raise FileNotFoundError(f"GT files not found for: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"no *.gt.txt files found under {root}")
    return paths


def load_tracks(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[1] < 4:
        raise ValueError(f"{path}: expected id timestamp x y")
    rows = rows[:, :4]
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    return rows


def point_speed_curves(rows: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    curves = []
    for track_id in np.unique(rows[:, 0]).astype(int):
        track = rows[rows[:, 0] == track_id]
        _, unique_indices = np.unique(track[:, 1], return_index=True)
        track = track[np.sort(unique_indices)]
        if len(track) < 2:
            continue
        dt = np.diff(track[:, 1])
        valid = dt > 0
        midpoint = 0.5 * (track[:-1, 1] + track[1:, 1])
        displacement = np.linalg.norm(np.diff(track[:, 2:4], axis=0), axis=1)
        curves.append((track_id, midpoint[valid], displacement[valid] / dt[valid]))
    return curves


def aggregate_speed(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.unique(rows[:, 1])
    track_ids = np.unique(rows[:, 0]).astype(int)
    id_to_column = {track_id: column for column, track_id in enumerate(track_ids)}
    time_to_row = {timestamp: index for index, timestamp in enumerate(timestamps)}
    coordinates = np.full((len(timestamps), len(track_ids), 2), np.nan)
    for track_id, timestamp, x, y in rows:
        coordinates[time_to_row[timestamp], id_to_column[int(track_id)]] = (x, y)
    dt = np.diff(timestamps)
    valid = (
        np.all(np.isfinite(coordinates[:-1]), axis=2)
        & np.all(np.isfinite(coordinates[1:]), axis=2)
        & (dt[:, None] > 0)
    )
    speed = np.linalg.norm(np.diff(coordinates, axis=0), axis=2) / dt[:, None]
    speed[~valid] = np.nan
    midpoint = 0.5 * (timestamps[:-1] + timestamps[1:])
    return (
        midpoint,
        np.nanmedian(speed, axis=1),
        np.nanquantile(speed, 0.25, axis=1),
        np.nanquantile(speed, 0.75, axis=1),
        np.sum(np.isfinite(speed), axis=1),
    )


def plot_on_axis(
    axis: plt.Axes,
    name: str,
    rows: np.ndarray,
    log_y: bool,
    show_legend: bool = True,
) -> dict[str, float | str | int]:
    origin = float(np.min(rows[:, 1]))
    curves = point_speed_curves(rows)
    all_speeds = []
    for _, timestamps, speeds in curves:
        axis.plot(
            timestamps - origin,
            speeds,
            color="tab:blue",
            linewidth=0.65,
            alpha=0.22,
        )
        all_speeds.append(speeds)
    midpoint, median, lower, upper, counts = aggregate_speed(rows)
    axis.fill_between(
        midpoint - origin,
        lower,
        upper,
        color="tab:orange",
        alpha=0.22,
        label="point-speed IQR",
    )
    axis.plot(
        midpoint - origin,
        median,
        color="darkorange",
        linewidth=2.2,
        label="median point speed",
    )
    axis.set_title(name)
    axis.set_xlabel("Time since GT start [s]")
    axis.set_ylabel("Pixel speed [px/s]")
    axis.grid(alpha=0.25)
    if log_y:
        axis.set_yscale("symlog", linthresh=1.0)
    if show_legend:
        handles = [
            Line2D([0], [0], color="tab:blue", alpha=0.35, linewidth=1, label="individual GT points"),
            Line2D([0], [0], color="darkorange", linewidth=2.2, label="median point speed"),
            plt.Rectangle((0, 0), 1, 1, color="tab:orange", alpha=0.22, label="25–75% points"),
        ]
        axis.legend(handles=handles, fontsize=8)

    flattened = np.concatenate(all_speeds)
    return {
        "sequence": name,
        "duration_s": float(np.max(rows[:, 1]) - origin),
        "num_tracks": int(len(curves)),
        "num_timestamps": int(len(np.unique(rows[:, 1]))),
        "speed_min_pxps": float(np.min(flattened)),
        "speed_q10_pxps": float(np.quantile(flattened, 0.10)),
        "speed_q25_pxps": float(np.quantile(flattened, 0.25)),
        "speed_median_pxps": float(np.median(flattened)),
        "speed_q75_pxps": float(np.quantile(flattened, 0.75)),
        "speed_q90_pxps": float(np.quantile(flattened, 0.90)),
        "speed_max_pxps": float(np.max(flattened)),
        "median_active_tracks": float(np.median(counts)),
    }


def main() -> None:
    args = parse_args()
    paths = discover_files(args.gt_root, args.sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(sequence_name(path), load_tracks(path)) for path in paths]
    summaries = []

    columns = 2
    rows_count = int(np.ceil(len(loaded) / columns))
    figure, axes_grid = plt.subplots(
        rows_count, columns, figsize=(15, 4.6 * rows_count), squeeze=False
    )
    axes = axes_grid.ravel()
    for axis, (name, rows) in zip(axes, loaded):
        summaries.append(plot_on_axis(axis, name, rows, args.log_y))
    for axis in axes[len(loaded) :]:
        axis.set_visible(False)
    figure.suptitle("GT feature-point image-plane speeds", fontsize=16)
    figure.tight_layout()
    figure.savefig(args.output_dir / "all_sequences_pixel_speed.png", dpi=180)
    plt.close(figure)

    if not args.no_individual:
        for name, rows in loaded:
            figure, axis = plt.subplots(figsize=(12, 5.5))
            plot_on_axis(axis, name, rows, args.log_y)
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{name}_pixel_speed.png", dpi=180)
            plt.close(figure)

    with (args.output_dir / "pixel_speed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(loaded)} sequence plots to {args.output_dir}")


if __name__ == "__main__":
    main()
