#!/usr/bin/env python3
"""Plot EC duration/accuracy relationships separately for every sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIFFICULTIES = ("slow", "medium", "fast")
COLORS = {"slow": "tab:blue", "medium": "tab:orange", "fast": "tab:red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one speed-duration analysis figure per EC sequence."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("output/ec_duration_experiment_1s")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--speed-bins", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path, text_fields: set[str]) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in set(row) - text_fields:
            row[field] = float(row[field])
    return rows


def weighted_accuracy(rows: list[dict[str, object]]) -> float:
    return float(
        np.average(
            [float(row["accuracy_5px"]) for row in rows],
            weights=[float(row["num_valid_tracks"]) for row in rows],
        )
    )


def physical_speed_rows(
    frame_rows: list[dict[str, object]], sequence: str
) -> list[dict[str, object]]:
    """Use one duration only; physical GT speed repeats for every duration."""
    selected = [row for row in frame_rows if row["sequence"] == sequence]
    first_duration = min(float(row["duration_ms"]) for row in selected)
    return [row for row in selected if float(row["duration_ms"]) == first_duration]


def difficulty_speed_stats(
    frame_rows: list[dict[str, object]], sequence: str
) -> dict[str, tuple[float, float, float]]:
    rows = physical_speed_rows(frame_rows, sequence)
    result = {}
    for difficulty in DIFFICULTIES:
        speeds = np.asarray(
            [
                float(row["pixel_speed_pxps"])
                for row in rows
                if row["clip_difficulty"] == difficulty
            ]
        )
        if len(speeds):
            result[difficulty] = (
                float(np.median(speeds)),
                float(np.min(speeds)),
                float(np.max(speeds)),
            )
    return result


def speed_duration_matrix(
    frame_rows: list[dict[str, object]],
    sequence: str,
    durations: list[float],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical = physical_speed_rows(frame_rows, sequence)
    physical_speeds = np.asarray([float(row["pixel_speed_pxps"]) for row in physical])
    edges = np.unique(np.quantile(physical_speeds, np.linspace(0, 1, num_bins + 1)))
    if len(edges) < 2:
        edges = np.array([physical_speeds[0] - 0.5, physical_speeds[0] + 0.5])
    centers = np.empty(len(edges) - 1)
    matrix = np.full((len(edges) - 1, len(durations)), np.nan)
    all_sequence_rows = [row for row in frame_rows if row["sequence"] == sequence]
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin_physical = physical_speeds[
            (physical_speeds >= left)
            & (physical_speeds <= right if bin_index == len(edges) - 2 else physical_speeds < right)
        ]
        centers[bin_index] = float(np.median(in_bin_physical))
        for duration_index, duration in enumerate(durations):
            selected = [
                row
                for row in all_sequence_rows
                if float(row["duration_ms"]) == duration
                and float(row["pixel_speed_pxps"]) >= left
                and (
                    float(row["pixel_speed_pxps"]) <= right
                    if bin_index == len(edges) - 2
                    else float(row["pixel_speed_pxps"]) < right
                )
            ]
            if selected:
                matrix[bin_index, duration_index] = weighted_accuracy(selected)
    return centers, edges, matrix


def plot_sequence(
    sequence: str,
    results: list[dict[str, object]],
    frame_rows: list[dict[str, object]],
    output_dir: Path,
    speed_bins: int,
) -> list[dict[str, object]]:
    sequence_results = [row for row in results if row["sequence"] == sequence]
    durations = sorted({float(row["duration_ms"]) for row in sequence_results})
    speed_stats = difficulty_speed_stats(frame_rows, sequence)
    ranked_sources = sorted(
        speed_stats, key=lambda difficulty: speed_stats[difficulty][0]
    )
    measured_rank = {
        source: rank for source, rank in zip(ranked_sources, DIFFICULTIES)
    }
    centers, edges, matrix = speed_duration_matrix(
        frame_rows, sequence, durations, speed_bins
    )
    best_indices = np.nanargmax(matrix, axis=1)
    best_durations = np.asarray(durations)[best_indices]
    correlation = (
        float(np.corrcoef(centers, best_durations)[0, 1])
        if len(centers) > 1 and np.std(best_durations) > 0
        else math.nan
    )
    if len(centers) > 1:
        slope, intercept = np.polyfit(centers, best_durations, 1)
    else:
        slope, intercept = math.nan, float(best_durations[0])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    accuracy_axis, error_axis, heatmap_axis, optimal_axis = axes.ravel()

    for difficulty in DIFFICULTIES:
        group = sorted(
            (row for row in sequence_results if row["difficulty"] == difficulty),
            key=lambda row: float(row["duration_ms"]),
        )
        if not group:
            continue
        median_speed, minimum_speed, maximum_speed = speed_stats[difficulty]
        pixel_difficulty = measured_rank[difficulty]
        label = (
            f"pixel-{pixel_difficulty} (source clip: {difficulty}): "
            f"median {median_speed:.1f} px/s [{minimum_speed:.1f}, {maximum_speed:.1f}]"
        )
        group_durations = [float(row["duration_ms"]) for row in group]
        accuracy_axis.plot(
            group_durations,
            [float(row["accuracy_5px"]) for row in group],
            marker="o",
            ms=3,
            color=COLORS[pixel_difficulty],
            label=label,
        )
        error_axis.plot(
            group_durations,
            [float(row["mean_endpoint_error_px"]) for row in group],
            marker="o",
            ms=3,
            color=COLORS[pixel_difficulty],
            label=f"pixel-{pixel_difficulty} (source: {difficulty})",
        )

    accuracy_axis.set_title("Clip difficulty: accuracy vs duration")
    accuracy_axis.set_ylabel("Tracking accuracy @ 5 px")
    accuracy_axis.set_xlabel("Duration [ms]")
    accuracy_axis.legend(fontsize=8)
    error_axis.set_title("Clip difficulty: endpoint error vs duration")
    error_axis.set_ylabel("Mean endpoint error [px]")
    error_axis.set_xlabel("Duration [ms]")
    error_axis.legend()

    image = heatmap_axis.imshow(
        matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="magma"
    )
    tick_step = max(1, len(durations) // 12)
    tick_indices = list(range(0, len(durations), tick_step))
    if tick_indices[-1] != len(durations) - 1:
        tick_indices.append(len(durations) - 1)
    heatmap_axis.set_xticks(
        tick_indices, [f"{durations[index]:g}" for index in tick_indices], rotation=45
    )
    heatmap_axis.set_yticks(
        range(len(centers)), [f"{center:.1f}" for center in centers]
    )
    heatmap_axis.set_xlabel("Duration [ms]")
    heatmap_axis.set_ylabel("Within-sequence pixel speed bin [px/s]")
    heatmap_axis.set_title("Per-frame Accuracy@5px")
    fig.colorbar(image, ax=heatmap_axis, label="Accuracy @ 5 px")

    optimal_axis.scatter(centers, best_durations, s=48, color="tab:purple")
    if len(centers) > 1:
        line_x = np.linspace(centers.min(), centers.max(), 100)
        optimal_axis.plot(
            line_x,
            intercept + slope * line_x,
            "k--",
            label=f"slope={slope:.3f} ms/(px/s), r={correlation:.3f}",
        )
    for speed, duration in zip(centers, best_durations):
        optimal_axis.annotate(f"{duration:g}ms", (speed, duration), xytext=(4, 4), textcoords="offset points", fontsize=8)
    optimal_axis.set_title("Best duration vs pixel speed")
    optimal_axis.set_xlabel("Within-sequence median pixel speed [px/s]")
    optimal_axis.set_ylabel("Best duration [ms]")
    optimal_axis.legend(fontsize=8)

    for axis in (accuracy_axis, error_axis, optimal_axis):
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"{sequence}: speed range and duration sensitivity",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{sequence}_speed_duration.png", dpi=180)
    plt.close(fig)

    summary = []
    for bin_index, (speed, best_duration) in enumerate(zip(centers, best_durations)):
        summary.append(
            {
                "sequence": sequence,
                "speed_bin": bin_index,
                "speed_left_pxps": float(edges[bin_index]),
                "speed_right_pxps": float(edges[bin_index + 1]),
                "median_speed_pxps": float(speed),
                "best_duration_ms": float(best_duration),
                "best_accuracy_5px": float(matrix[bin_index, best_indices[bin_index]]),
                "duration_speed_pearson_r": correlation,
                "duration_speed_slope_ms_per_pxps": float(slope),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.speed_bins < 2:
        raise ValueError("--speed-bins must be at least 2")
    results_path = args.results_dir / "results.csv"
    frame_path = args.results_dir / "frame_results.csv"
    if not results_path.is_file() or not frame_path.is_file():
        raise FileNotFoundError(
            f"expected both {results_path} and {frame_path}"
        )
    output_dir = args.output_dir or args.results_dir / "per_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = read_csv(
        results_path, {"sequence", "difficulty", "prediction_path"}
    )
    frame_rows = read_csv(
        frame_path, {"sequence", "clip_difficulty", "pixel_speed_class"}
    )
    sequences = sorted({str(row["sequence"]) for row in results})
    all_summary = []
    for sequence in sequences:
        print(f"Plotting {sequence}...")
        all_summary.extend(
            plot_sequence(
                sequence, results, frame_rows, output_dir, args.speed_bins
            )
        )
    summary_path = output_dir / "per_sequence_speed_duration.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0]))
        writer.writeheader()
        writer.writerows(all_summary)
    with (output_dir / "per_sequence_relationship.json").open("w") as handle:
        json.dump(
            {
                sequence: {
                    "pearson_r": next(
                        row["duration_speed_pearson_r"]
                        for row in all_summary
                        if row["sequence"] == sequence
                    ),
                    "slope_ms_per_pxps": next(
                        row["duration_speed_slope_ms_per_pxps"]
                        for row in all_summary
                        if row["sequence"] == sequence
                    ),
                }
                for sequence in sequences
            },
            handle,
            indent=2,
        )
    print(f"Wrote {len(sequences)} figures to {output_dir}")


if __name__ == "__main__":
    main()
