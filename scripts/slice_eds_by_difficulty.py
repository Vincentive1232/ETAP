#!/usr/bin/env python3
"""Create slow/medium/fast EDS tracking-GT clips using image-plane speed."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ec_duration_experiment import load_tracks


DEFAULT_SEQUENCES = (
    "01_peanuts_light",
    "02_rocket_earth_light",
    "08_peanuts_running",
    "14_ziggy_in_the_arena",
)
DIFFICULTIES = ("slow", "medium", "fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split EDS DDFT tracks by pixel-speed difficulty.")
    parser.add_argument("--gt-tracks-root", type=Path, default=Path("config/misc/eds/gt_tracks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/eds_difficulty"))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--clip-duration", type=float, default=0.4)
    parser.add_argument("--selection-stride", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pixel_speed_series(
    times: np.ndarray, tracks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(times)
    valid = np.all(np.isfinite(tracks[:-1]), axis=2) & np.all(
        np.isfinite(tracks[1:]), axis=2
    )
    speed = np.linalg.norm(np.diff(tracks, axis=0), axis=2) / dt[:, None]
    speed[~valid] = np.nan
    return 0.5 * (times[:-1] + times[1:]), np.nanmedian(speed, axis=1)


def mean_window_speed(
    speed_times: np.ndarray,
    speeds: np.ndarray,
    starts: np.ndarray,
    duration: float,
) -> np.ndarray:
    result = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        selected = speeds[(speed_times >= start) & (speed_times < start + duration)]
        result[index] = float(np.nanmean(selected)) if len(selected) else np.nan
    return result


def select_intervals(
    sequence: str,
    times: np.ndarray,
    tracks: np.ndarray,
    duration: float,
    stride: float,
) -> list[dict[str, object]]:
    total = float(times[-1] - times[0])
    if total + 1e-9 < 3 * duration:
        raise ValueError(
            f"{sequence}: GT duration {total:.3f}s cannot contain three "
            f"non-overlapping {duration:.3f}s clips; maximum is {total / 3:.3f}s"
        )
    speed_times, speeds = pixel_speed_series(times, tracks)
    boundaries = np.linspace(float(times[0]), float(times[-1]), 4)
    chosen = []
    for region in range(3):
        first = boundaries[region]
        last = boundaries[region + 1] - duration
        starts = np.arange(first, last + 1e-12, stride)
        if len(starts) == 0 or starts[-1] < last - 1e-9:
            starts = np.append(starts, last)
        window_speeds = mean_window_speed(speed_times, speeds, starts, duration)
        target = float(np.nanquantile(speeds, (0.2, 0.5, 0.8)[region]))
        index = int(np.nanargmin(np.abs(window_speeds - target)))
        chosen.append(
            {
                "sequence": sequence,
                "start_timestamp": float(starts[index]),
                "end_timestamp": float(starts[index] + duration),
                "duration_s": float(duration),
                "mean_pixel_speed_pxps": float(window_speeds[index]),
            }
        )
    chosen.sort(key=lambda row: float(row["mean_pixel_speed_pxps"]))
    for difficulty, row in zip(DIFFICULTIES, chosen):
        row["difficulty"] = difficulty
        row["source_start_time_s"] = float(row["start_timestamp"]) - float(times[0])
        row["source_end_time_s"] = float(row["end_timestamp"]) - float(times[0])
    return chosen


def write_track_clip(source: Path, destination: Path, start: float, end: float) -> int:
    count = 0
    with source.open() as input_handle, destination.open("w") as output_handle:
        for line in input_handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                output_handle.write(line)
                continue
            fields = stripped.split()
            timestamp = float(fields[1])
            if start <= timestamp < end:
                output_handle.write(line)
                count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.clip_duration <= 0 or args.selection_stride <= 0:
        raise ValueError("clip duration and selection stride must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sequence in args.sequences:
        source = args.gt_tracks_root / f"{sequence}.gt.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        times, _, tracks = load_tracks(source)
        intervals = select_intervals(
            sequence, times, tracks, args.clip_duration, args.selection_stride
        )
        sequence_output = args.output_root / sequence
        if sequence_output.exists():
            if not args.overwrite:
                raise FileExistsError(f"{sequence_output} exists; use --overwrite")
            shutil.rmtree(sequence_output)
        for interval in intervals:
            output = sequence_output / str(interval["difficulty"])
            output.mkdir(parents=True)
            count = write_track_clip(
                source,
                output / "tracks.gt.txt",
                float(interval["start_timestamp"]),
                float(interval["end_timestamp"]),
            )
            metadata = {**interval, "track_rows": count, "source_tracks": str(source)}
            with (output / "clip.json").open("w") as handle:
                json.dump(metadata, handle, indent=2)
            interval["track_rows"] = count
            interval["output_dir"] = str(output)
            print(
                f"{sequence:24s} {interval['difficulty']:6s} "
                f"{interval['source_start_time_s']:.3f}-"
                f"{interval['source_end_time_s']:.3f}s "
                f"speed={interval['mean_pixel_speed_pxps']:.1f}px/s rows={count}"
            )
            manifest.append(interval)
    fields = list(manifest[0])
    with (args.output_root / "clips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Manifest: {args.output_root / 'clips.csv'}")


if __name__ == "__main__":
    main()
