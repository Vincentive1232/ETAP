#!/usr/bin/env python3
"""Create slow/medium/fast clips from each EC evaluation sequence.

Intervals are selected from ground-truth camera motion.  The source sequence is
split into three temporal regions, a representative clip is selected from each
region, and the three clips are ranked by their measured motion score.  Text
streams are read only once per source sequence and dispatched to all clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import TextIO

import numpy as np

from analyze_ec_motion import DEFAULT_SEQUENCES, estimate_motion, load_groundtruth


DIFFICULTIES = ("slow", "medium", "fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize three speed-ranked clips for every EC sequence."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ec"))
    parser.add_argument("--output-root", type=Path, default=Path("data/ec_difficulty"))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--clip-duration", type=float, default=5.0)
    parser.add_argument(
        "--selection-stride",
        type=float,
        default=0.10,
        help="Spacing between candidate clip starts, in seconds.",
    )
    parser.add_argument("--smooth-seconds", type=float, default=0.10)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=180.0)
    parser.add_argument(
        "--gt-tracks-root",
        type=Path,
        default=Path("config/misc/ec/gt_tracks"),
        help="DDFT track directory; pass a nonexistent path to skip tracks.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="How selected APS images are materialized (default: hardlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output sequence directory.",
    )
    return parser.parse_args()


def window_stat(
    times: np.ndarray, values: np.ndarray, starts: np.ndarray, duration: float
) -> np.ndarray:
    """Mean of irregularly sampled values in each [start, start+duration)."""
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, starts + duration, side="left")
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    counts = right - left
    if np.any(counts == 0):
        raise ValueError("candidate interval contains no ground-truth samples")
    return (prefix[right] - prefix[left]) / counts


def select_intervals(
    sequence: str,
    motion: dict[str, np.ndarray],
    duration: float,
    stride: float,
) -> list[dict[str, object]]:
    """Select one representative clip per temporal third, then rank by speed."""
    times = motion["timestamp"]
    t0, t1 = float(times[0]), float(times[-1])
    total = t1 - t0
    if total + 1e-9 < 3.0 * duration:
        raise ValueError(
            f"{sequence}: duration is {total:.3f}s, but three non-overlapping "
            f"{duration:.3f}s clips require at least {3.0 * duration:.3f}s"
        )

    boundaries = np.linspace(t0, t1, 4)
    targets = (0.2, 0.5, 0.8)
    selected: list[dict[str, object]] = []
    for region_index, target_quantile in enumerate(targets):
        region_start = boundaries[region_index]
        latest_start = boundaries[region_index + 1] - duration
        starts = np.arange(region_start, latest_start + 1e-12, stride)
        if len(starts) == 0 or starts[-1] < latest_start - 1e-9:
            starts = np.append(starts, latest_start)

        scores = window_stat(times, motion["motion_score"], starts, duration)
        trans = window_stat(times, motion["translation_speed_mps"], starts, duration)
        angular = window_stat(times, motion["angular_speed_degps"], starts, duration)
        target = float(np.quantile(motion["motion_score"], target_quantile))
        index = int(np.argmin(np.abs(scores - target)))
        selected.append(
            {
                "sequence": sequence,
                "start_timestamp": float(starts[index]),
                "end_timestamp": float(starts[index] + duration),
                "duration_s": float(duration),
                "mean_translation_speed_mps": float(trans[index]),
                "mean_angular_speed_degps": float(angular[index]),
                "mean_motion_score": float(scores[index]),
            }
        )

    # Difficulty is determined by measured speed, not merely by time order.
    selected.sort(key=lambda row: float(row["mean_motion_score"]))
    for difficulty, row in zip(DIFFICULTIES, selected):
        row["difficulty"] = difficulty
        row["source_start_time_s"] = float(row["start_timestamp"]) - t0
        row["source_end_time_s"] = float(row["end_timestamp"]) - t0
    return selected


def parse_timestamp(line: str, column: int) -> float | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split()
    try:
        return float(fields[column])
    except (IndexError, ValueError) as error:
        raise ValueError(f"cannot parse timestamp column {column} from: {stripped[:120]}") from error


def open_outputs(
    intervals: list[dict[str, object]], filename: str
) -> list[tuple[dict[str, object], TextIO]]:
    result = []
    for interval in intervals:
        path = Path(interval["output_dir"]) / filename
        result.append((interval, path.open("w")))
    return result


def slice_timestamp_file(
    source: Path,
    intervals: list[dict[str, object]],
    filename: str,
    timestamp_column: int = 0,
) -> dict[str, int]:
    if not source.is_file():
        return {}
    outputs = open_outputs(intervals, filename)
    counts = {str(interval["difficulty"]): 0 for interval in intervals}
    try:
        with source.open("r") as handle:
            for line in handle:
                timestamp = parse_timestamp(line, timestamp_column)
                if timestamp is None:
                    for _, output in outputs:
                        output.write(line)
                    continue
                for interval, output in outputs:
                    if float(interval["start_timestamp"]) <= timestamp < float(
                        interval["end_timestamp"]
                    ):
                        output.write(line)
                        counts[str(interval["difficulty"])] += 1
    finally:
        for _, output in outputs:
            output.close()
    return counts


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "none":
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def slice_images(
    source_dir: Path,
    intervals: list[dict[str, object]],
    image_mode: str,
) -> dict[str, int]:
    index_path = source_dir / "images.txt"
    if not index_path.is_file():
        return {}
    outputs = open_outputs(intervals, "images.txt")
    counts = {str(interval["difficulty"]): 0 for interval in intervals}
    try:
        with index_path.open("r") as handle:
            for line in handle:
                timestamp = parse_timestamp(line, 0)
                if timestamp is None:
                    for _, output in outputs:
                        output.write(line)
                    continue
                fields = line.strip().split(maxsplit=1)
                if len(fields) != 2:
                    raise ValueError(f"invalid image index line in {index_path}: {line.strip()}")
                relative_image = Path(fields[1])
                if relative_image.is_absolute() or ".." in relative_image.parts:
                    raise ValueError(f"unsafe image path in {index_path}: {relative_image}")
                for interval, output in outputs:
                    if float(interval["start_timestamp"]) <= timestamp < float(
                        interval["end_timestamp"]
                    ):
                        output.write(line)
                        source_image = source_dir / relative_image
                        destination = Path(interval["output_dir"]) / relative_image
                        if not source_image.is_file():
                            raise FileNotFoundError(source_image)
                        materialize_image(source_image, destination, image_mode)
                        counts[str(interval["difficulty"])] += 1
    finally:
        for _, output in outputs:
            output.close()
    return counts


def prepare_output_dirs(
    sequence_dir: Path,
    intervals: list[dict[str, object]],
    output_root: Path,
    overwrite: bool,
) -> None:
    sequence_output = output_root / sequence_dir.name
    if sequence_output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {sequence_output}; use --overwrite to replace it"
            )
        shutil.rmtree(sequence_output)
    for interval in intervals:
        output_dir = sequence_output / str(interval["difficulty"])
        output_dir.mkdir(parents=True)
        interval["output_dir"] = str(output_dir)
        calib = sequence_dir / "calib.txt"
        if calib.is_file():
            shutil.copy2(calib, output_dir / "calib.txt")


def materialize_sequence(
    sequence_dir: Path,
    intervals: list[dict[str, object]],
    gt_tracks_root: Path,
    image_mode: str,
) -> None:
    counts: dict[str, dict[str, int]] = {}
    for filename in ("events.txt", "groundtruth.txt", "imu.txt"):
        file_counts = slice_timestamp_file(sequence_dir / filename, intervals, filename)
        for difficulty, count in file_counts.items():
            counts.setdefault(difficulty, {})[filename] = count

    image_counts = slice_images(sequence_dir, intervals, image_mode)
    for difficulty, count in image_counts.items():
        counts.setdefault(difficulty, {})["images"] = count

    tracks_path = gt_tracks_root / f"{sequence_dir.name}.gt.txt"
    track_counts = slice_timestamp_file(tracks_path, intervals, "tracks.gt.txt", timestamp_column=1)
    for difficulty, count in track_counts.items():
        counts.setdefault(difficulty, {})["tracks.gt.txt"] = count

    for interval in intervals:
        difficulty = str(interval["difficulty"])
        metadata = {key: value for key, value in interval.items() if key != "output_dir"}
        metadata["counts"] = counts.get(difficulty, {})
        metadata["timestamps_rebased"] = False
        metadata["image_mode"] = image_mode
        output_dir = Path(interval["output_dir"])
        with (output_dir / "clip.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sequence",
        "difficulty",
        "start_timestamp",
        "end_timestamp",
        "source_start_time_s",
        "source_end_time_s",
        "duration_s",
        "mean_translation_speed_mps",
        "mean_angular_speed_degps",
        "mean_motion_score",
        "output_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    positive = {
        "clip duration": args.clip_duration,
        "selection stride": args.selection_stride,
        "translation scale": args.translation_scale,
        "rotation scale": args.rotation_scale,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.smooth_seconds < 0:
        raise ValueError("smoothing duration must be non-negative")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for sequence in args.sequences:
        sequence_dir = args.data_root / sequence
        gt_path = sequence_dir / "groundtruth.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        t, position, quaternion = load_groundtruth(gt_path)
        motion = estimate_motion(
            t,
            position,
            quaternion,
            args.smooth_seconds,
            args.translation_scale,
            args.rotation_scale,
        )
        intervals = select_intervals(
            sequence, motion, args.clip_duration, args.selection_stride
        )
        prepare_output_dirs(sequence_dir, intervals, args.output_root, args.overwrite)
        print(f"Slicing {sequence}; events.txt is scanned once...")
        materialize_sequence(sequence_dir, intervals, args.gt_tracks_root, args.image_mode)
        for interval in intervals:
            print(
                f"  {interval['difficulty']:>6}: "
                f"{interval['source_start_time_s']:.3f}-"
                f"{interval['source_end_time_s']:.3f}s, "
                f"score={interval['mean_motion_score']:.4f}"
            )
        manifest.extend(intervals)

    write_manifest(args.output_root / "clips.csv", manifest)
    print(f"Done. Clip manifest: {args.output_root / 'clips.csv'}")


if __name__ == "__main__":
    main()
