from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
import yaml

'''
Trajectory Segmentation
Purpose: break long vessel trajectories into smaller analysis windows.
Methods used:
---Fixed-time window segmentation
---Per-vessel segmentation
------30-minute windows
------1-hour windows
------Segment ID assignment
---Segment filtering by minimum AIS points

Filtering rule:
---remove segment if number_of_points < 10

Output:
segments.parquet

Why: ML models need fixed-size behavioral units instead of one huge full-day trajectory.

Next:
eez_processor.py
geofence.py
'''


REQUIRED_COLUMNS = ["MMSI", "timestamp", "latitude", "longitude", "speed", "heading"]
DEFAULT_WINDOW_MINUTES = [30, 60]
DEFAULT_MIN_POINTS = 10


def load_cleaned_ais(cleaned_ais_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(cleaned_ais_path, engine="pyarrow")

'''
Validates and sorts the input AIS data by time to prepare for grouping.
Iterates through specified time windows and individual vessels to create subsets.
Concatenates all resulting segments into a final, sorted DataFrame for analysis.
'''
def segment_trajectories(
    cleaned_ais: pd.DataFrame,
    window_minutes: Iterable[int] = DEFAULT_WINDOW_MINUTES,
    min_points: int = DEFAULT_MIN_POINTS,
) -> pd.DataFrame:
    """Create fixed-duration trajectory windows per vessel."""
    _validate_cleaned_ais(cleaned_ais)
    cleaned = cleaned_ais.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)
    cleaned = cleaned.dropna(subset=["MMSI", "timestamp"])
    cleaned = cleaned.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

    segmented_frames: list[pd.DataFrame] = []
    for window in window_minutes:
        for _, vessel_frame in cleaned.groupby("MMSI", sort=True):
            segmented_frames.extend(_segment_vessel(vessel_frame, int(window), min_points))

    if not segmented_frames:
        return _empty_segments_frame(cleaned)

    segments = pd.concat(segmented_frames, ignore_index=True)
    return segments.sort_values(
        ["window_minutes", "MMSI", "segment_start", "timestamp"]
    ).reset_index(drop=True)


'''
Ensures the output directory exists to prevent file system errors.
Converts the processed segments DataFrame into a structured Parquet file.
Uses the PyArrow engine to maintain high-performance data serialization.
'''
def write_segments(segments: pd.DataFrame, output_path: str | Path) -> None:
    parquet_path = Path(output_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(parquet_path, engine="pyarrow", index=False)

'''
Orchestrates the workflow by loading the cleaned AIS data from a path.
Triggers the segmentation logic with the provided window and threshold parameters.
Writes the final segments to disk and returns the result for immediate use.
'''
def build_segments(
    cleaned_ais_path: str | Path,
    segments_output_path: str | Path,
    window_minutes: Iterable[int] = DEFAULT_WINDOW_MINUTES,
    min_points: int = DEFAULT_MIN_POINTS,
) -> pd.DataFrame:
    cleaned_ais = load_cleaned_ais(cleaned_ais_path)
    segments = segment_trajectories(
        cleaned_ais,
        window_minutes=window_minutes,
        min_points=min_points,
    )
    write_segments(segments, segments_output_path)
    return segments

'''
Loads configuration settings from a YAML file to drive the pipeline.
Extracts segmentation parameters like window size and minimum point thresholds.
Initiates the build_segments process using the parsed configuration.
'''
def main(config_path: str | Path = "config.yaml") -> None:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    segmentation_config = config.get("segmentation", {})
    build_segments(
        cleaned_ais_path=config["output"]["cleaned_ais_path"],
        segments_output_path=config["output"].get(
            "segments_path", "data/processed/segments.parquet"
        ),
        window_minutes=segmentation_config.get("window_minutes", DEFAULT_WINDOW_MINUTES),
        min_points=segmentation_config.get("min_points", DEFAULT_MIN_POINTS),
    )


def _segment_vessel(
    vessel_frame: pd.DataFrame,
    window_minutes: int,
    min_points: int,
) -> list[pd.DataFrame]:
    if window_minutes <= 0:
        raise ValueError("Window size must be positive.")
    if min_points <= 0:
        raise ValueError("Minimum point threshold must be positive.")

    vessel = vessel_frame.sort_values("timestamp").reset_index(drop=True).copy()
    trajectory_start = vessel["timestamp"].min()
    window_delta = pd.Timedelta(minutes=window_minutes)
    vessel["_window_index"] = ((vessel["timestamp"] - trajectory_start) // window_delta).astype(
        "int64"
    )

    segment_frames: list[pd.DataFrame] = []
    for window_index, segment in vessel.groupby("_window_index", sort=True):
        number_of_points = len(segment)
        if number_of_points < min_points:
            continue

        segment = segment.drop(columns=["_window_index"]).copy()
        segment_start = segment["timestamp"].min()
        segment_end = segment["timestamp"].max()
        duration = segment_end - segment_start
        mmsi = int(segment["MMSI"].iloc[0])
        segment_id = _make_segment_id(mmsi, window_minutes, int(window_index), segment_start)

        segment["segment_id"] = segment_id
        segment["window_minutes"] = window_minutes
        segment["segment_start"] = segment_start
        segment["segment_end"] = segment_end
        segment["duration"] = duration
        segment["duration_seconds"] = duration.total_seconds()
        segment["number_of_points"] = number_of_points
        segment_frames.append(segment)

    return segment_frames


def _make_segment_id(
    mmsi: int,
    window_minutes: int,
    window_index: int,
    segment_start: pd.Timestamp,
) -> str:
    timestamp_label = segment_start.strftime("%Y%m%dT%H%M%SZ")
    return f"{mmsi}_{window_minutes}min_{window_index:06d}_{timestamp_label}"


def _validate_cleaned_ais(dataframe: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required cleaned AIS columns: {missing}")


def _empty_segments_frame(cleaned_ais: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *cleaned_ais.columns.tolist(),
        "segment_id",
        "window_minutes",
        "segment_start",
        "segment_end",
        "duration",
        "duration_seconds",
        "number_of_points",
    ]
    return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    main()
