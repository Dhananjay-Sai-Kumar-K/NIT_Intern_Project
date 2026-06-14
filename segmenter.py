from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def build_segments_chunked(
    cleaned_ais_path: str | Path, 
    segments_output_path: str | Path, 
    window_minutes: Iterable[int] = DEFAULT_WINDOW_MINUTES, 
    min_points: int = DEFAULT_MIN_POINTS,
    mmsi_batch_size: int = 500
) -> None:
    """Processes segments safely by loading chunks of vessels instead of the whole file."""
    cleaned_ais_path = Path(cleaned_ais_path)
    segments_output_path = Path(segments_output_path)
    
    print("Reading unique MMSIs for chunking...")
    # Read ONLY the MMSI column to save memory
    dataset = pq.ParquetDataset(cleaned_ais_path)
    all_mmsis = dataset.read(columns=['MMSI']).to_pandas()['MMSI'].unique()
    
    print(f"Found {len(all_mmsis)} unique vessels. Processing in batches of {mmsi_batch_size}...")
    
    # Clear existing file if we are replacing
    if segments_output_path.exists():
        segments_output_path.unlink()
        
    segments_output_path.parent.mkdir(parents=True, exist_ok=True)

    # FIX: Keep a single ParquetWriter open for the entire loop so each chunk
    # is appended rather than overwriting the file from scratch.
    writer: pq.ParquetWriter | None = None
    total_chunks = (len(all_mmsis) + mmsi_batch_size - 1) // mmsi_batch_size
    try:
        for i in range(0, len(all_mmsis), mmsi_batch_size):
            mmsi_batch = all_mmsis[i : i + mmsi_batch_size].tolist()
            
            # Load full data ONLY for this specific batch of vessels
            table = pq.read_table(
                cleaned_ais_path, 
                filters=[('MMSI', 'in', mmsi_batch)]
            )
            batch_df = table.to_pandas()
            
            # Run your existing segmenting logic on this small batch
            batch_segments = segment_trajectories(batch_df, window_minutes, min_points)
            
            if batch_segments.empty:
                continue
                
            batch_table = pa.Table.from_pandas(batch_segments, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(segments_output_path, batch_table.schema)
            writer.write_table(batch_table)
                    
            current_chunk = (i // mmsi_batch_size) + 1
            print(f"Segmented chunk {current_chunk} / {total_chunks}")
    finally:
        if writer is not None:
            writer.close()
        elif not segments_output_path.exists():
            # No segments were written at all — write an empty placeholder
            _empty_segments_frame(
                pd.DataFrame(columns=REQUIRED_COLUMNS)
            ).to_parquet(segments_output_path, engine="pyarrow", index=False)


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
    if "trajectory_date" not in cleaned:
        cleaned["trajectory_date"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d")
    cleaned["trajectory_date"] = cleaned["trajectory_date"].astype(str)
    cleaned = cleaned.sort_values(["MMSI", "trajectory_date", "timestamp"]).reset_index(drop=True)

    segmented_frames: list[pd.DataFrame] = []
    for window in window_minutes:
        for _, vessel_frame in cleaned.groupby(["MMSI", "trajectory_date"], sort=True):
            segmented_frames.extend(_segment_vessel(vessel_frame, int(window), min_points))

    if not segmented_frames:
        return _empty_segments_frame(cleaned)

    segments = pd.concat(segmented_frames, ignore_index=True)
    return segments.sort_values(
        ["window_minutes", "MMSI", "trajectory_date", "segment_start", "timestamp"]
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


def build_segments_by_date(
    cleaned_ais_path: str | Path,
    segments_output_path: str | Path,
    window_minutes: Iterable[int] = DEFAULT_WINDOW_MINUTES,
    min_points: int = DEFAULT_MIN_POINTS,
) -> int:
    """Segment a multi-day cleaned AIS Parquet one trajectory_date at a time."""
    cleaned_path = Path(cleaned_ais_path)
    output_path = Path(segments_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    dates = _read_trajectory_dates(cleaned_path)
    writer: pq.ParquetWriter | None = None
    total_segments = 0
    try:
        for trajectory_date in dates:
            daily_points = pd.read_parquet(
                cleaned_path,
                engine="pyarrow",
                filters=[("trajectory_date", "=", trajectory_date)],
            )
            daily_segments = segment_trajectories(
                daily_points,
                window_minutes=window_minutes,
                min_points=min_points,
            )
            if daily_segments.empty:
                continue

            total_segments += int(daily_segments["segment_id"].nunique())
            table = pa.Table.from_pandas(daily_segments, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        _empty_segments_frame(pd.DataFrame(columns=REQUIRED_COLUMNS)).to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
        )
    return total_segments


'''
Loads configuration settings from a YAML file to drive the pipeline.
Extracts segmentation parameters like window size and minimum point thresholds.
Initiates the build_segments process using the parsed configuration.
'''
def main(config_path: str | Path = "config.yaml") -> None:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    segmentation_config = config.get("segmentation", {})
    
    # Using the memory-safe chunked builder for scalable execution
    build_segments_chunked(
        cleaned_ais_path=config["output"]["cleaned_ais_path"],
        segments_output_path=config["output"].get(
            "segments_path", "data/processed/segments.parquet"
        ),
        window_minutes=segmentation_config.get("window_minutes", DEFAULT_WINDOW_MINUTES),
        min_points=segmentation_config.get("min_points", DEFAULT_MIN_POINTS),
        mmsi_batch_size=500
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
        trajectory_date = str(segment["trajectory_date"].iloc[0])
        segment_id = _make_segment_id(
            mmsi,
            window_minutes,
            int(window_index),
            segment_start,
            trajectory_date,
        )

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
    trajectory_date: str | None = None,
) -> str:
    timestamp_label = segment_start.strftime("%Y%m%dT%H%M%SZ")
    date_label = (trajectory_date or segment_start.strftime("%Y-%m-%d")).replace("-", "")
    return f"{mmsi}_{date_label}_{window_minutes}min_{window_index:06d}_{timestamp_label}"


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


def _read_trajectory_dates(cleaned_ais_path: Path) -> list[str]:
    parquet_file = pq.ParquetFile(cleaned_ais_path)
    if "trajectory_date" not in parquet_file.schema_arrow.names:
        dataframe = pd.read_parquet(cleaned_ais_path, engine="pyarrow", columns=["timestamp"])
        dates = pd.to_datetime(dataframe["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
        return sorted(dates.dropna().unique().tolist())

    dates: set[str] = set()
    for batch in parquet_file.iter_batches(columns=["trajectory_date"], batch_size=1_000_000):
        series = batch.to_pandas()["trajectory_date"].dropna().astype(str)
        dates.update(series.unique().tolist())
    return sorted(dates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment cleaned AIS trajectories into fixed-duration windows."
    )
    parser.add_argument(
        "--ais-points",
        type=Path,
        default=None,
        help="Path to the cleaned AIS parquet file (e.g. data/processed/aisdk-2024-03-01_cleaned_ais.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the segments parquet file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=None,
        help="Segmentation window size(s) in minutes (e.g. --window 30 60). Defaults to [30, 60].",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="Minimum number of AIS points per segment (default: 10).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml. Used as fallback for any unspecified arguments.",
    )
    return parser.parse_args()


def cli() -> None:
    """Command-line entry point that accepts explicit paths and falls back to config.yaml."""
    import argparse as _argparse
    args = parse_args()

    # Load config for fallback values
    config: dict = {}
    config_path = args.config or Path("config.yaml")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

    seg_config = config.get("segmentation", {})

    cleaned_ais_path = args.ais_points or Path(
        config.get("output", {}).get("cleaned_ais_path", "data/processed/cleaned_ais.parquet")
    )
    segments_output_path = args.output or Path(
        config.get("output", {}).get("segments_path", "data/processed/segments.parquet")
    )
    window_minutes = args.window or seg_config.get("window_minutes", DEFAULT_WINDOW_MINUTES)
    min_points = args.min_points if args.min_points is not None else seg_config.get("min_points", DEFAULT_MIN_POINTS)

    print(f"AIS input:  {cleaned_ais_path}")
    print(f"Output:     {segments_output_path}")
    print(f"Windows:    {window_minutes} min")
    print(f"Min points: {min_points}")

    build_segments_chunked(
        cleaned_ais_path=cleaned_ais_path,
        segments_output_path=segments_output_path,
        window_minutes=window_minutes,
        min_points=min_points,
        mmsi_batch_size=500,
    )
    print("Segmentation complete.")


if __name__ == "__main__":
    cli()
