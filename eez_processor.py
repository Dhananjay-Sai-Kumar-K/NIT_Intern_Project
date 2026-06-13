from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import yaml
from geofence import (
    detect_boundary_events_with_state,
    ais_points_to_geodataframe,
    classify_points_by_eez,
    load_eez_polygons,
)

"""
1. Inputs
Cleaned AIS Data:, EEZ Shapefile, Configuration

2. The Process (Geofencing Pipeline)
The script uses a batch processing approach to handle potentially massive amounts of AIS data without overloading your computer's memory.

Batching: It reads the AIS data in chunks (defined by batch_size, defaulting to 250,000 rows).
Spatial Conversion: Each chunk of latitude/longitude points is converted into a GeoDataFrame 
(a table with geometric spatial data).
EEZ Classification: The script performs a "spatial join" between the vessel points and the EEZ polygons 
to determine if each point is "inside" or "outside" an EEZ.

State Detection: This is the most critical part. It keeps track of the vessel's "previous state" 
(previous_states dictionary).

If a point is outside and the next point is inside, it flags an "Entry" event.
If a point is inside and the next point is outside, it flags an "Exit" event.

State Persistence: It carries the previous_states dictionary forward to the next batch, ensuring that 
if a vessel crosses a boundary right at the edge of a batch, the transition is still detected correctly.

Outputs
Boundary Events Parquet: A new file (boundary_events.parquet) containing only the moments a vessel crossed a line.

Structure: The final table typically contains:
mmsi: The vessel's unique ID.
event_timestamp: The exact time the crossing occurred.
event_type: A label (e.g., 'entry' or 'exit').

Next:
event_extractor.py
"""


DEFAULT_CLEANED_PATH = Path("data/processed/cleaned_ais.parquet")
DEFAULT_EEZ_PATH = Path("data/raw/eez/eez.shp")
DEFAULT_BOUNDARY_EVENTS_PATH = Path("data/processed/boundary_events.parquet")
DEFAULT_BATCH_SIZE = 250_000


def build_boundary_events(
    ais_points_path: str | Path,
    eez_path: str | Path,
    output_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    eez_polygons = load_eez_polygons(eez_path)
    event_frames: list[pd.DataFrame] = []
    previous_states: dict[int, bool] = {}

    for ais_points in _read_ais_batches(Path(ais_points_path), batch_size=batch_size):
        point_geometries = ais_points_to_geodataframe(ais_points)
        classified_points = classify_points_by_eez(point_geometries, eez_polygons)
        events, previous_states = detect_boundary_events_with_state(
            classified_points,
            previous_states=previous_states,
        )
        if not events.empty:
            event_frames.append(events)

    if event_frames:
        boundary_events = pd.concat(event_frames, ignore_index=True)
    else:
        boundary_events = pd.DataFrame(
            {
                "event_type": pd.Series(dtype="object"),
                "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "mmsi": pd.Series(dtype="int64"),
            }
        )

    boundary_events = boundary_events.sort_values(["mmsi", "event_timestamp"]).reset_index(
        drop=True
    )
    write_boundary_events(boundary_events, output_path)
    return boundary_events


def write_boundary_events(boundary_events: pd.DataFrame, output_path: str | Path) -> None:
    parquet_path = Path(output_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_events.to_parquet(parquet_path, engine="pyarrow", index=False)


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    geofence_config = config.get("geofencing", {})
    build_boundary_events(
        ais_points_path=config["output"].get("cleaned_ais_path", DEFAULT_CLEANED_PATH),
        eez_path=config["input"].get("eez_shapefile_path", DEFAULT_EEZ_PATH),
        output_path=config["output"].get(
            "boundary_events_path", DEFAULT_BOUNDARY_EVENTS_PATH
        ),
        batch_size=geofence_config.get("batch_size", DEFAULT_BATCH_SIZE),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect AIS EEZ entry and exit events from cleaned AIS points."
    )
    parser.add_argument(
        "--ais-points",
        type=Path,
        default=DEFAULT_CLEANED_PATH,
        help="Path to cleaned AIS Parquet or CSV.",
    )
    parser.add_argument(
        "--eez",
        type=Path,
        default=DEFAULT_EEZ_PATH,
        help="Path to Marine Regions EEZ shapefile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BOUNDARY_EVENTS_PATH,
        help="Output path for boundary events Parquet.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Rows per batch when reading AIS Parquet.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    events = build_boundary_events(
        ais_points_path=args.ais_points,
        eez_path=args.eez,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(f"Boundary events: {len(events)}")
    print(f"Wrote: {args.output}")


def _read_ais_batches(path: Path, batch_size: int):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    if not path.exists():
        raise FileNotFoundError(f"AIS points file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
    elif suffix == ".csv":
        yield from pd.read_csv(path, chunksize=batch_size)
    else:
        raise ValueError(f"Unsupported AIS point file type: {path.suffix}")


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()
