from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from geofence import load_eez_polygons

'''
Feature Engineering
Purpose: convert segments and events into ML-ready numerical features.

Now that you have segmented your trajectories and extracted meaningful behavioral events 
(like "Anchoring" or "Maneuvering"), the Feature Engineering script acts as the final bridge 
between raw maritime data and a Machine Learning model.

At this stage, you are no longer just looking at points on a map; you are calculating 
quantitative features that a computer can use to recognize patterns (e.g., distinguishing a 
fishing vessel from a cargo ship, or identifying abnormal behavior).

1. Inputs
This script consolidates all your previous work into a single view:

Segments: The segments.parquet (the "time-windows" of movement).
Boundary Events: The boundary_events.parquet (when they crossed borders).
Event Sequence: The event_sequence.json (the high-level behaviors like "LOITER" or "ANCHOR").
EEZ Shapefile: The raw map data, used to calculate how close a ship is to a specific boundary.

2. The Process: Feature Construction
The script calculates four categories of numerical features for every single segment:

Motion Features: It looks at the segment's speed dynamics. It calculates the avg_speed, 
max_speed, and speed_variance. High speed variance, for instance, is often a sign of erratic behavior.

Spatial/Geographic Features:

Path Length: Uses the Haversine formula to measure the total distance the ship traveled within the window.

Displacement: Measures the straight-line distance from the start to the end of the segment.

Trajectory Curvature: A ratio of path_length to displacement. If this number is high, the 
ship is moving in circles or zig-zags rather than a straight line.

Boundary Proximity: It uses geopandas to perform a "nearest spatial join." It calculates the 
exact distance (in km) from the center of the segment to the nearest EEZ boundary and counts 
how many times the ship crossed that boundary during the segment window.

Temporal Features: It calculates the "time-overlap." If a segment is 60 minutes long and the 
ship was "Anchored" for 30 of those minutes, the feature anchor_duration will be 30.0.

1. Reducing Dimensionality
A single ship's trajectory might have 5,000 GPS points. Training a model on 5,000 individual 
variables for every single ship is computationally expensive and noisy. Feature construction 
compresses those 5,000 points into a handful of descriptive statistics (like avg_speed, max_speed, 
and path_length) that capture the essence of the movement without the "noise" of individual pings.

2. Providing Domain Context
Raw coordinates are just numbers. By engineering features like distance_to_boundary or crossing_count, 
you are giving the model expert maritime knowledge. You are explicitly telling the model: "This specific 
piece of information—the proximity to a border—is important for identifying abnormal behavior."

3. Enabling Pattern Recognition
Machine learning models look for correlations.

Without features: The model struggles to relate a "slow speed" at point A to a "slow speed" at point B.

With features: You provide the model with anchor_duration. Now, the model can clearly see that high anchor 
duration is strongly correlated with, say, a vessel performing cargo transfers. You have created a symbolic 
representation of behavior.

4. Normalizing Irregular Data
AIS data is notoriously messy. Ships turn their transponders off, or they transmit pings at irregular 
intervals (e.g., every 10 seconds vs. every 10 minutes). Feature construction "normalizes" this; by 
calculating features like trajectory_curvature or speed_variance over fixed time windows, you ensure 
that your model is comparing "apples to apples" regardless of how many pings a ship transmitted during that period.

3. Output: The ML-Ready Dataset
The output is features.parquet, a clean, wide-format table. Every row represents a unique segment_id, 
and every column is a numerical feature representing that segment's behavior.

Why this is "ML-Ready":
A machine learning model (like a Random Forest or XGBoost) cannot "read" a JSON file or a raw GPS 
ping directly. It needs a table where every column is a number.

Next:
label_generator.py
'''

REQUIRED_SEGMENT_COLUMNS = [
    "MMSI",
    "timestamp",
    "latitude",
    "longitude",
    "speed",
    "segment_id",
    "segment_start",
    "segment_end",
    "number_of_points",
]
DEFAULT_SEGMENTS_PATH = Path("data/processed/segments.parquet")
DEFAULT_BOUNDARY_EVENTS_PATH = Path("data/processed/boundary_events.parquet")
DEFAULT_EVENT_SEQUENCE_PATH = Path("data/processed/event_sequence.json")
DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_EEZ_PATH = Path("data/raw/eez/eez.shp")
EARTH_RADIUS_KM = 6371.0088


def build_features(
    segments_path: str | Path,
    output_path: str | Path,
    boundary_events_path: str | Path | None = None,
    event_sequence_path: str | Path | None = None,
    eez_path: str | Path | None = None,
) -> pd.DataFrame:
    segments = load_segments(segments_path)
    boundary_events = load_boundary_events(boundary_events_path)
    event_sequence = load_event_sequence(event_sequence_path)

    features = calculate_segment_features(
        segments=segments,
        boundary_events=boundary_events,
        event_sequence=event_sequence,
        eez_path=eez_path,
    )
    write_features(features, output_path)
    return features


def load_segments(segments_path: str | Path) -> pd.DataFrame:
    path = Path(segments_path)
    if not path.exists():
        raise FileNotFoundError(f"Segments file not found: {path}")

    segments = pd.read_parquet(path, engine="pyarrow")
    missing_columns = [column for column in REQUIRED_SEGMENT_COLUMNS if column not in segments]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required segment columns: {missing}")
    return _prepare_segments(segments)


def load_boundary_events(boundary_events_path: str | Path | None) -> pd.DataFrame:
    if boundary_events_path is None:
        return _empty_boundary_events()

    path = Path(boundary_events_path)
    if not path.exists():
        return _empty_boundary_events()

    events = pd.read_parquet(path, engine="pyarrow")
    if events.empty:
        return _empty_boundary_events()

    return _prepare_boundary_events(events)


def load_event_sequence(event_sequence_path: str | Path | None) -> pd.DataFrame:
    if event_sequence_path is None:
        return _empty_event_sequence()

    path = Path(event_sequence_path)
    if not path.exists():
        return _empty_event_sequence()

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    events = pd.DataFrame(payload.get("events", []))
    if events.empty:
        return _empty_event_sequence()

    expected_columns = ["event_type", "mmsi", "start_timestamp", "end_timestamp"]
    missing_columns = [column for column in expected_columns if column not in events]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required event sequence columns: {missing}")

    return _prepare_event_sequence(events)


def calculate_segment_features(
    segments: pd.DataFrame,
    boundary_events: pd.DataFrame | None = None,
    event_sequence: pd.DataFrame | None = None,
    eez_path: str | Path | None = None,
) -> pd.DataFrame:
    segments = _prepare_segments(segments)
    boundary_events = _prepare_boundary_events(
        boundary_events if boundary_events is not None else _empty_boundary_events()
    )
    event_sequence = _prepare_event_sequence(
        event_sequence if event_sequence is not None else _empty_event_sequence()
    )
    base_features = _calculate_motion_and_spatial_features(segments)
    crossing_counts = _calculate_crossing_counts(
        base_features,
        boundary_events,
    )
    temporal_features = _calculate_temporal_features(
        base_features,
        event_sequence,
    )
    distance_features = _calculate_distance_to_boundary(base_features, eez_path)

    features = base_features.merge(crossing_counts, on="segment_id", how="left")
    features = features.merge(distance_features, on="segment_id", how="left")
    features = features.merge(temporal_features, on="segment_id", how="left")
    features["crossing_count"] = features["crossing_count"].fillna(0).astype("int64")
    features["anchor_duration"] = features["anchor_duration"].fillna(0.0)
    features["loiter_duration"] = features["loiter_duration"].fillna(0.0)

    ordered_columns = [
        "segment_id",
        "mmsi",
        "window_minutes",
        "segment_start",
        "segment_end",
        "duration_seconds",
        "number_of_points",
        "avg_speed",
        "max_speed",
        "speed_variance",
        "path_length",
        "displacement",
        "trajectory_curvature",
        "path_tortuosity",
        "heading_variance",
        "mean_heading_change",
        "loop_count",
        "turn_density",
        "stationary_ratio",
        "crossing_count",
        "distance_to_boundary",
        "anchor_duration",
        "loiter_duration",
    ]
    return features[ordered_columns].sort_values(["mmsi", "segment_start", "window_minutes"]).reset_index(
        drop=True
    )


def write_features(features: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, engine="pyarrow", index=False)


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    build_features(
        segments_path=config["output"].get("segments_path", DEFAULT_SEGMENTS_PATH),
        boundary_events_path=config["output"].get(
            "boundary_events_path", DEFAULT_BOUNDARY_EVENTS_PATH
        ),
        event_sequence_path=config["output"].get(
            "event_sequence_path", DEFAULT_EVENT_SEQUENCE_PATH
        ),
        eez_path=config["input"].get("eez_shapefile_path", DEFAULT_EEZ_PATH),
        output_path=config["output"].get("features_path", DEFAULT_FEATURES_PATH),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate segment-level AIS features for situation classification."
    )
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS_PATH)
    parser.add_argument("--boundary-events", type=Path, default=DEFAULT_BOUNDARY_EVENTS_PATH)
    parser.add_argument("--event-sequence", type=Path, default=DEFAULT_EVENT_SEQUENCE_PATH)
    parser.add_argument("--eez", type=Path, default=DEFAULT_EEZ_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config:
        config = _load_config(args.config)
        features = build_features(
            segments_path=config["output"].get("segments_path", DEFAULT_SEGMENTS_PATH),
            boundary_events_path=config["output"].get(
                "boundary_events_path", DEFAULT_BOUNDARY_EVENTS_PATH
            ),
            event_sequence_path=config["output"].get(
                "event_sequence_path", DEFAULT_EVENT_SEQUENCE_PATH
            ),
            eez_path=config["input"].get("eez_shapefile_path", DEFAULT_EEZ_PATH),
            output_path=config["output"].get("features_path", DEFAULT_FEATURES_PATH),
        )
    else:
        features = build_features(
            segments_path=args.segments,
            boundary_events_path=args.boundary_events,
            event_sequence_path=args.event_sequence,
            eez_path=args.eez,
            output_path=args.output,
        )
    print(f"Feature rows: {len(features)}")
    print(f"Wrote: {args.output}")


def _prepare_segments(segments: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_SEGMENT_COLUMNS if column not in segments]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required segment columns: {missing}")

    prepared = segments.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True)
    prepared["segment_start"] = pd.to_datetime(
        prepared["segment_start"],
        errors="coerce",
        utc=True,
    )
    prepared["segment_end"] = pd.to_datetime(
        prepared["segment_end"],
        errors="coerce",
        utc=True,
    )
    for column in ["MMSI", "latitude", "longitude", "speed", "number_of_points"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(
        subset=[
            "MMSI",
            "timestamp",
            "latitude",
            "longitude",
            "segment_id",
            "segment_start",
            "segment_end",
        ]
    )
    prepared["MMSI"] = prepared["MMSI"].astype("int64")
    return prepared.sort_values(["MMSI", "segment_id", "timestamp"]).reset_index(drop=True)


def _prepare_boundary_events(boundary_events: pd.DataFrame) -> pd.DataFrame:
    if boundary_events.empty:
        return _empty_boundary_events()

    missing_columns = [
        column
        for column in ["event_type", "event_timestamp", "mmsi"]
        if column not in boundary_events
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required boundary event columns: {missing}")

    events = boundary_events[["event_type", "event_timestamp", "mmsi"]].copy()
    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"],
        errors="coerce",
        utc=True,
    )
    events["mmsi"] = pd.to_numeric(events["mmsi"], errors="coerce")
    events = events.dropna(subset=["event_timestamp", "mmsi"])
    events["mmsi"] = events["mmsi"].astype("int64")
    return events.sort_values(["mmsi", "event_timestamp"]).reset_index(drop=True)


def _prepare_event_sequence(event_sequence: pd.DataFrame) -> pd.DataFrame:
    if event_sequence.empty:
        return _empty_event_sequence()

    missing_columns = [
        column
        for column in ["event_type", "mmsi", "start_timestamp", "end_timestamp"]
        if column not in event_sequence
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required event sequence columns: {missing}")

    events = event_sequence.copy()
    events["mmsi"] = pd.to_numeric(events["mmsi"], errors="coerce")
    events["start_timestamp"] = pd.to_datetime(
        events["start_timestamp"],
        errors="coerce",
        utc=True,
    )
    events["end_timestamp"] = pd.to_datetime(
        events["end_timestamp"],
        errors="coerce",
        utc=True,
    )
    events = events.dropna(subset=["event_type", "mmsi", "start_timestamp", "end_timestamp"])
    events["mmsi"] = events["mmsi"].astype("int64")
    return events.sort_values(["mmsi", "start_timestamp"]).reset_index(drop=True)


def _calculate_motion_and_spatial_features(segments: pd.DataFrame) -> pd.DataFrame:
    ordered = segments.sort_values(["segment_id", "timestamp"]).copy()
    same_segment_as_previous = ordered["segment_id"].eq(ordered["segment_id"].shift())
    ordered["_previous_latitude"] = ordered["latitude"].shift()
    ordered["_previous_longitude"] = ordered["longitude"].shift()
    ordered["_step_distance_km"] = 0.0
    step_mask = same_segment_as_previous & ordered["_previous_latitude"].notna()
    ordered.loc[step_mask, "_step_distance_km"] = _haversine_km(
        ordered.loc[step_mask, "_previous_latitude"].to_numpy(dtype=float),
        ordered.loc[step_mask, "_previous_longitude"].to_numpy(dtype=float),
        ordered.loc[step_mask, "latitude"].to_numpy(dtype=float),
        ordered.loc[step_mask, "longitude"].to_numpy(dtype=float),
    )
    ordered["_speed_squared"] = ordered["speed"] ** 2

    # New Features
    ordered["_heading_change"] = ordered["heading"].diff().abs()
    ordered["_heading_change"] = np.minimum(ordered["_heading_change"], 360 - ordered["_heading_change"])
    ordered.loc[~step_mask, "_heading_change"] = np.nan # Only diff within same segment
    ordered["_is_stationary"] = ordered["speed"] < 1.0
    ordered["_is_significant_turn"] = ordered["_heading_change"] > 30.0

    grouped = ordered.groupby("segment_id", sort=False)
    features = grouped.agg(
        mmsi=("MMSI", "first"),
        window_minutes=("window_minutes", "first")
        if "window_minutes" in ordered
        else ("MMSI", "size"),
        segment_start=("segment_start", "first"),
        segment_end=("segment_end", "first"),
        number_of_points=("number_of_points", "first"),
        avg_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        speed_squared_mean=("_speed_squared", "mean"),
        path_length=("_step_distance_km", "sum"),
        heading_variance=("heading", "var"),
        mean_heading_change=("_heading_change", "mean"),
        total_heading_change=("_heading_change", "sum"),
        significant_turn_count=("_is_significant_turn", "sum"),
        stationary_ratio=("_is_stationary", "mean"),
        centroid_latitude=("latitude", "mean"),
        centroid_longitude=("longitude", "mean"),
        start_latitude=("latitude", "first"),
        start_longitude=("longitude", "first"),
        end_latitude=("latitude", "last"),
        end_longitude=("longitude", "last"),
    ).reset_index()

    features["duration_seconds"] = (
        features["segment_end"] - features["segment_start"]
    ).dt.total_seconds()
    features["speed_variance"] = (
        features["speed_squared_mean"] - features["avg_speed"] ** 2
    ).clip(lower=0.0)
    features["displacement"] = _haversine_km(
        features["start_latitude"].to_numpy(dtype=float),
        features["start_longitude"].to_numpy(dtype=float),
        features["end_latitude"].to_numpy(dtype=float),
        features["end_longitude"].to_numpy(dtype=float),
    )
    features["trajectory_curvature"] = np.where(
        features["displacement"] > 0,
        features["path_length"] / features["displacement"],
        0.0,
    )
    features["path_tortuosity"] = features["trajectory_curvature"]
    features["loop_count"] = features["total_heading_change"] / 360.0
    features["turn_density"] = np.where(
        features["path_length"] > 0, 
        features["significant_turn_count"] / features["path_length"], 
        0.0
    )
    features["mmsi"] = features["mmsi"].astype("int64")
    features["number_of_points"] = features["number_of_points"].astype("int64")

    # --- Fallback: if AIS-reported speed is entirely missing (all NaN), derive
    # avg_speed from the GPS-computed path_length and segment duration.
    # 1 knot = 1.852 km/h.  path_length is in km, duration_seconds in seconds.
    # Formula: speed_knots = (path_length_km / duration_seconds) * 3600 / 1.852
    if features["avg_speed"].isna().all():
        safe_duration = features["duration_seconds"].replace(0, np.nan)
        features["avg_speed"] = (
            features["path_length"] / safe_duration * 3600.0 / 1.852
        ).fillna(0.0)
        features["max_speed"] = features["avg_speed"]  # best estimate available
        # Variance is unknown when speed is derived from positions only
        features["speed_variance"] = 0.0

    return features.drop(
        columns=[
            "speed_squared_mean",
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
        ]
    )


def _calculate_crossing_counts(
    features: pd.DataFrame,
    boundary_events: pd.DataFrame,
) -> pd.DataFrame:
    crossing_counts = pd.Series(0, index=features["segment_id"], dtype="int64")
    if boundary_events.empty:
        return crossing_counts.rename("crossing_count").reset_index()

    for mmsi, vessel_segments in features.groupby("mmsi", sort=False):
        vessel_events = boundary_events.loc[boundary_events["mmsi"] == mmsi]
        if vessel_events.empty:
            continue
        event_times = vessel_events["event_timestamp"].sort_values().to_numpy()
        starts = vessel_segments["segment_start"].to_numpy()
        ends = vessel_segments["segment_end"].to_numpy()
        left = np.searchsorted(event_times, starts, side="left")
        right = np.searchsorted(event_times, ends, side="right")
        counts = right - left
        crossing_counts.loc[vessel_segments["segment_id"].to_numpy()] = counts

    return crossing_counts.rename("crossing_count").reset_index()


def _calculate_temporal_features(
    features: pd.DataFrame,
    event_sequence: pd.DataFrame,
) -> pd.DataFrame:
    temporal = pd.DataFrame(
        {
            "segment_id": features["segment_id"],
            "anchor_duration": 0.0,
            "loiter_duration": 0.0,
        }
    )
    if event_sequence.empty:
        return temporal

    relevant_events = event_sequence.loc[
        event_sequence["event_type"].isin(["ANCHOR", "LOITER"])
    ].copy()
    if relevant_events.empty:
        return temporal

    temporal = temporal.set_index("segment_id")
    for mmsi, vessel_segments in features.groupby("mmsi", sort=False):
        vessel_events = relevant_events.loc[relevant_events["mmsi"] == mmsi]
        if vessel_events.empty:
            continue

        for segment in vessel_segments.itertuples(index=False):
            overlapping = vessel_events.loc[
                (vessel_events["end_timestamp"] >= segment.segment_start)
                & (vessel_events["start_timestamp"] <= segment.segment_end)
            ]
            for event in overlapping.itertuples(index=False):
                overlap_minutes = _overlap_minutes(
                    segment.segment_start,
                    segment.segment_end,
                    event.start_timestamp,
                    event.end_timestamp,
                )
                if event.event_type == "ANCHOR":
                    temporal.loc[segment.segment_id, "anchor_duration"] += overlap_minutes
                elif event.event_type == "LOITER":
                    temporal.loc[segment.segment_id, "loiter_duration"] += overlap_minutes

    return temporal.reset_index()


def _calculate_distance_to_boundary(
    features: pd.DataFrame,
    eez_path: str | Path | None,
) -> pd.DataFrame:
    distances = pd.DataFrame(
        {
            "segment_id": features["segment_id"],
            "distance_to_boundary": np.nan,
        }
    )
    if eez_path is None or not Path(eez_path).exists() or features.empty:
        return distances

    eez = load_eez_polygons(eez_path)
    boundary = gpd.GeoDataFrame(geometry=eez.geometry.boundary, crs=eez.crs)
    boundary = boundary.loc[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        return distances

    centroids = gpd.GeoDataFrame(
        features[["segment_id"]].copy(),
        geometry=gpd.points_from_xy(
            features["centroid_longitude"],
            features["centroid_latitude"],
            crs="EPSG:4326",
        ),
        crs="EPSG:4326",
    )
    projected_centroids = centroids.to_crs("EPSG:3857")
    projected_boundary = boundary.to_crs("EPSG:3857")
    _ = projected_boundary.sindex

    nearest = gpd.sjoin_nearest(
        projected_centroids,
        projected_boundary,
        how="left",
        distance_col="distance_meters",
    )
    nearest = nearest.groupby("segment_id", as_index=False)["distance_meters"].min()
    nearest["distance_to_boundary"] = nearest["distance_meters"] / 1000.0
    return distances[["segment_id"]].merge(
        nearest[["segment_id", "distance_to_boundary"]],
        on="segment_id",
        how="left",
    )


def _path_length_km(latitudes: pd.Series, longitudes: pd.Series) -> float:
    if len(latitudes) < 2:
        return 0.0
    return float(
        np.nansum(
            _haversine_km(
                latitudes.iloc[:-1].to_numpy(dtype=float),
                longitudes.iloc[:-1].to_numpy(dtype=float),
                latitudes.iloc[1:].to_numpy(dtype=float),
                longitudes.iloc[1:].to_numpy(dtype=float),
            )
        )
    )


def _displacement_km(latitudes: pd.Series, longitudes: pd.Series) -> float:
    if len(latitudes) < 2:
        return 0.0
    return float(
        _haversine_km(
            np.array([latitudes.iloc[0]], dtype=float),
            np.array([longitudes.iloc[0]], dtype=float),
            np.array([latitudes.iloc[-1]], dtype=float),
            np.array([longitudes.iloc[-1]], dtype=float),
        )[0]
    )


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _duration_seconds(segment: pd.DataFrame) -> float:
    if "duration_seconds" in segment:
        return float(segment["duration_seconds"].iloc[0])
    return float(
        (segment["segment_end"].iloc[0] - segment["segment_start"].iloc[0]).total_seconds()
    )


def _first_or_none(segment: pd.DataFrame, column: str) -> Any:
    return segment[column].iloc[0] if column in segment else None


def _overlap_minutes(
    segment_start: pd.Timestamp,
    segment_end: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> float:
    start = max(segment_start, event_start)
    end = min(segment_end, event_end)
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 60.0


def _empty_boundary_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": pd.Series(dtype="object"),
            "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "mmsi": pd.Series(dtype="int64"),
        }
    )


def _empty_event_sequence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": pd.Series(dtype="object"),
            "mmsi": pd.Series(dtype="int64"),
            "start_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "end_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()
