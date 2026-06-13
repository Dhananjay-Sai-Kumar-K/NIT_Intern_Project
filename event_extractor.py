from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

'''
Event Extraction
Purpose: convert low-level AIS movement into symbolic events.

Events implemented:
---ENTRY
---EXIT
---ANCHOR
---LOITER
---REPEATED_CROSSING
---MANEUVERING

Methods used:
---Rule-based event extraction
---Threshold-based detection
---Per-vessel temporal scanning
---AIS gap-aware run splitting
---Circular heading variance
---Haversine distance
---Sliding 24-hour crossing window

Rules:
ANCHOR:
---speed < 0.5 knots
---duration > 60 minutes

LOITER:
---speed < 2 knots
---within 2 km radius
---duration > 2 hours

MANEUVERING:
---high circular heading variance
---enough AIS points in window
---REPEATED_CROSSING:
---more than 3 EEZ entry/exit events within 24 hours

False-positive control:
---ANCHOR suppresses overlapping LOITER.
---AIS gaps break continuous event runs.
---Boundary crossing is treated as neutral, not suspicious.
---Maneuvering uses circular variance, not normal linear variance.

1. Inputs
Cleaned AIS Data: The same cleaned_ais.parquet you generated in the first step. 
This provides the raw positional data.

Boundary Events: The boundary_events.parquet file created by your previous "Geofencing" script. 
This provides the context for when a ship enters or leaves an EEZ.

Configuration/Thresholds: Parameters defined in a config.yaml that set the "sensitivity" for
events (e.g., how long a ship must be still to be considered "Anchored").

2. The Process: Rule-Based Logic
This script acts as an "interpreter" that reads raw GPS coordinates and labels 
them with behavior. It uses a Per-vessel temporal scan, looking for specific patterns in the data:

Anchor Detection: Identifies segments where the speed remains below 0.5 knots for more than 60 minutes.

Loiter Detection: Identifies segments where the ship stays within a 2 km radius at 
a slow speed (< 2 knots) for a long duration.

Logic constraint: It ensures that ANCHOR events are prioritized, meaning if a ship is 
anchored, it won't be double-counted as "loitering."

Maneuvering Detection: Uses Circular Heading Variance to see if a ship is changing 
direction frequently. If the variance is high over a 30-minute window, it marks the ship as "maneuvering."

Repeated Crossing Detection: Scans the boundary_events to see if a vessel has crossed 
an EEZ boundary more than 3 times in 24 hours (a common signal for irregular or suspicious activity).

3. Output: The Event Sequence
The final result is event_sequence.json, a structured file that tells a story for every vessel:

Format: A JSON object containing metadata about the thresholds used and a chronological list of events.
Content: Each event record includes:
event_type: (e.g., ANCHOR, ENTRY, EXIT, MANEUVERING, REPEATED_CROSSING).
start_timestamp & end_timestamp: When the behavior began and ended.
duration_minutes: How long the behavior lasted.
details: A technical dictionary explaining why it was classified that way 
(e.g., the mean speed during the anchor or the heading variance calculated).

Next:
feature_engineering.py
'''

AIS_COLUMNS = ["MMSI", "timestamp", "latitude", "longitude", "speed", "heading"]
BOUNDARY_COLUMNS = ["event_type", "event_timestamp", "mmsi"]
DEFAULT_AIS_PATH = Path("data/processed/cleaned_ais.parquet")
DEFAULT_BOUNDARY_EVENTS_PATH = Path("data/processed/boundary_events.parquet")
DEFAULT_EVENT_SEQUENCE_PATH = Path("data/processed/event_sequence.json")
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class EventThresholds:
    anchor_speed_knots: float = 0.5
    anchor_min_duration_minutes: float = 60.0
    loiter_speed_knots: float = 2.0
    loiter_radius_km: float = 2.0
    loiter_min_duration_minutes: float = 120.0
    max_ais_gap_minutes: float = 30.0
    maneuver_window_minutes: int = 30
    maneuver_min_points: int = 10
    maneuver_min_point_coverage_ratio: float = 0.5
    maneuver_heading_circular_variance_threshold: float = 0.5
    repeated_crossing_count: int = 3
    repeated_crossing_window_hours: float = 24.0


def load_ais_points(ais_path: str | Path) -> pd.DataFrame:
    path = Path(ais_path)
    if not path.exists():
        raise FileNotFoundError(f"AIS file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow", columns=AIS_COLUMNS)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=lambda column: column in AIS_COLUMNS)
    raise ValueError(f"Unsupported AIS file type: {path.suffix}")


def load_boundary_events(boundary_events_path: str | Path) -> pd.DataFrame:
    path = Path(boundary_events_path)
    if not path.exists():
        return _empty_boundary_events()

    if path.suffix.lower() == ".parquet":
        events = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        events = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported boundary event file type: {path.suffix}")

    missing_columns = [column for column in BOUNDARY_COLUMNS if column not in events]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required boundary event columns: {missing}")
    return events[BOUNDARY_COLUMNS]


def extract_event_sequence(
    ais_points: pd.DataFrame,
    boundary_events: pd.DataFrame | None = None,
    thresholds: EventThresholds | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or EventThresholds()
    ais = _prepare_ais_points(ais_points)
    boundary = _prepare_boundary_events(boundary_events)
    anchor_events = _extract_anchor_events(ais, thresholds)

    event_frames = [
        _boundary_events_to_sequence(boundary),
        anchor_events,
        _extract_loiter_events(ais, thresholds, anchor_events),
        _extract_maneuvering_events(ais, thresholds),
        _extract_repeated_crossing_events(boundary, thresholds),
    ]
    event_frames = [frame for frame in event_frames if not frame.empty]
    if not event_frames:
        return []

    event_sequence = pd.concat(event_frames, ignore_index=True)
    event_sequence = event_sequence.sort_values(
        ["event_timestamp", "mmsi", "event_type"]
    ).reset_index(drop=True)
    return [_record_to_json(record) for record in event_sequence.to_dict(orient="records")]


def build_event_sequence(
    ais_path: str | Path,
    boundary_events_path: str | Path,
    output_path: str | Path,
    thresholds: EventThresholds | None = None,
) -> list[dict[str, Any]]:
    ais_points = load_ais_points(ais_path)
    boundary_events = load_boundary_events(boundary_events_path)
    events = extract_event_sequence(
        ais_points=ais_points,
        boundary_events=boundary_events,
        thresholds=thresholds,
    )
    write_event_sequence(events, output_path, thresholds or EventThresholds())
    return events


def write_event_sequence(
    events: list[dict[str, Any]],
    output_path: str | Path,
    thresholds: EventThresholds,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": asdict(thresholds),
        "event_count": len(events),
        "events": events,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    thresholds = EventThresholds(**config.get("event_extraction", {}))
    build_event_sequence(
        ais_path=config["output"].get("cleaned_ais_path", DEFAULT_AIS_PATH),
        boundary_events_path=config["output"].get(
            "boundary_events_path", DEFAULT_BOUNDARY_EVENTS_PATH
        ),
        output_path=config["output"].get(
            "event_sequence_path", DEFAULT_EVENT_SEQUENCE_PATH
        ),
        thresholds=thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract maritime events from cleaned AIS and boundary crossings."
    )
    parser.add_argument("--ais-points", type=Path, default=DEFAULT_AIS_PATH)
    parser.add_argument(
        "--boundary-events",
        type=Path,
        default=DEFAULT_BOUNDARY_EVENTS_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EVENT_SEQUENCE_PATH)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config:
        config = _load_config(args.config)
        thresholds = EventThresholds(**config.get("event_extraction", {}))
    else:
        thresholds = EventThresholds()

    events = build_event_sequence(
        ais_path=args.ais_points,
        boundary_events_path=args.boundary_events,
        output_path=args.output,
        thresholds=thresholds,
    )
    print(f"Events: {len(events)}")
    print(f"Wrote: {args.output}")


def _prepare_ais_points(ais_points: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in AIS_COLUMNS if column not in ais_points]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required AIS columns: {missing}")

    ais = ais_points[AIS_COLUMNS].copy()
    ais["timestamp"] = pd.to_datetime(ais["timestamp"], errors="coerce", utc=True)
    for column in ["MMSI", "latitude", "longitude", "speed", "heading"]:
        ais[column] = pd.to_numeric(ais[column], errors="coerce")
    ais = ais.dropna(subset=["MMSI", "timestamp", "latitude", "longitude"])
    ais["MMSI"] = ais["MMSI"].astype("int64")
    return ais.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)


def _prepare_boundary_events(boundary_events: pd.DataFrame | None) -> pd.DataFrame:
    if boundary_events is None or boundary_events.empty:
        return _empty_boundary_events()

    events = boundary_events[BOUNDARY_COLUMNS].copy()
    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"],
        errors="coerce",
        utc=True,
    )
    events["mmsi"] = pd.to_numeric(events["mmsi"], errors="coerce")
    events = events.dropna(subset=["event_type", "event_timestamp", "mmsi"])
    events["mmsi"] = events["mmsi"].astype("int64")
    return events.sort_values(["mmsi", "event_timestamp"]).reset_index(drop=True)


def _boundary_events_to_sequence(boundary_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in boundary_events.itertuples(index=False):
        event_type = _normalize_boundary_event_type(str(event.event_type))
        if event_type is None:
            continue
        rows.append(
            {
                "event_type": event_type,
                "event_timestamp": event.event_timestamp,
                "mmsi": int(event.mmsi),
                "start_timestamp": event.event_timestamp,
                "end_timestamp": event.event_timestamp,
                "duration_minutes": 0.0,
                "details": {},
            }
        )
    return _events_frame(rows)


def _extract_anchor_events(
    ais: pd.DataFrame,
    thresholds: EventThresholds,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    duration_threshold = pd.Timedelta(minutes=thresholds.anchor_min_duration_minutes)
    max_gap = pd.Timedelta(minutes=thresholds.max_ais_gap_minutes)

    for mmsi, vessel_points in ais.groupby("MMSI", sort=True):
        mask = vessel_points["speed"] < thresholds.anchor_speed_knots
        for run in _iter_true_runs(vessel_points, mask, max_gap=max_gap):
            duration = run["timestamp"].iloc[-1] - run["timestamp"].iloc[0]
            if duration <= duration_threshold:
                continue
            rows.append(
                _duration_event(
                    event_type="ANCHOR",
                    mmsi=int(mmsi),
                    start=run["timestamp"].iloc[0],
                    end=run["timestamp"].iloc[-1],
                    details={
                        "speed_threshold_knots": thresholds.anchor_speed_knots,
                        "mean_speed_knots": float(run["speed"].mean()),
                        "number_of_points": int(len(run)),
                    },
                )
            )
    return _events_frame(rows)


def _extract_loiter_events(
    ais: pd.DataFrame,
    thresholds: EventThresholds,
    anchor_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    duration_threshold = pd.Timedelta(minutes=thresholds.loiter_min_duration_minutes)
    max_gap = pd.Timedelta(minutes=thresholds.max_ais_gap_minutes)
    anchor_events = anchor_events if anchor_events is not None else _events_frame([])

    for mmsi, vessel_points in ais.groupby("MMSI", sort=True):
        mask = vessel_points["speed"] < thresholds.loiter_speed_knots
        for run in _iter_true_runs(vessel_points, mask, max_gap=max_gap):
            duration = run["timestamp"].iloc[-1] - run["timestamp"].iloc[0]
            if duration <= duration_threshold:
                continue
            if _overlaps_existing_event(
                int(mmsi),
                run["timestamp"].iloc[0],
                run["timestamp"].iloc[-1],
                anchor_events,
            ):
                continue
            radius_km = _max_radius_from_centroid_km(run["latitude"], run["longitude"])
            if radius_km > thresholds.loiter_radius_km:
                continue
            rows.append(
                _duration_event(
                    event_type="LOITER",
                    mmsi=int(mmsi),
                    start=run["timestamp"].iloc[0],
                    end=run["timestamp"].iloc[-1],
                    details={
                        "speed_threshold_knots": thresholds.loiter_speed_knots,
                        "radius_threshold_km": thresholds.loiter_radius_km,
                        "observed_radius_km": radius_km,
                        "mean_speed_knots": float(run["speed"].mean()),
                        "number_of_points": int(len(run)),
                    },
                )
            )
    return _events_frame(rows)


def _extract_maneuvering_events(
    ais: pd.DataFrame,
    thresholds: EventThresholds,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    window_delta = pd.Timedelta(minutes=thresholds.maneuver_window_minutes)
    expected_points = max(1, int(thresholds.maneuver_min_points))
    min_coverage = max(0.0, min(1.0, thresholds.maneuver_min_point_coverage_ratio))

    for mmsi, vessel_points in ais.groupby("MMSI", sort=True):
        vessel = vessel_points.dropna(subset=["heading"]).copy()
        if vessel.empty:
            continue
        trajectory_start = vessel["timestamp"].min()
        window_index = ((vessel["timestamp"] - trajectory_start) // window_delta).astype(
            "int64"
        )
        for _, window in vessel.groupby(window_index, sort=True):
            if len(window) < thresholds.maneuver_min_points:
                continue
            if _point_coverage_ratio(window, window_delta, expected_points) < min_coverage:
                continue
            circular_variance = _heading_circular_variance(window["heading"])
            if circular_variance < thresholds.maneuver_heading_circular_variance_threshold:
                continue
            rows.append(
                _duration_event(
                    event_type="MANEUVERING",
                    mmsi=int(mmsi),
                    start=window["timestamp"].iloc[0],
                    end=window["timestamp"].iloc[-1],
                    details={
                        "window_minutes": thresholds.maneuver_window_minutes,
                        "heading_circular_variance": circular_variance,
                        "circular_variance_unit": "unitless_0_to_1",
                        "circular_variance_threshold": (
                            thresholds.maneuver_heading_circular_variance_threshold
                        ),
                        "point_coverage_ratio": _point_coverage_ratio(
                            window,
                            window_delta,
                            expected_points,
                        ),
                        "min_point_coverage_ratio": min_coverage,
                        "number_of_points": int(len(window)),
                    },
                )
            )
    return _events_frame(rows)


def _extract_repeated_crossing_events(
    boundary_events: pd.DataFrame,
    thresholds: EventThresholds,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    crossings = _boundary_events_to_sequence(boundary_events)
    if crossings.empty:
        return _events_frame(rows)

    window_delta = pd.Timedelta(hours=thresholds.repeated_crossing_window_hours)
    for mmsi, vessel_crossings in crossings.groupby("mmsi", sort=True):
        timestamps = vessel_crossings["event_timestamp"].tolist()
        left = 0
        last_reported_end: pd.Timestamp | None = None
        for right, timestamp in enumerate(timestamps):
            while timestamp - timestamps[left] > window_delta:
                left += 1
            crossing_count = right - left + 1
            if crossing_count <= thresholds.repeated_crossing_count:
                continue
            if last_reported_end is not None and timestamp <= last_reported_end:
                continue
            start = timestamps[left]
            rows.append(
                {
                    "event_type": "REPEATED_CROSSING",
                    "event_timestamp": timestamp,
                    "mmsi": int(mmsi),
                    "start_timestamp": start,
                    "end_timestamp": timestamp,
                    "duration_minutes": (timestamp - start).total_seconds() / 60.0,
                    "details": {
                        "crossing_count": int(crossing_count),
                        "crossing_threshold": thresholds.repeated_crossing_count,
                        "window_hours": thresholds.repeated_crossing_window_hours,
                    },
                }
            )
            last_reported_end = timestamp
    return _events_frame(rows)


def _iter_true_runs(
    vessel_points: pd.DataFrame,
    mask: pd.Series,
    max_gap: pd.Timedelta | None = None,
) -> list[pd.DataFrame]:
    if vessel_points.empty:
        return []

    gap_breaks = pd.Series(False, index=vessel_points.index)
    if max_gap is not None:
        gap_breaks = vessel_points["timestamp"].diff().gt(max_gap).fillna(False)
    run_ids = (mask.ne(mask.shift(fill_value=False)) | gap_breaks).cumsum()
    runs: list[pd.DataFrame] = []
    for _, run in vessel_points.loc[mask].groupby(run_ids[mask], sort=True):
        runs.append(run)
    return runs


def _duration_event(
    event_type: str,
    mmsi: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    details: dict[str, Any],
) -> dict[str, Any]:
    duration_minutes = (end - start).total_seconds() / 60.0
    return {
        "event_type": event_type,
        "event_timestamp": start,
        "mmsi": mmsi,
        "start_timestamp": start,
        "end_timestamp": end,
        "duration_minutes": duration_minutes,
        "details": details,
    }


def _overlaps_existing_event(
    mmsi: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    events: pd.DataFrame,
) -> bool:
    if events.empty:
        return False
    vessel_events = events.loc[events["mmsi"] == mmsi]
    if vessel_events.empty:
        return False
    overlaps = (vessel_events["end_timestamp"] >= start) & (
        vessel_events["start_timestamp"] <= end
    )
    return bool(overlaps.any())


def _point_coverage_ratio(
    window: pd.DataFrame,
    window_delta: pd.Timedelta,
    expected_points: int,
) -> float:
    if window.empty:
        return 0.0
    observed_span = window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]
    time_coverage = min(1.0, observed_span / window_delta) if window_delta.total_seconds() else 0.0
    point_coverage = min(1.0, len(window) / expected_points)
    return float(min(time_coverage, point_coverage))


def _heading_circular_variance(headings: pd.Series) -> float:
    radians = np.deg2rad(pd.to_numeric(headings, errors="coerce").dropna() % 360)
    if len(radians) == 0:
        return 0.0
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()
    resultant_length = math.sqrt(mean_sin**2 + mean_cos**2)
    return float(1.0 - resultant_length)


def _max_radius_from_centroid_km(
    latitudes: pd.Series,
    longitudes: pd.Series,
) -> float:
    center_latitude = float(latitudes.mean())
    center_longitude = float(longitudes.mean())
    distances = _haversine_km(
        latitudes.to_numpy(dtype=float),
        longitudes.to_numpy(dtype=float),
        center_latitude,
        center_longitude,
    )
    return float(np.nanmax(distances)) if len(distances) else 0.0


def _haversine_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    center_latitude: float,
    center_longitude: float,
) -> np.ndarray:
    lat1 = np.deg2rad(latitudes)
    lon1 = np.deg2rad(longitudes)
    lat2 = math.radians(center_latitude)
    lon2 = math.radians(center_longitude)

    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _normalize_boundary_event_type(event_type: str) -> str | None:
    normalized = event_type.strip().lower().replace("_", " ")
    if "entry" in normalized or normalized == "enter":
        return "ENTRY"
    if "exit" in normalized or normalized == "leave":
        return "EXIT"
    return None


def _events_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "event_type",
        "event_timestamp",
        "mmsi",
        "start_timestamp",
        "end_timestamp",
        "duration_minutes",
        "details",
    ]
    return pd.DataFrame(rows, columns=columns)


def _record_to_json(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": record["event_type"],
        "event_timestamp": _json_value(record["event_timestamp"]),
        "mmsi": int(record["mmsi"]),
        "start_timestamp": _json_value(record["start_timestamp"]),
        "end_timestamp": _json_value(record["end_timestamp"]),
        "duration_minutes": _json_value(record["duration_minutes"]),
        "details": _json_value(record["details"]),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _empty_boundary_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": pd.Series(dtype="object"),
            "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "mmsi": pd.Series(dtype="int64"),
        }
    )


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()
