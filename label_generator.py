from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import json

REQUIRED_FEATURE_COLUMNS = [
    "segment_id",
    "mmsi",
    "segment_start",
    "segment_end",
    "avg_speed",
    "speed_variance",
    "trajectory_curvature",
    "crossing_count",
    "distance_to_boundary",
    "anchor_duration",
    "loiter_duration",
]
DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_LABELS_PATH = Path("data/processed/situation_labels.parquet")


@dataclass(frozen=True)
class LabelThresholds:
    transit_min_avg_speed: float = 5.0
    transit_max_speed_variance: float = 4.0
    transit_max_trajectory_curvature: float = 2.5
    anchorage_min_anchor_duration_minutes: float = 20.0
    boundary_crossing_min_crossings: int = 2
    prolonged_boundary_max_distance_km: float = 5.0
    prolonged_boundary_min_loiter_duration_minutes: float = 20.0
    fishing_confidence: float = 0.95
    anchorage_confidence: float = 0.9
    prolonged_boundary_confidence: float = 0.85
    boundary_crossing_confidence: float = 0.8
    transit_confidence: float = 0.75
    unknown_confidence: float = 0.25


def build_situation_labels(
    features_path: str | Path,
    output_path: str | Path,
    thresholds: LabelThresholds | None = None,
    fishing_labels_path: str | Path | None = None,
) -> pd.DataFrame:
    features = load_features(features_path)
    fishing_labels = load_fishing_labels(fishing_labels_path)
    labels = generate_situation_labels(features, thresholds or LabelThresholds(), fishing_labels)
    write_situation_labels(labels, output_path)
    return labels


def load_features(features_path: str | Path) -> pd.DataFrame:
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    features = pd.read_parquet(path, engine="pyarrow")
    missing_columns = [column for column in REQUIRED_FEATURE_COLUMNS if column not in features]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required feature columns: {missing}")
    return _prepare_features(features)


def load_fishing_labels(fishing_labels_path: str | Path | None) -> pd.DataFrame:
    if fishing_labels_path is None:
        return _empty_fishing_labels()

    path = Path(fishing_labels_path)
    if not path.exists():
        return _empty_fishing_labels()

    if path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        labels = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported fishing label file type: {path.suffix}")

    return _prepare_fishing_labels(labels)


def generate_situation_labels(
    features: pd.DataFrame,
    thresholds: LabelThresholds | None = None,
    fishing_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or LabelThresholds()
    features = _prepare_features(features)
    fishing_labels = _prepare_fishing_labels(
        fishing_labels if fishing_labels is not None else _empty_fishing_labels()
    )

    labels = pd.DataFrame(
        {
            "segment_id": features["segment_id"],
            "mmsi": features["mmsi"],
            "segment_start": features["segment_start"],
            "segment_end": features["segment_end"],
            "situation_label": "Unknown",
            "label_confidence": thresholds.unknown_confidence,
            "label_source": "default_unknown",
            "rule_details": [{} for _ in range(len(features))],
        }
    )

    _apply_transit_rule(labels, features, thresholds)
    _apply_boundary_crossing_rule(labels, features, thresholds)
    _apply_prolonged_boundary_presence_rule(labels, features, thresholds)
    _apply_anchorage_rule(labels, features, thresholds)
    _apply_fishing_labels(labels, features, fishing_labels, thresholds)
    return labels.sort_values(["mmsi", "segment_start", "segment_id"]).reset_index(drop=True)


def write_situation_labels(labels: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = labels.copy()

    if "rule_details" in labels.columns:
        labels["rule_details"] = labels["rule_details"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )

    labels.to_parquet(path, engine="pyarrow", index=False)
    print(labels["situation_label"].value_counts())


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    thresholds = LabelThresholds(**config.get("weak_supervision", {}))
    build_situation_labels(
        features_path=config["output"].get("features_path", DEFAULT_FEATURES_PATH),
        output_path=config["output"].get("situation_labels_path", DEFAULT_LABELS_PATH),
        thresholds=thresholds,
        fishing_labels_path=config.get("input", {}).get("gfw_fishing_labels_path"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weak supervision situation labels from AIS segment features."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--fishing-labels", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config:
        config = _load_config(args.config)
        thresholds = LabelThresholds(**config.get("weak_supervision", {}))
    else:
        thresholds = LabelThresholds()

    labels = build_situation_labels(
        features_path=args.features,
        output_path=args.output,
        thresholds=thresholds,
        fishing_labels_path=args.fishing_labels,
    )
    print(f"Label rows: {len(labels)}")
    print("Label counts:")
    print(labels["situation_label"].value_counts().to_string())
    print(f"Wrote: {args.output}")


def _apply_transit_rule(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: LabelThresholds,
) -> None:
    mask = (
        (features["avg_speed"] >= thresholds.transit_min_avg_speed)
        & (features["speed_variance"] <= thresholds.transit_max_speed_variance)
        & (features["trajectory_curvature"] <= thresholds.transit_max_trajectory_curvature)
        & (features["crossing_count"] == 0)
        & (features["anchor_duration"] == 0)
        & (features["loiter_duration"] == 0)
    )
    _assign(
        labels,
        mask,
        label="Transit",
        confidence=thresholds.transit_confidence,
        source="rule_transit",
        details={
            "avg_speed_min": thresholds.transit_min_avg_speed,
            "speed_variance_max": thresholds.transit_max_speed_variance,
            "trajectory_curvature_max": thresholds.transit_max_trajectory_curvature,
        },
        min_existing_confidence=thresholds.transit_confidence,
    )


def _apply_boundary_crossing_rule(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: LabelThresholds,
) -> None:
    mask = features["crossing_count"] >= thresholds.boundary_crossing_min_crossings
    _assign(
        labels,
        mask,
        label="Boundary Crossing",
        confidence=thresholds.boundary_crossing_confidence,
        source="rule_boundary_crossing",
        details={"crossing_count_min": thresholds.boundary_crossing_min_crossings},
        min_existing_confidence=thresholds.boundary_crossing_confidence,
    )


def _apply_prolonged_boundary_presence_rule(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: LabelThresholds,
) -> None:
    mask = (
        (features["distance_to_boundary"] <= thresholds.prolonged_boundary_max_distance_km)
        & (features["loiter_duration"] >= thresholds.prolonged_boundary_min_loiter_duration_minutes)
    )
    _assign(
        labels,
        mask,
        label="Prolonged Boundary Presence",
        confidence=thresholds.prolonged_boundary_confidence,
        source="rule_prolonged_boundary_presence",
        details={
            "distance_to_boundary_max_km": thresholds.prolonged_boundary_max_distance_km,
            "loiter_duration_min_minutes": (
                thresholds.prolonged_boundary_min_loiter_duration_minutes
            ),
        },
        min_existing_confidence=thresholds.prolonged_boundary_confidence,
    )


def _apply_anchorage_rule(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: LabelThresholds,
) -> None:
    mask = features["anchor_duration"] >= thresholds.anchorage_min_anchor_duration_minutes
    _assign(
        labels,
        mask,
        label="Anchorage",
        confidence=thresholds.anchorage_confidence,
        source="rule_anchorage",
        details={
            "anchor_duration_min_minutes": thresholds.anchorage_min_anchor_duration_minutes
        },
        min_existing_confidence=thresholds.anchorage_confidence,
    )


def _apply_fishing_labels(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    fishing_labels: pd.DataFrame,
    thresholds: LabelThresholds,
) -> None:
    if fishing_labels.empty:
        return

    segment_ids = _matching_fishing_segment_ids(features, fishing_labels)
    mask = features["segment_id"].isin(segment_ids)
    _assign(
        labels,
        mask,
        label="Fishing Activity",
        confidence=thresholds.fishing_confidence,
        source="gfw_fishing_label",
        details={"source": "Global Fishing Watch"},
        min_existing_confidence=thresholds.fishing_confidence,
    )


def _matching_fishing_segment_ids(
    features: pd.DataFrame,
    fishing_labels: pd.DataFrame,
) -> set[str]:
    if "segment_id" in fishing_labels:
        return set(fishing_labels.loc[fishing_labels["is_fishing"], "segment_id"].astype(str))

    matches: set[str] = set()
    fishing = fishing_labels.loc[fishing_labels["is_fishing"]]
    for mmsi, vessel_segments in features.groupby("mmsi", sort=False):
        vessel_fishing = fishing.loc[fishing["mmsi"] == mmsi]
        if vessel_fishing.empty:
            continue
        for fishing_event in vessel_fishing.itertuples(index=False):
            mask = (vessel_segments["segment_end"] >= fishing_event.start_timestamp) & (
                vessel_segments["segment_start"] <= fishing_event.end_timestamp
            )
            matches.update(vessel_segments.loc[mask, "segment_id"].astype(str).tolist())
    return matches


def _assign(
    labels: pd.DataFrame,
    mask: pd.Series,
    label: str,
    confidence: float,
    source: str,
    details: dict[str, Any],
    min_existing_confidence: float,  # Kept for signature compatibility
) -> None:
    """
    Assigns a weak label only if the row is currently 'Unknown' OR 
    if the incoming rule provides a strictly higher confidence score.
    """
    # A row is eligible if the mask is true AND (it's unassigned OR the new confidence is higher)
    eligible = mask & (
        (labels["situation_label"] == "Unknown") | 
        (confidence > labels["label_confidence"])
    )
    
    if eligible.any():
        labels.loc[eligible, "situation_label"] = label
        labels.loc[eligible, "label_confidence"] = confidence
        labels.loc[eligible, "label_source"] = source
        
        # Safe object assignment for list of dicts in Pandas
        for idx in labels[eligible].index:
            labels.at[idx, "rule_details"] = details.copy()


def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_FEATURE_COLUMNS if column not in features]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required feature columns: {missing}")

    prepared = features.copy()
    prepared["segment_id"] = prepared["segment_id"].astype(str)
    prepared["segment_start"] = pd.to_datetime(
        prepared["segment_start"],
        errors="coerce",
        utc=True,
    )
    prepared["segment_end"] = pd.to_datetime(prepared["segment_end"], errors="coerce", utc=True)
    for column in [
        "mmsi",
        "avg_speed",
        "speed_variance",
        "trajectory_curvature",
        "crossing_count",
        "distance_to_boundary",
        "anchor_duration",
        "loiter_duration",
    ]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["segment_id", "mmsi", "segment_start", "segment_end"])
    prepared["mmsi"] = prepared["mmsi"].astype("int64")
    for column in [
        "avg_speed",
        "speed_variance",
        "trajectory_curvature",
        "crossing_count",
        "distance_to_boundary",
        "anchor_duration",
        "loiter_duration",
    ]:
        prepared[column] = prepared[column].fillna(0.0)
    return prepared.sort_values(["mmsi", "segment_start", "segment_id"]).reset_index(drop=True)


def _prepare_fishing_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return _empty_fishing_labels()

    prepared = labels.copy()
    if "is_fishing" not in prepared:
        if "label" in prepared:
            prepared["is_fishing"] = prepared["label"].astype(str).str.lower().eq("fishing")
        elif "event_type" in prepared:
            prepared["is_fishing"] = prepared["event_type"].astype(str).str.lower().str.contains(
                "fishing"
            )
        else:
            prepared["is_fishing"] = True

    if "segment_id" in prepared:
        prepared["segment_id"] = prepared["segment_id"].astype(str)
        return prepared

    required = ["mmsi", "start_timestamp", "end_timestamp"]
    missing_columns = [column for column in required if column not in prepared]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            "Fishing labels must contain either segment_id or mmsi/start_timestamp/"
            f"end_timestamp. Missing: {missing}"
        )

    prepared["mmsi"] = pd.to_numeric(prepared["mmsi"], errors="coerce")
    prepared["start_timestamp"] = pd.to_datetime(
        prepared["start_timestamp"],
        errors="coerce",
        utc=True,
    )
    prepared["end_timestamp"] = pd.to_datetime(
        prepared["end_timestamp"],
        errors="coerce",
        utc=True,
    )
    prepared = prepared.dropna(subset=["mmsi", "start_timestamp", "end_timestamp"])
    prepared["mmsi"] = prepared["mmsi"].astype("int64")
    return prepared


def _empty_fishing_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": pd.Series(dtype="object"),
            "is_fishing": pd.Series(dtype="bool"),
        }
    )


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()
