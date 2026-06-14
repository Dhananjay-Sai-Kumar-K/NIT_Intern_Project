"""
Temporary fix script: re-runs segmentation filtering out AIS base stations (MMSI 992xxxxxx)
and then rebuilds features with the corrected segments.
"""
from __future__ import annotations
import pandas as pd
from segmenter import segment_trajectories, write_segments
from feature_engineering import build_features
from pathlib import Path


def main() -> None:
    print("=== STEP 1: Re-run Segmentation (vessel data only) ===")
    cleaned = pd.read_parquet("data/processed/aisdk-2024-03-01_cleaned_ais.parquet")
    n_total = len(cleaned)
    n_mmsis = cleaned["MMSI"].nunique()
    print(f"Total cleaned AIS rows: {n_total}, MMSIs: {n_mmsis}")

    # Filter out AIS base stations (MMSI starting with 992)
    is_base_station = cleaned["MMSI"].astype(str).str.startswith("992")
    vessel_data = cleaned[~is_base_station].copy()
    n_vessel = len(vessel_data)
    n_vessel_mmsis = vessel_data["MMSI"].nunique()
    print(f"After removing {is_base_station.sum()} base-station rows: {n_vessel} rows, {n_vessel_mmsis} MMSIs")

    print("Running segmentation (30 & 60 min windows, min 10 points)...")
    segments = segment_trajectories(vessel_data, window_minutes=[30, 60], min_points=10)
    n_segments = segments["segment_id"].nunique()
    print(f"Segments created: {n_segments} unique segment IDs, {len(segments)} rows")

    speed_valid = segments["speed"].count()
    print(f"Speed non-null in segments: {speed_valid} / {len(segments)}")

    output_path = Path("data/processed/segments.parquet")
    write_segments(segments, output_path)
    print(f"Wrote: {output_path}")

    print()
    print("=== STEP 2: Re-run Feature Engineering ===")
    features = build_features(
        segments_path="data/processed/segments.parquet",
        output_path="data/processed/aisdk-2024-03-01_features.parquet",
        boundary_events_path="data/processed/aisdk-2024-03-01_boundary_events.parquet",
        event_sequence_path="data/processed/aisdk-2024-03-01_event_sequence.json",
        eez_path="data/raw/eez/eez.shp",
    )
    print(f"Features shape: {features.shape}")
    print()
    print("avg_speed stats:")
    print(features["avg_speed"].describe())
    print()
    print("Feature rule-hit counts:")
    print(f"  avg_speed >= 5 knots:          {(features['avg_speed'] >= 5).sum()}")
    print(f"  anchor_duration >= 20 min:     {(features['anchor_duration'] >= 20).sum()}")
    print(f"  loiter_duration >= 20 min:     {(features['loiter_duration'] >= 20).sum()}")
    print(f"  crossing_count >= 2:           {(features['crossing_count'] >= 2).sum()}")
    print(f"  distance_to_boundary <= 5 km:  {(features['distance_to_boundary'] <= 5).sum()}")


if __name__ == "__main__":
    main()
