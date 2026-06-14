from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

'''
Phase 1: Preporcessings implemented

Purpose: convert raw AIS CSV into clean trajectory data.
Methods used:
---Column standardization
Maps different AIS column names like MMSI, BaseDateTime, LAT, LON, SOG, COG into one schema.

---Timestamp parsing
Converts raw timestamp strings into UTC datetime format.

---Coordinate validation
Removes invalid points where:

latitude not in [-90, 90]
longitude not in [-180, 180]

---Duplicate removal
Removes duplicate AIS records using:
MMSI + timestamp

---Trajectory sorting
Sorts all AIS points by:
MMSI, timestamp

---Vessel trajectory grouping
Creates one trajectory per vessel using MMSI.

---Output:
cleaned_ais.parquet
dataset_statistics.json

Next:
Segmenter.py
'''

STANDARD_COLUMNS = ["MMSI", "timestamp", "latitude", "longitude", "speed", "heading"]


@dataclass(frozen=True)
class CleaningStats:
    raw_rows: int
    cleaned_rows: int
    invalid_coordinate_rows: int
    duplicate_timestamp_rows: int
    vessel_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "raw_rows": self.raw_rows,
            "cleaned_rows": self.cleaned_rows,
            "invalid_coordinate_rows": self.invalid_coordinate_rows,
            "duplicate_timestamp_rows": self.duplicate_timestamp_rows,
            "vessel_count": self.vessel_count,
        }


def standardize_columns(
    dataframe: pd.DataFrame,
    column_aliases: dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    
    """Rename common AIS column variants to the project's standard schema."""
    aliases = column_aliases or {}
    normalized_lookup = {
        _normalize_column_name(column): column for column in dataframe.columns
    }


    rename_map: dict[str, str] = {}
    for standard_column in STANDARD_COLUMNS:
        candidates = [standard_column, *aliases.get(standard_column, [])]
        for candidate in candidates:
            source_column = normalized_lookup.get(_normalize_column_name(candidate))
            if source_column is not None:
                rename_map[source_column] = standard_column
                break

    standardized = dataframe.rename(columns=rename_map).copy()
    missing_columns = [column for column in STANDARD_COLUMNS if column not in standardized]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required AIS columns: {missing}")

    return standardized[STANDARD_COLUMNS]


def clean_ais_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Validate, deduplicate, and sort AIS records."""
    raw_rows = len(dataframe)
    cleaned = dataframe.copy()

    # Use dayfirst=True to correctly parse European date formats (e.g. DD/MM/YYYY)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True, dayfirst=True)
    for column in ["MMSI", "latitude", "longitude", "speed", "heading"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["MMSI", "timestamp", "latitude", "longitude"])
    cleaned["MMSI"] = cleaned["MMSI"].astype("int64")
    cleaned["trajectory_date"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d")

    valid_coordinates = cleaned["latitude"].between(-90, 90) & cleaned["longitude"].between(
        -180, 180
    )
    invalid_coordinate_rows = int((~valid_coordinates).sum())
    cleaned = cleaned.loc[valid_coordinates].copy()

    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["MMSI", "timestamp"], keep="first")
    duplicate_timestamp_rows = before_deduplication - len(cleaned)

    # Filter out AIS shore-based base stations and repeaters.
    # These have MMSI values in the 992xxxxxx range and do not move,
    # so their speed is always NaN/0, which corrupts the feature pipeline.
    is_base_station = (cleaned["MMSI"] >= 992_000_000) & (cleaned["MMSI"] <= 992_999_999)
    cleaned = cleaned.loc[~is_base_station].copy()

    cleaned = cleaned.sort_values(["MMSI", "trajectory_date", "timestamp"]).reset_index(drop=True)

    stats = CleaningStats(
        raw_rows=raw_rows,
        cleaned_rows=len(cleaned),
        invalid_coordinate_rows=invalid_coordinate_rows,
        duplicate_timestamp_rows=duplicate_timestamp_rows,
        vessel_count=int(cleaned["MMSI"].nunique()),
    )
    return cleaned, stats


def split_trajectories_by_vessel(dataframe: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Create one sorted dataframe per vessel trajectory."""
    trajectories: dict[int, pd.DataFrame] = {}
    for mmsi, vessel_frame in dataframe.groupby("MMSI", sort=True):
        trajectories[int(mmsi)] = vessel_frame.reset_index(drop=True).copy()
    return trajectories


def _normalize_column_name(column: str) -> str:
    return "".join(character for character in str(column).lower() if character.isalnum())
