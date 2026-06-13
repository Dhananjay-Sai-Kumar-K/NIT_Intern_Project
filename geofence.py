from __future__ import annotations
from pathlib import Path
from typing import Mapping
import geopandas as gpd
import pandas as pd

'''
EEZ Geofencing / Boundary Detection
Purpose: detect when vessels enter or exit Exclusive Economic Zones.
Methods used:
---GeoPandas shapefile loading
---Marine Regions EEZ polygon processing

---CRS normalization to WGS84:
EPSG:4326

---AIS point-to-geometry conversion
---Spatial indexing
---Spatial join / point-in-polygon test

Algorithm:
---AIS point within EEZ polygon → inside EEZ
---AIS point not within EEZ polygon → outside EEZ

Then for each vessel:
outside → inside = EEZ entry
inside → outside = EEZ exit

Output:
boundary_events.parquet

Important: this does not mean illegal crossing. It is only a factual boundary event.
'''


AIS_COLUMNS = ["MMSI", "timestamp", "latitude", "longitude"]
EVENT_COLUMNS = ["event_type", "event_timestamp", "mmsi"]
CRS_WGS84 = "EPSG:4326"


def load_eez_polygons(eez_path: str | Path) -> gpd.GeoDataFrame:
    """Load Marine Regions EEZ polygons and prepare their spatial index."""
    path = Path(eez_path)
    if not path.exists():
        raise FileNotFoundError(f"EEZ shapefile not found: {path}")

    eez = gpd.read_file(path)
    if eez.empty:
        raise ValueError(f"EEZ file contains no polygons: {path}")

    if eez.crs is None:
        eez = eez.set_crs(CRS_WGS84)
    else:
        eez = eez.to_crs(CRS_WGS84)

    eez = eez.loc[eez.geometry.notna() & ~eez.geometry.is_empty].copy()
    if eez.empty:
        raise ValueError(f"EEZ file contains no valid geometries: {path}")

    polygon_mask = eez.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not polygon_mask.any():
        geometry_types = ", ".join(sorted(eez.geometry.geom_type.unique()))
        raise ValueError(
            "EEZ file must contain Polygon or MultiPolygon geometries for point-in-EEZ "
            f"classification. Found: {geometry_types}. Use eez_v12.shp, not "
            "eez_boundaries_v12.shp."
        )
    eez = eez.loc[polygon_mask].copy()
    eez = eez.explode(index_parts=False).reset_index(drop=True)
    _ = eez.sindex
    return eez


def ais_points_to_geodataframe(ais_points: pd.DataFrame) -> gpd.GeoDataFrame:
    missing_columns = [column for column in AIS_COLUMNS if column not in ais_points]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required AIS point columns: {missing}")

    points = ais_points.copy()
    points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce", utc=True)
    for column in ["MMSI", "latitude", "longitude"]:
        points[column] = pd.to_numeric(points[column], errors="coerce")

    points = points.dropna(subset=["MMSI", "timestamp", "latitude", "longitude"])
    points["MMSI"] = points["MMSI"].astype("int64")
    points = points.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

    geometry = gpd.points_from_xy(points["longitude"], points["latitude"], crs=CRS_WGS84)
    return gpd.GeoDataFrame(points, geometry=geometry, crs=CRS_WGS84)


def classify_points_by_eez(
    ais_points: gpd.GeoDataFrame,
    eez_polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Annotate AIS points with inside/outside EEZ flags using a spatial join."""
    if ais_points.empty:
        classified = ais_points.copy()
        classified["inside_eez"] = pd.Series(dtype="bool")
        classified["outside_eez"] = pd.Series(dtype="bool")
        return classified

    if ais_points.crs is None:
        ais_points = ais_points.set_crs(CRS_WGS84)
    if eez_polygons.crs is None:
        eez_polygons = eez_polygons.set_crs(CRS_WGS84)

    points = ais_points.to_crs(CRS_WGS84).copy()
    eez = eez_polygons.to_crs(CRS_WGS84)
    _ = eez.sindex

    points["_point_id"] = range(len(points))
    joined = gpd.sjoin(
        points[["_point_id", "geometry"]],
        eez[["geometry"]],
        how="left",
        predicate="within",
    )
    inside_ids = joined.loc[joined["index_right"].notna(), "_point_id"].unique()

    points["inside_eez"] = points["_point_id"].isin(inside_ids)
    points["outside_eez"] = ~points["inside_eez"]
    return points.drop(columns=["_point_id"])


def detect_boundary_events(classified_points: pd.DataFrame) -> pd.DataFrame:
    events, _ = detect_boundary_events_with_state(classified_points)
    return events


def detect_boundary_events_with_state(
    classified_points: pd.DataFrame,
    previous_states: Mapping[int, bool] | None = None,
) -> tuple[pd.DataFrame, dict[int, bool]]:
    """Detect EEZ entry/exit transitions, optionally carrying state across batches."""
    if "inside_eez" not in classified_points:
        raise ValueError("Missing required classified AIS column: inside_eez")

    previous = dict(previous_states or {})
    event_rows: list[dict[str, object]] = []
    points = classified_points.sort_values(["MMSI", "timestamp"])

    for mmsi, vessel_points in points.groupby("MMSI", sort=True):
        mmsi_int = int(mmsi)
        last_state = previous.get(mmsi_int)
        for point in vessel_points.itertuples(index=False):
            current_state = bool(point.inside_eez)
            if last_state is not None and current_state != last_state:
                event_rows.append(
                    {
                        "event_type": "EEZ entry" if current_state else "EEZ exit",
                        "event_timestamp": point.timestamp,
                        "mmsi": mmsi_int,
                    }
                )
            last_state = current_state
        if last_state is not None:
            previous[mmsi_int] = last_state

    if not event_rows:
        return _empty_events_frame(), previous

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    events["mmsi"] = events["mmsi"].astype("int64")
    return events, previous


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": pd.Series(dtype="object"),
            "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "mmsi": pd.Series(dtype="int64"),
        }
    )
