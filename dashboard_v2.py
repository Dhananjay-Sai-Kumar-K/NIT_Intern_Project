from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import pandas as pd
import geopandas as gpd
import folium
from folium import FeatureGroup, LayerControl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
CLEANED_AIS_PATH = DATA_DIR / "cleaned_ais.parquet"
SEGMENTS_PATH = DATA_DIR / "aisdk-2024-03-01_segments.parquet"
EVENTS_PATH = DATA_DIR / "aisdk-2024-03-01_event_sequence.json"
PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"
EEZ_SHAPEFILE_PATH = Path("data/raw/eez/World_EEZ_v12_20231025/World_EEZ_v12_20231025/eez_v12.shp")
GFW_FISHING_LABELS_PATH = DATA_DIR / "aisdk-2024-03-01_gfw_fishing_labels.parquet"

OUTPUT_HTML = "interactive_map.html"

# Maximum vessels to pre-embed into the HTML (slider goes up to this)
MAX_EMBED_VESSELS = 200
# Default shown when page opens
DEFAULT_VESSEL_COUNT = 10


# ---------------------------------------------------------------------------
# Situation / event colour maps
# ---------------------------------------------------------------------------
SITUATION_COLORS = {
    "Transit":                    "#28a745",
    "Anchorage":                  "#007bff",
    "Fishing Activity":           "#ffc107",
    "Boundary Crossing":          "#fd7e14",
    "Prolonged Boundary Presence":"#dc3545",
    "Unknown":                    "#6c757d",
}

EVENT_COLORS = {
    "ENTRY":            "#fd7e14",
    "EXIT":             "#6f42c1",
    "ANCHOR":           "#007bff",
    "LOITER":           "#dc3545",
    "MANEUVERING":      "#28a745",
    "REPEATED_CROSSING":"#343a40",
}

VESSEL_PALETTE = [
    "#6f42c1","#e83e8c","#17a2b8","#fd7e14","#20c997",
    "#ffc107","#dc3545","#28a745","#007bff","#6c757d",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dashboard_data() -> tuple[
    pd.DataFrame, pd.DataFrame, list[dict[str, Any]],
    pd.DataFrame | None, gpd.GeoDataFrame | None, pd.DataFrame | None
]:
    print("Loading data for visualization dashboard...")
    cleaned_ais = pd.read_parquet(CLEANED_AIS_PATH)
    segments    = pd.read_parquet(SEGMENTS_PATH)

    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        events_data = json.load(f)
    events: list[dict[str, Any]] = events_data["events"]

    predictions: pd.DataFrame | None = None
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_parquet(PREDICTIONS_PATH)

    eez_gdf: gpd.GeoDataFrame | None = None
    if EEZ_SHAPEFILE_PATH.exists():
        print("Loading EEZ shapefile ...")
        try:
            min_lat = cleaned_ais["latitude"].min()
            max_lat = cleaned_ais["latitude"].max()
            min_lon = cleaned_ais["longitude"].min()
            max_lon = cleaned_ais["longitude"].max()
            eez_gdf = gpd.read_file(
                EEZ_SHAPEFILE_PATH,
                bbox=(min_lon - 0.5, min_lat - 0.5, max_lon + 0.5, max_lat + 0.5),
            )
            if not eez_gdf.empty:
                eez_gdf["geometry"] = eez_gdf["geometry"].simplify(0.005, preserve_topology=True)
                eez_gdf = eez_gdf.to_crs("EPSG:4326")
        except Exception as exc:
            print(f"Warning: Could not load EEZ shapefile: {exc}")

    fishing_labels: pd.DataFrame | None = None
    if GFW_FISHING_LABELS_PATH.exists():
        print("Loading GFW fishing labels ...")
        fishing_labels = pd.read_parquet(GFW_FISHING_LABELS_PATH)
    else:
        print("Warning: GFW fishing labels file not found. Skipping fishing markers.")

    return cleaned_ais, segments, events, predictions, eez_gdf, fishing_labels


# ---------------------------------------------------------------------------
# Build the dashboard
# ---------------------------------------------------------------------------
def build_interactive_dashboard() -> None:
    cleaned_ais, segments, events, predictions, eez_gdf, fishing_labels = load_dashboard_data()

    # ------------------------------------------------------------------
    # Select vessels to embed (sorted by point count descending)
    # ------------------------------------------------------------------
    vessel_counts = cleaned_ais["MMSI"].value_counts()
    embed_mmsis   = list(vessel_counts.index[:MAX_EMBED_VESSELS])
    total_vessels = len(vessel_counts)

    print(f"Total vessels in dataset: {total_vessels}")
    print(f"Pre-embedding top {len(embed_mmsis)} vessels into the map...")

    # ------------------------------------------------------------------
    # Pre-build vessel track data as compact JSON
    # ------------------------------------------------------------------
    vessel_ais = cleaned_ais[cleaned_ais["MMSI"].isin(embed_mmsis)].copy()
    vessel_ais["timestamp"] = pd.to_datetime(vessel_ais["timestamp"], utc=True).astype(str)

    # Downsample each vessel track for rendering (keep every Nth point)
    TRACK_MAX_POINTS = 300
    tracks_data: list[dict[str, Any]] = []
    for rank, mmsi in enumerate(embed_mmsis):
        vdf = vessel_ais[vessel_ais["MMSI"] == mmsi].sort_values("timestamp")
        n   = len(vdf)
        step = max(1, n // TRACK_MAX_POINTS)
        vdf = vdf.iloc[::step]
        coords = list(zip(vdf["latitude"].tolist(), vdf["longitude"].tolist()))
        color  = VESSEL_PALETTE[rank % len(VESSEL_PALETTE)]
        tracks_data.append({
            "mmsi":   mmsi,
            "rank":   rank,                   # 0-based index for the slider
            "color":  color,
            "points": n,
            "coords": coords,
            "start":  vdf["timestamp"].iloc[0]  if len(vdf) > 0 else "",
            "end":    vdf["timestamp"].iloc[-1]  if len(vdf) > 0 else "",
        })

    # ------------------------------------------------------------------
    # Pre-build event data as compact JSON (ALL events, not just top-5)
    # ------------------------------------------------------------------
    mmsi_set = set(embed_mmsis)
    # Build quick lat/lon lookup: mmsi -> sorted list of (ts_epoch, lat, lon)
    print("Building timestamp -> coordinate index for event placement...")
    ais_for_lookup = cleaned_ais[cleaned_ais["MMSI"].isin(mmsi_set)].copy()
    ais_for_lookup["ts_epoch"] = pd.to_datetime(
        ais_for_lookup["timestamp"], utc=True
    ).astype("int64") // 10**9
    ais_idx: dict[int, pd.DataFrame] = {
        mmsi: grp.sort_values("ts_epoch").reset_index(drop=True)
        for mmsi, grp in ais_for_lookup.groupby("MMSI")
    }

    events_data_out: list[dict[str, Any]] = []
    for ev in events:
        mmsi = ev.get("mmsi")
        if mmsi not in mmsi_set:
            continue
        etype = ev.get("event_type", "UNKNOWN")
        start_ts = ev.get("start_timestamp", "")
        try:
            ts_epoch = int(pd.to_datetime(start_ts, utc=True).timestamp())
        except Exception:
            ts_epoch = 0

        # Nearest AIS point
        lat, lon = 0.0, 0.0
        vdf = ais_idx.get(mmsi)
        if vdf is not None and len(vdf):
            idx = (vdf["ts_epoch"] - ts_epoch).abs().idxmin()
            lat = float(vdf.at[idx, "latitude"])
            lon = float(vdf.at[idx, "longitude"])

        events_data_out.append({
            "mmsi":     mmsi,
            "rank":     embed_mmsis.index(mmsi) if mmsi in embed_mmsis else -1,
            "type":     etype,
            "color":    EVENT_COLORS.get(etype, "#6c757d"),
            "lat":      lat,
            "lon":      lon,
            "start":    start_ts,
            "end":      ev.get("end_timestamp", ""),
            "duration": ev.get("duration_minutes", 0.0),
        })

    print(f"Prepared {len(events_data_out)} event markers for {len(embed_mmsis)} vessels.")

    # ------------------------------------------------------------------
    # Pre-build GFW Fishing markers
    # ------------------------------------------------------------------
    fishing_data_out: list[dict[str, Any]] = []
    if fishing_labels is not None and not fishing_labels.empty:
        print("Processing fishing labels for embedded vessels...")
        for _, row in fishing_labels.iterrows():
            mmsi = row.get("mmsi")
            if pd.isna(mmsi) or mmsi not in mmsi_set:
                continue

            start_ts = str(row.get("start_timestamp", ""))
            try:
                ts_epoch = int(pd.to_datetime(start_ts, utc=True).timestamp())
            except Exception:
                continue  # skip if we can't parse the start time

            vdf = ais_idx.get(mmsi)
            if vdf is not None and len(vdf):
                idx = (vdf["ts_epoch"] - ts_epoch).abs().idxmin()
                lat = float(vdf.at[idx, "latitude"])
                lon = float(vdf.at[idx, "longitude"])
                
                f_hours = row.get("fishing_hours")
                l_conf = row.get("label_confidence")

                fishing_data_out.append({
                    "mmsi": int(mmsi),
                    "rank": embed_mmsis.index(mmsi),
                    "type": "Fishing Activity",
                    "color": "#ffc107",
                    "lat": lat,
                    "lon": lon,
                    "start": start_ts,
                    "end": str(row.get("end_timestamp", "")),
                    "fishing_hours": float(f_hours) if pd.notna(f_hours) else None,
                    "confidence": float(l_conf) if pd.notna(l_conf) else None
                })
        print(f"Prepared {len(fishing_data_out)} fishing markers.")

    # ------------------------------------------------------------------
    # Base Folium map (minimal — JS will do the heavy rendering)
    # ------------------------------------------------------------------
    center_lat = vessel_ais["latitude"].mean()
    center_lon = vessel_ais["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="cartodbpositron",
    )

    # EEZ layer (static Folium GeoJson)
    if eez_gdf is not None and not eez_gdf.empty:
        eez_fg = FeatureGroup(name="EEZ Boundaries", show=True)
        folium.GeoJson(
            eez_gdf,
            style_function=lambda _: {
                "fillColor": "#17a2b8",
                "color": "#17a2b8",
                "weight": 1.5,
                "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(fields=["GEONAME"], aliases=["EEZ Area:"]),
        ).add_to(eez_fg)
        eez_fg.add_to(m)

    LayerControl().add_to(m)

    # ------------------------------------------------------------------
    # Inject vessel / event / fishing data + slider UI as raw HTML/JS
    # ------------------------------------------------------------------
    tracks_json = json.dumps(tracks_data,  separators=(",", ":"))
    events_json = json.dumps(events_data_out, separators=(",", ":"))
    fishing_json = json.dumps(fishing_data_out, separators=(",", ":"))
    sit_colors_json = json.dumps(SITUATION_COLORS)
    evt_colors_json = json.dumps(EVENT_COLORS)

    custom_js = f"""
<script>
// ---------------------------------------------------------------
// Embedded data
// ---------------------------------------------------------------
var TRACKS       = {tracks_json};
var EVENTS       = {events_json};
var FISHING      = {fishing_json};
var SIT_COLORS   = {sit_colors_json};
var EVT_COLORS   = {evt_colors_json};
var MAX_VESSELS  = {len(embed_mmsis)};
var DEFAULT_N    = {DEFAULT_VESSEL_COUNT};

// ---------------------------------------------------------------
// Track + Event layer references
// ---------------------------------------------------------------
var trackPolylines = [];   // one per vessel
var eventMarkers   = [];   // one per event
var fishingMarkers = [];   // one per fishing label

// ---------------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------------
function drawTracks(n) {{
    // Remove existing
    trackPolylines.forEach(function(pl){{ if(pl) pl.remove(); }});
    trackPolylines = [];

    for(var i = 0; i < Math.min(n, TRACKS.length); i++) {{
        var t = TRACKS[i];
        if(!t.coords || t.coords.length < 2) continue;
        var pl = L.polyline(t.coords, {{
            color:   t.color,
            weight:  3,
            opacity: 0.85,
        }}).bindPopup(
            "<b>MMSI:</b> " + t.mmsi +
            "<br><b>Rank:</b> #" + (t.rank+1) +
            "<br><b>AIS points:</b> " + t.points +
            "<br><b>Start:</b> " + t.start +
            "<br><b>End:</b> " + t.end
        ).addTo(window._leaflet_map);

        // Start / end markers
        if(t.coords.length > 0) {{
            L.marker(t.coords[0], {{
                icon: L.divIcon({{
                    html:'<div style="background:' + t.color +
                         ';width:12px;height:12px;border-radius:50%;border:2px solid #fff;"></div>',
                    iconSize:[12,12], iconAnchor:[6,6]
                }})
            }}).bindTooltip("Start: MMSI " + t.mmsi).addTo(window._leaflet_map);
            var last = t.coords[t.coords.length-1];
            L.marker(last, {{
                icon: L.divIcon({{
                    html:'<div style="background:#fff;border:3px solid ' + t.color +
                         ';width:12px;height:12px;border-radius:2px;"></div>',
                    iconSize:[12,12], iconAnchor:[6,6]
                }})
            }}).bindTooltip("End: MMSI " + t.mmsi).addTo(window._leaflet_map);
        }}
        trackPolylines.push(pl);
    }}
}}

function drawEvents(n) {{
    eventMarkers.forEach(function(m){{ if(m) m.remove(); }});
    eventMarkers = [];

    for(var i = 0; i < EVENTS.length; i++) {{
        var ev = EVENTS[i];
        if(ev.rank >= n) continue;   // only show events for displayed vessels
        if(!ev.lat && !ev.lon) continue;

        var dot = L.circleMarker([ev.lat, ev.lon], {{
            radius:      7,
            color:       "#fff",
            weight:      1.5,
            fillColor:   ev.color,
            fillOpacity: 0.85,
        }}).bindPopup(
            "<b>" + ev.type + "</b><br>" +
            "MMSI: " + ev.mmsi + "<br>" +
            "Start: " + ev.start + "<br>" +
            "End: " + ev.end + "<br>" +
            "Duration: " + (typeof ev.duration === 'number' ? ev.duration.toFixed(1) : ev.duration) + " min"
        ).addTo(window._leaflet_map);
        eventMarkers.push(dot);
    }}
}}

function drawFishing(n) {{
    fishingMarkers.forEach(function(m){{ if(m) m.remove(); }});
    fishingMarkers = [];

    for(var i = 0; i < FISHING.length; i++) {{
        var f = FISHING[i];
        if(f.rank >= n) continue;
        if(!f.lat && !f.lon) continue;

        var popupHtml = "<b>" + f.type + "</b><br>" +
                        "MMSI: " + f.mmsi + "<br>" +
                        "Start: " + f.start + "<br>" +
                        "End: " + f.end;
        if(f.fishing_hours !== null) {{
            popupHtml += "<br>Fishing Hours: " + f.fishing_hours.toFixed(2);
        }}
        if(f.confidence !== null) {{
            popupHtml += "<br>Confidence: " + (f.confidence * 100).toFixed(1) + "%";
        }}

        var dot = L.circleMarker([f.lat, f.lon], {{
            radius:      8,
            color:       "#000",
            weight:      1,
            fillColor:   f.color,
            fillOpacity: 0.9,
        }}).bindPopup(popupHtml).addTo(window._leaflet_map);
        fishingMarkers.push(dot);
    }}
}}

function refreshMap(n) {{
    drawTracks(n);
    drawEvents(n);
    drawFishing(n);
    document.getElementById('vessel-count-label').textContent = n + ' / {len(embed_mmsis)} vessels';
    document.getElementById('event-count-label').textContent =
        eventMarkers.length + ' events shown | ' + fishingMarkers.length + ' fishing areas';
}}

// ---------------------------------------------------------------
// Wait for Leaflet map to be ready, then initialise
// ---------------------------------------------------------------
function waitForMap() {{
    var maps = document.querySelectorAll('.folium-map');
    if(maps.length === 0) {{ setTimeout(waitForMap, 200); return; }}
    var mapId = maps[0].id;
    if(!window[mapId]) {{ setTimeout(waitForMap, 200); return; }}
    window._leaflet_map = window[mapId];
    refreshMap(DEFAULT_N);
}}
document.addEventListener('DOMContentLoaded', waitForMap);
</script>

<div id="vessel-control-panel" style="
    position: fixed;
    top: 80px; right: 15px;
    z-index: 9999;
    background: rgba(255,255,255,0.97);
    border: 2px solid #dee2e6;
    border-radius: 10px;
    padding: 14px 16px;
    width: 260px;
    font-family: 'Segoe UI', sans-serif;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
">
  <div style="font-weight:700;font-size:14px;margin-bottom:10px;color:#343a40;">
    🚢 Vessel Display Control
  </div>

  <label style="font-size:12px;color:#495057;">Number of vessels shown:</label>
  <input type="range" id="vessel-slider"
         min="1" max="{len(embed_mmsis)}" value="{DEFAULT_VESSEL_COUNT}"
         style="width:100%;margin:6px 0;"
         oninput="refreshMap(parseInt(this.value))">

  <div style="display:flex;justify-content:space-between;font-size:12px;color:#495057;margin-bottom:8px;">
    <span>1</span>
    <span id="vessel-count-label" style="font-weight:600;color:#007bff;">{DEFAULT_VESSEL_COUNT} / {len(embed_mmsis)} vessels</span>
    <span>{len(embed_mmsis)}</span>
  </div>

  <div style="font-size:11px;color:#6c757d;margin-bottom:6px;"
       id="event-count-label">0 events shown | 0 fishing areas</div>

  <hr style="margin:8px 0;border-color:#dee2e6;">

  <div style="font-size:11px;color:#495057;margin-bottom:6px;font-weight:600;">Quick Presets:</div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    <button onclick="document.getElementById('vessel-slider').value=5;  refreshMap(5)"
            style="{_btn_style()}">5</button>
    <button onclick="document.getElementById('vessel-slider').value=10; refreshMap(10)"
            style="{_btn_style()}">10</button>
    <button onclick="document.getElementById('vessel-slider').value=25; refreshMap(25)"
            style="{_btn_style()}">25</button>
    <button onclick="document.getElementById('vessel-slider').value=50; refreshMap(50)"
            style="{_btn_style()}">50</button>
    <button onclick="document.getElementById('vessel-slider').value={len(embed_mmsis)}; refreshMap({len(embed_mmsis)})"
            style="{_btn_style(accent=True)}">All {len(embed_mmsis)}</button>
  </div>

  <hr style="margin:8px 0;border-color:#dee2e6;">

  <div style="font-size:11px;color:#495057;font-weight:600;margin-bottom:4px;">Event Types:</div>
  <div style="font-size:10.5px;line-height:1.8;">
    <span>&#9679;</span><span style="color:{EVENT_COLORS['ENTRY']};font-weight:600;">&#9679;</span> ENTRY &nbsp;
    <span style="color:{EVENT_COLORS['EXIT']};font-weight:600;">&#9679;</span> EXIT<br>
    <span style="color:{EVENT_COLORS['ANCHOR']};font-weight:600;">&#9679;</span> ANCHOR &nbsp;
    <span style="color:{EVENT_COLORS['LOITER']};font-weight:600;">&#9679;</span> LOITER<br>
    <span style="color:{EVENT_COLORS['MANEUVERING']};font-weight:600;">&#9679;</span> MANEUVERING<br>
    <span style="color:{EVENT_COLORS['REPEATED_CROSSING']};font-weight:600;">&#9679;</span> REPEATED CROSSING<br>
    <span style="color:#ffc107;font-weight:600;">&#9679;</span> Fishing Activity
  </div>

  <hr style="margin:8px 0;border-color:#dee2e6;">

  <div style="font-size:11px;color:#495057;font-weight:600;margin-bottom:4px;">Situation Labels:</div>
  <div style="font-size:10.5px;line-height:1.8;">
    <span style="color:{SITUATION_COLORS['Transit']};font-weight:600;">&#9679;</span> Transit<br>
    <span style="color:{SITUATION_COLORS['Anchorage']};font-weight:600;">&#9679;</span> Anchorage<br>
    <span style="color:{SITUATION_COLORS['Fishing Activity']};font-weight:600;">&#9679;</span> Fishing Activity<br>
    <span style="color:{SITUATION_COLORS['Boundary Crossing']};font-weight:600;">&#9679;</span> Boundary Crossing<br>
    <span style="color:{SITUATION_COLORS['Prolonged Boundary Presence']};font-weight:600;">&#9679;</span> Prolonged Boundary Presence
  </div>

  <div style="margin-top:8px;font-size:10px;color:#adb5bd;">
    Total vessels in dataset: {total_vessels:,}<br>
    Total events: {len(events_data_out):,} | Fishing: {len(fishing_data_out):,}
  </div>
</div>
"""

    m.get_root().html.add_child(folium.Element(custom_js))
    m.save(OUTPUT_HTML)
    print(f"Exported interactive map to {OUTPUT_HTML}")
    print(f"  Embedded {len(embed_mmsis)} vessels (slider: 1–{len(embed_mmsis)})")
    print(f"  Embedded {len(events_data_out)} events and {len(fishing_data_out)} fishing markers")
    print(f"  Default view: top {DEFAULT_VESSEL_COUNT} vessels")


def _btn_style(accent: bool = False) -> str:
    bg = "#007bff" if accent else "#f8f9fa"
    fg = "#fff"    if accent else "#495057"
    return (
        f"background:{bg};color:{fg};border:1px solid #dee2e6;"
        "border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;"
    )


if __name__ == "__main__":
    build_interactive_dashboard()