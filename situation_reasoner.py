from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yaml

DEFAULT_EVENT_SEQUENCE_PATH = Path("data/processed/event_sequence.json")
DEFAULT_SITUATIONS_PATH = Path("data/processed/reasoned_situations.json")


@dataclass(frozen=True)
class ReasoningThresholds:
    transit_max_trip_hours: float = 12.0
    prolonged_presence_min_loiter_events: int = 1
    anchorage_min_duration_minutes: float = 60.0
    repeated_crossing_base_confidence: float = 0.9
    boundary_crossing_confidence: float = 0.75
    transit_confidence: float = 0.7
    anchorage_confidence: float = 0.9
    prolonged_presence_confidence: float = 0.85
    maneuvering_confidence: float = 0.65


class SituationRule(Protocol):
    name: str

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        ...


def build_reasoned_situations(
    event_sequence_path: str | Path,
    output_path: str | Path,
    thresholds: ReasoningThresholds | None = None,
) -> list[dict[str, Any]]:
    events = load_event_sequence(event_sequence_path)
    situations = reason_over_events(events, thresholds or ReasoningThresholds())
    write_reasoned_situations(situations, output_path)
    return situations


def load_event_sequence(event_sequence_path: str | Path) -> pd.DataFrame:
    path = Path(event_sequence_path)
    if not path.exists():
        raise FileNotFoundError(f"Event sequence file not found: {path}")

    with path.open("r", encoding="utf-8") as event_file:
        payload = json.load(event_file)
    events = pd.DataFrame(payload.get("events", []))
    if events.empty:
        return _empty_events()

    required_columns = ["event_type", "event_timestamp", "mmsi"]
    missing_columns = [column for column in required_columns if column not in events]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required event sequence columns: {missing}")
    return _prepare_events(events)


def reason_over_events(
    events: pd.DataFrame,
    thresholds: ReasoningThresholds | None = None,
    rules: list[SituationRule] | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or ReasoningThresholds()
    rules = rules or default_rules()
    
    # Avoid double mutation checks by performing safety checks once
    if events.empty:
        return []

    situations: list[dict[str, Any]] = []
    for _, vessel_events in events.groupby("mmsi", sort=True):
        vessel_events = vessel_events.sort_values("event_timestamp").reset_index(drop=True)
        for rule in rules:
            situations.extend(rule.match(vessel_events, thresholds))

    situations = _deduplicate_situations(situations)
    return sorted(
        situations,
        key=lambda item: (
            item["start_timestamp"] if item["start_timestamp"] else "",
            item["mmsi"] if item["mmsi"] else 0,
            item["situation"],
        ),
    )


def write_reasoned_situations(situations: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "situation_count": len(situations),
        "situations": situations,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def default_rules() -> list[SituationRule]:
    return [
        AnchorageRule(),
        ProlongedBoundaryPresenceRule(),
        RepeatedCrossingRule(),
        BoundaryCrossingRule(),
        ManeuveringRule(),
        TransitRule(),
    ]


class AnchorageRule:
    name = "Anchorage"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        anchor_events = vessel_events.loc[vessel_events["event_type"] == "ANCHOR"]
        for event in anchor_events.itertuples(index=False):
            if event.duration_minutes < thresholds.anchorage_min_duration_minutes:
                continue
            rows.append(
                _situation(
                    situation="Anchorage",
                    confidence=thresholds.anchorage_confidence,
                    events=[event],
                    explanation="ANCHOR event exceeded the configured duration threshold.",
                )
            )
        return rows


class ProlongedBoundaryPresenceRule:
    name = "Prolonged Boundary Presence"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        event_types = vessel_events["event_type"].tolist()
        for entry_index, event_type in enumerate(event_types):
            if event_type != "ENTRY":
                continue
            
            # Bound context windows to avoid double matching down-sequence exits
            exit_index = _next_index(event_types, "EXIT", entry_index + 1)
            next_entry_index = _next_index(event_types, "ENTRY", entry_index + 1)
            
            search_end = exit_index if exit_index is not None else len(vessel_events)
            if next_entry_index is not None and next_entry_index < search_end:
                search_end = next_entry_index

            window = vessel_events.iloc[entry_index:search_end]
            loiter_events = window.loc[window["event_type"] == "LOITER"]
            if len(loiter_events) < thresholds.prolonged_presence_min_loiter_events:
                continue
                
            support = [
                vessel_events.iloc[entry_index],
                *[row for row in loiter_events.itertuples(index=False)],
            ]
            if exit_index is not None and (
                next_entry_index is None or exit_index < next_entry_index
            ):
                support.append(vessel_events.iloc[exit_index])
                
            rows.append(
                _situation(
                    situation="Prolonged Boundary Presence",
                    confidence=thresholds.prolonged_presence_confidence,
                    events=support,
                    explanation="ENTRY followed by LOITER inside the tracked boundary space.",
                )
            )
        return rows


class RepeatedCrossingRule:
    name = "Repeated Crossing"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        repeated = vessel_events.loc[vessel_events["event_type"] == "REPEATED_CROSSING"]
        for event in repeated.itertuples(index=False):
            rows.append(
                _situation(
                    situation="Repeated Crossing",
                    confidence=thresholds.repeated_crossing_base_confidence,
                    events=[event],
                    explanation="REPEATED_CROSSING event was detected by the event engine.",
                )
            )
        return rows


class BoundaryCrossingRule:
    name = "Boundary Crossing"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        event_types = vessel_events["event_type"].tolist()
        for entry_index, event_type in enumerate(event_types):
            if event_type != "ENTRY":
                continue
            exit_index = _next_index(event_types, "EXIT", entry_index + 1)
            support = [vessel_events.iloc[entry_index]]
            if exit_index is not None:
                support.append(vessel_events.iloc[exit_index])
            rows.append(
                _situation(
                    situation="Boundary Crossing",
                    confidence=thresholds.boundary_crossing_confidence,
                    events=support,
                    explanation="EEZ ENTRY event observed, optionally followed by EXIT.",
                )
            )
        return rows


class ManeuveringRule:
    name = "Maneuvering"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        maneuvering = vessel_events.loc[vessel_events["event_type"] == "MANEUVERING"]
        for event in maneuvering.itertuples(index=False):
            rows.append(
                _situation(
                    situation="Maneuvering",
                    confidence=thresholds.maneuvering_confidence,
                    events=[event],
                    explanation="MANEUVERING event indicated high heading circular variance.",
                )
            )
        return rows


class TransitRule:
    name = "Transit"

    def match(
        self,
        vessel_events: pd.DataFrame,
        thresholds: ReasoningThresholds,
    ) -> list[dict[str, Any]]:
        rows = []
        event_types = vessel_events["event_type"].tolist()
        for entry_index, event_type in enumerate(event_types):
            if event_type != "ENTRY":
                continue
            exit_index = _next_index(event_types, "EXIT", entry_index + 1)
            if exit_index is None:
                continue
            window = vessel_events.iloc[entry_index : exit_index + 1]
            if any(window["event_type"].isin(["ANCHOR", "LOITER", "REPEATED_CROSSING"])):
                continue
            
            # Explicitly force datetime conversions to protect metrics from Series drops
            start_t = pd.to_datetime(window["event_timestamp"].iloc[0], utc=True)
            end_t = pd.to_datetime(window["event_timestamp"].iloc[-1], utc=True)
            
            elapsed_hours = (end_t - start_t).total_seconds() / 3600.0
            if elapsed_hours > thresholds.transit_max_trip_hours:
                continue
            rows.append(
                _situation(
                    situation="Transit",
                    confidence=thresholds.transit_confidence,
                    events=[vessel_events.iloc[entry_index], vessel_events.iloc[exit_index]],
                    explanation="ENTRY followed by EXIT without dwell or repeated crossing events.",
                )
            )
        return rows


def _situation(
    situation: str,
    confidence: float,
    events: list[Any],
    explanation: str,
) -> dict[str, Any]:
    support = [_event_to_dict(event) for event in events]
    return {
        "situation": situation,
        "confidence": confidence,
        "mmsi": support[0]["mmsi"] if support else None,
        "start_timestamp": support[0]["event_timestamp"] if support else None,
        "end_timestamp": support[-1]["event_timestamp"] if support else None,
        "supporting_events": support,
        "explanation": explanation,
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, pd.Series):
        record = event.to_dict()
    elif hasattr(event, "_asdict"):
        record = event._asdict()
    else:
        record = dict(event)

    return {
        "event_type": record["event_type"],
        # Guarantee explicit timezone awareness during formatting string builds
        "event_timestamp": _to_iso(record["event_timestamp"]),
        "mmsi": int(record["mmsi"]),
        "start_timestamp": _to_iso(record.get("start_timestamp")),
        "end_timestamp": _to_iso(record.get("end_timestamp")),
        "duration_minutes": float(record.get("duration_minutes", 0.0)),
        "details": record.get("details", {}),
    }


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_events()

    prepared = events.copy()
    prepared["event_timestamp"] = pd.to_datetime(
        prepared["event_timestamp"],
        errors="coerce",
        utc=True,
    )
    if "start_timestamp" not in prepared:
        prepared["start_timestamp"] = prepared["event_timestamp"]
    if "end_timestamp" not in prepared:
        prepared["end_timestamp"] = prepared["event_timestamp"]
    if "duration_minutes" not in prepared:
        prepared["duration_minutes"] = 0.0
    if "details" not in prepared:
        prepared["details"] = [{} for _ in range(len(prepared))]

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
    prepared["mmsi"] = pd.to_numeric(prepared["mmsi"], errors="coerce")
    prepared["duration_minutes"] = pd.to_numeric(
        prepared["duration_minutes"],
        errors="coerce",
    ).fillna(0.0)
    prepared = prepared.dropna(subset=["event_type", "event_timestamp", "mmsi"])
    prepared["event_type"] = prepared["event_type"].astype(str).str.upper()
    prepared["mmsi"] = prepared["mmsi"].astype("int64")
    return prepared.sort_values(["mmsi", "event_timestamp", "event_type"]).reset_index(
        drop=True
    )


def _deduplicate_situations(situations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for situation in situations:
        key = (
            situation["mmsi"],
            situation["situation"],
            situation["start_timestamp"],
            situation["end_timestamp"],
        )
        previous = best_by_key.get(key)
        if previous is None or situation["confidence"] > previous["confidence"]:
            best_by_key[key] = situation
    return list(best_by_key.values())


def _next_index(values: list[str], target: str, start: int) -> int | None:
    for index in range(start, len(values)):
        if values[index] == target:
            return index
    return None


def _to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    # Explicitly coerce ISO strings to preserve standard formatting profiles (+00:00)
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat()
    return pd.Timestamp(value, tz="UTC").isoformat()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": pd.Series(dtype="object"),
            "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "mmsi": pd.Series(dtype="int64"),
            "start_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "end_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "duration_minutes": pd.Series(dtype="float64"),
            "details": pd.Series(dtype="object"),
        }
    )


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    thresholds = ReasoningThresholds(**config.get("situation_reasoning", {}))
    build_reasoned_situations(
        event_sequence_path=config.get("output", {}).get(
            "event_sequence_path",
            DEFAULT_EVENT_SEQUENCE_PATH,
        ),
        output_path=config.get("output", {}).get("reasoned_situations_path", DEFAULT_SITUATIONS_PATH),
        thresholds=thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map AIS event patterns into maritime situations."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_SEQUENCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SITUATIONS_PATH)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config and Path(args.config).exists():
        config = _load_config(args.config)
        thresholds = ReasoningThresholds(**config.get("situation_reasoning", {}))
        output_path = config.get("output", {}).get("reasoned_situations_path", args.output)
        events_path = config.get("output", {}).get("event_sequence_path", args.events)
    else:
        thresholds = ReasoningThresholds()
        output_path = args.output
        events_path = args.events

    situations = build_reasoned_situations(events_path, output_path, thresholds)
    print(f"Reasoned situations: {len(situations)}")
    print(f"Wrote: {output_path}")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


if __name__ == "__main__":
    制造_ = cli()
