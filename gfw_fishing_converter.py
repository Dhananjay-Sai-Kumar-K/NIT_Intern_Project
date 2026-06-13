from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_GFW_INPUT_PATH = Path("data/raw/gfw/fishing_effort.csv")
DEFAULT_GFW_LABELS_PATH = Path("data/processed/gfw_fishing_labels.parquet")


def convert_gfw_fishing_effort(
    gfw_input_path: str | Path,
    output_path: str | Path,
    min_fishing_hours: float = 0.1,
) -> pd.DataFrame:
    effort = load_gfw_effort(gfw_input_path)
    labels = gfw_effort_to_interval_labels(effort, min_fishing_hours=min_fishing_hours)
    write_gfw_labels(labels, output_path)
    return labels


def load_gfw_effort(gfw_input_path: str | Path) -> pd.DataFrame:
    path = Path(gfw_input_path)
    if not path.exists():
        raise FileNotFoundError(f"GFW fishing effort file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported GFW file type: {path.suffix}")


def gfw_effort_to_interval_labels(
    effort: pd.DataFrame,
    min_fishing_hours: float = 0.1,
) -> pd.DataFrame:
    mmsi_column = _find_column(effort, ["mmsi", "ssvid", "vessel_mmsi"])
    date_column = _find_column(effort, ["date", "day", "timestamp", "date_utc"])
    hours_column = _find_column(
        effort,
        [
            "fishing_hours",
            "apparent_fishing_hours",
            "fishing_hours_sum",
            "hours",
        ],
    )

    labels = effort[[mmsi_column, date_column, hours_column]].copy()
    labels = labels.rename(
        columns={
            mmsi_column: "mmsi",
            date_column: "date",
            hours_column: "fishing_hours",
        }
    )
    labels["mmsi"] = pd.to_numeric(labels["mmsi"], errors="coerce")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce", utc=True)
    labels["fishing_hours"] = pd.to_numeric(labels["fishing_hours"], errors="coerce")
    labels = labels.dropna(subset=["mmsi", "date", "fishing_hours"])
    labels = labels.loc[labels["fishing_hours"] >= min_fishing_hours].copy()
    labels["mmsi"] = labels["mmsi"].astype("int64")
    labels["start_timestamp"] = labels["date"].dt.floor("D")
    labels["end_timestamp"] = labels["start_timestamp"] + pd.Timedelta(days=1)
    labels["is_fishing"] = True
    labels["label_confidence"] = _confidence_from_hours(labels["fishing_hours"])

    return (
        labels[
            [
                "mmsi",
                "start_timestamp",
                "end_timestamp",
                "is_fishing",
                "fishing_hours",
                "label_confidence",
            ]
        ]
        .sort_values(["mmsi", "start_timestamp"])
        .reset_index(drop=True)
    )


def write_gfw_labels(labels: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, engine="pyarrow", index=False)


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    gfw_config = config.get("gfw", {})
    convert_gfw_fishing_effort(
        gfw_input_path=config["input"].get("gfw_fishing_effort_path", DEFAULT_GFW_INPUT_PATH),
        output_path=config["input"].get("gfw_fishing_labels_path", DEFAULT_GFW_LABELS_PATH),
        min_fishing_hours=gfw_config.get("min_fishing_hours", 0.1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Global Fishing Watch effort data to weak supervision labels."
    )
    parser.add_argument("--gfw-effort", type=Path, default=DEFAULT_GFW_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_GFW_LABELS_PATH)
    parser.add_argument("--min-fishing-hours", type=float, default=0.1)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config:
        config = _load_config(args.config)
        gfw_config = config.get("gfw", {})
        labels = convert_gfw_fishing_effort(
            gfw_input_path=config["input"].get(
                "gfw_fishing_effort_path",
                DEFAULT_GFW_INPUT_PATH,
            ),
            output_path=config["input"].get(
                "gfw_fishing_labels_path",
                DEFAULT_GFW_LABELS_PATH,
            ),
            min_fishing_hours=gfw_config.get("min_fishing_hours", 0.1),
        )
        output_path = config["input"].get("gfw_fishing_labels_path", DEFAULT_GFW_LABELS_PATH)
    else:
        labels = convert_gfw_fishing_effort(
            gfw_input_path=args.gfw_effort,
            output_path=args.output,
            min_fishing_hours=args.min_fishing_hours,
        )
        output_path = args.output

    print(f"GFW fishing label rows: {len(labels)}")
    print(f"Wrote: {output_path}")


def _find_column(dataframe: pd.DataFrame, candidates: list[str]) -> str:
    normalized_columns = {_normalize(column): column for column in dataframe.columns}
    for candidate in candidates:
        column = normalized_columns.get(_normalize(candidate))
        if column is not None:
            return column
    raise ValueError(
        "Could not find required GFW column. Expected one of: "
        + ", ".join(candidates)
    )


def _normalize(column: Any) -> str:
    return "".join(character for character in str(column).lower() if character.isalnum())


def _confidence_from_hours(fishing_hours: pd.Series) -> pd.Series:
    confidence = 0.70 + (fishing_hours.clip(lower=0, upper=6) / 6.0) * 0.25
    return confidence.clip(upper=0.95)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()
