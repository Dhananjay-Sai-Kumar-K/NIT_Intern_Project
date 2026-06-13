from __future__ import annotations
import glob
import json
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import yaml
from preprocessing import (
    CleaningStats,
    clean_ais_dataframe,
    split_trajectories_by_vessel,
    standardize_columns,
)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_csv_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    return sorted(dict.fromkeys(paths))


def load_ais_csv_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"AIS CSV file not found: {csv_path}")
        frames.append(pd.read_csv(csv_path))

    if not frames:
        raise ValueError("No AIS CSV files were provided.")

    return pd.concat(frames, ignore_index=True)


def build_dataset(
    csv_paths: Iterable[str | Path],
    cleaned_output_path: str | Path,
    statistics_output_path: str | Path,
    column_aliases: dict[str, Iterable[str]] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], CleaningStats]:
    
    raw_dataframe = load_ais_csv_files(csv_paths)
    standardized = standardize_columns(raw_dataframe, column_aliases=column_aliases)
    cleaned, stats = clean_ais_dataframe(standardized)
    trajectories = split_trajectories_by_vessel(cleaned)

    write_outputs(cleaned, stats, cleaned_output_path, statistics_output_path)
    return cleaned, trajectories, stats


def write_outputs(
    cleaned: pd.DataFrame,
    stats: CleaningStats,
    cleaned_output_path: str | Path,
    statistics_output_path: str | Path,
) -> None:
    
    parquet_path = Path(cleaned_output_path)
    statistics_path = Path(statistics_output_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_parquet(parquet_path, engine="pyarrow", index=False)
    with statistics_path.open("w", encoding="utf-8") as statistics_file:
        json.dump(stats.to_dict(), statistics_file, indent=2)


def main(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    csv_paths = resolve_csv_paths(config["input"]["ais_csv_paths"])
    build_dataset(
        csv_paths=csv_paths,
        cleaned_output_path=config["output"]["cleaned_ais_path"],
        statistics_output_path=config["output"]["statistics_path"],
        column_aliases=config["schema"].get("column_aliases"),
    )


if __name__ == "__main__":
    main()
