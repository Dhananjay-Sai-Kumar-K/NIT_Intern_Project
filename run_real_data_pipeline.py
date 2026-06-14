from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from data_loader import build_dataset, build_dataset_chunked
from segmenter import build_segments, build_segments_by_date


DEFAULT_RAW_DIR = Path("data/raw/ais")
DEFAULT_CLEANED_PATH = Path("data/processed/cleaned_ais.parquet")
DEFAULT_STATS_PATH = Path("data/processed/dataset_statistics.json")
DEFAULT_SEGMENTS_PATH = Path("data/processed/segments.parquet")
DEFAULT_WINDOWS = [30, 60]
DEFAULT_MIN_POINTS = 10


def main() -> None:
    args = parse_args()

    staged_csv_paths = stage_local_ais_files(
        source_path=args.source,
        raw_dir=args.raw_dir,
        max_rows=args.max_rows,
        replace=args.replace_raw,
    )

    print(f"Staged {len(staged_csv_paths)} CSV file(s):")
    for csv_path in staged_csv_paths:
        print(f"  - {csv_path}")

    # FIX: Dynamically separate files if a single file source is specified
    cleaned_out = args.cleaned_output
    segments_out = args.segments_output
    stats_out = args.stats_output

    if args.source.is_file():
        suffix_stem = args.source.stem
        cleaned_out = cleaned_out.parent / f"{suffix_stem}_{cleaned_out.name}"
        segments_out = segments_out.parent / f"{suffix_stem}_{segments_out.name}"
        stats_out = stats_out.parent / f"{suffix_stem}_{stats_out.name}"

    if len(staged_csv_paths) > 1:
        stats = build_dataset_chunked(
            csv_paths=staged_csv_paths,
            cleaned_output_path=cleaned_out,
            statistics_output_path=stats_out,
            column_aliases=_real_ais_column_aliases(),
            chunksize=args.chunksize,
        )
        print(f"Cleaned AIS rows: {stats.cleaned_rows}")
        print(f"Vessel trajectories: {stats.vessel_count}")
    else:
        cleaned, trajectories, stats = build_dataset(
            csv_paths=staged_csv_paths,
            cleaned_output_path=cleaned_out,
            statistics_output_path=stats_out,
            column_aliases=_real_ais_column_aliases(),
        )
        print(f"Cleaned AIS rows: {len(cleaned)}")
        print(f"Vessel trajectories: {len(trajectories)}")
    print(f"Wrote: {cleaned_out}")

    if len(staged_csv_paths) > 1:
        segment_count = build_segments_by_date(
            cleaned_ais_path=cleaned_out,
            segments_output_path=segments_out,
            window_minutes=args.window_minutes,
            min_points=args.min_points,
        )
    else:
        segments = build_segments(
            cleaned_ais_path=cleaned_out,
            segments_output_path=segments_out,
            window_minutes=args.window_minutes,
            min_points=args.min_points,
        )
        segment_count = segments["segment_id"].nunique() if not segments.empty else 0
    print(f"Segments: {segment_count}")
    print(f"Wrote: {segments_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a locally downloaded AIS file and run cleaning + segmentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a downloaded AIS .zip, .csv, or directory containing CSV/ZIP files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw CSV files will be staged.",
    )
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=DEFAULT_CLEANED_PATH,
        help="Output path for cleaned AIS Parquet.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Output path for dataset statistics JSON.",
    )
    parser.add_argument(
        "--segments-output",
        type=Path,
        default=DEFAULT_SEGMENTS_PATH,
        help="Output path for segmented AIS Parquet.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="Segmentation window sizes in minutes.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=DEFAULT_MIN_POINTS,
        help="Drop segments with fewer than this many AIS points.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-CSV row limit for a quick first run on large AIS downloads.",
    )
    parser.add_argument(
        "--replace-raw",
        action="store_true",
        help="Delete existing staged CSV files in the raw directory before staging.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="CSV rows per chunk for memory-safe multi-file ingestion.",
    )
    return parser.parse_args()


def stage_local_ais_files(
    source_path: Path,
    raw_dir: Path,
    max_rows: int | None = None,
    replace: bool = False,
) -> list[Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        for old_csv in raw_dir.glob("*.csv"):
            old_csv.unlink()

    staged_csv_paths: list[Path] = []
    for file_path in _iter_source_files(source_path):
        if file_path.suffix.lower() == ".zip":
            staged_csv_paths.extend(_stage_zip(file_path, raw_dir, max_rows))
        elif file_path.suffix.lower() == ".csv":
            staged_csv_paths.append(_stage_csv(file_path, raw_dir, max_rows))

    if not staged_csv_paths:
        raise ValueError(f"No CSV or ZIP AIS files found in: {source_path}")

    return sorted(staged_csv_paths)


def _iter_source_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]
    return sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".zip"}
    )


def _stage_zip(zip_path: Path, raw_dir: Path, max_rows: int | None) -> list[Path]:
    staged_paths: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith(".csv") and not member.endswith("/")
        ]
        if not csv_members:
            raise ValueError(f"No CSV files found inside ZIP: {zip_path}")

        for member in csv_members:
            output_name = f"{zip_path.stem}_{Path(member).name}"
            output_path = raw_dir / output_name
            with archive.open(member) as source_file:
                if max_rows is None:
                    with output_path.open("wb") as output_file:
                        shutil.copyfileobj(source_file, output_file)
                else:
                    dataframe = pd.read_csv(source_file, nrows=max_rows)
                    dataframe.to_csv(output_path, index=False)
            staged_paths.append(output_path)
    return staged_paths


def _stage_csv(csv_path: Path, raw_dir: Path, max_rows: int | None) -> Path:
    output_path = raw_dir / csv_path.name
    if max_rows is None:
        shutil.copy2(csv_path, output_path)
    else:
        dataframe = pd.read_csv(csv_path, nrows=max_rows)
        dataframe.to_csv(output_path, index=False)
    return output_path


def _real_ais_column_aliases() -> dict[str, list[str]]:
    return {
        "MMSI": ["mmsi", "vessel_id", "vesselid"],
        "timestamp": ["timestamp", "time", "datetime", "basedatetime", "# timestamp"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "long", "lng"],
        "speed": ["speed", "sog", "speed_over_ground"],
        "heading": ["heading", "cog", "course", "course_over_ground"],
    }


if __name__ == "__main__":
    main()
