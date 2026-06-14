from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Load Metrics
# --------------------------------------------------

def load_metrics(metrics_path: str | Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Table A
# Event Abstraction Comparison
# --------------------------------------------------

def build_event_table(
    ais_messages: int,
    event_count: int,
) -> pd.DataFrame:

    reduction = (
        (ais_messages - event_count)
        / ais_messages
        * 100
    )

    return pd.DataFrame(
        [
            {
                "Method": "Paper A",
                "AIS Messages": 338_000_000,
                "Events": 53_000_000,
                "Reduction %": 84.0,
            },
            {
                "Method": "Proposed",
                "AIS Messages": ais_messages,
                "Events": event_count,
                "Reduction %": round(reduction, 2),
            },
        ]
    )


# --------------------------------------------------
# Table B
# Trajectory Abstraction
# --------------------------------------------------

def build_trajectory_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Method": "Paper B",
                "Segmentation": "✓",
                "Event Discovery": "✓",
            },
            {
                "Method": "Proposed",
                "Segmentation": "✓",
                "Event Discovery": "✓",
            },
        ]
    )


# --------------------------------------------------
# Table C
# Activity Classification
# --------------------------------------------------

def build_classifier_table(metrics: dict) -> pd.DataFrame:

    test = metrics["test"]

    return pd.DataFrame(
        [
            {
                "Method": "Paper C",
                "Precision": None,
                "Recall": None,
                "F1": 0.817,
            },
            {
                "Method": "Paper D",
                "Precision": None,
                "Recall": None,
                "F1": 0.890,
            },
            {
                "Method": "Proposed XGBoost",
                "Precision": round(test["precision_macro"], 4),
                "Recall": round(test["recall_macro"], 4),
                "F1": round(test["f1_macro"], 4),
            },
        ]
    )


# --------------------------------------------------
# Table D
# Hierarchical Recognition
# --------------------------------------------------

def build_hierarchical_table(metrics: dict) -> pd.DataFrame:

    test = metrics["test"]

    return pd.DataFrame(
        [
            {
                "Method": "Paper E",
                "Hierarchical": "✓",
                "Accuracy": 0.98,
                "F1": 0.98,
            },
            {
                "Method": "Proposed",
                "Hierarchical": "✓",
                "Accuracy": round(test["accuracy"], 4),
                "F1": round(test["f1_macro"], 4),
            },
        ]
    )


# --------------------------------------------------
# Save Tables
# --------------------------------------------------

def save_tables(
    output_dir: Path,
    event_table: pd.DataFrame,
    trajectory_table: pd.DataFrame,
    classifier_table: pd.DataFrame,
    hierarchical_table: pd.DataFrame,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    event_table.to_csv(
        output_dir / "table_a_event_abstraction.csv",
        index=False,
    )

    trajectory_table.to_csv(
        output_dir / "table_b_trajectory_abstraction.csv",
        index=False,
    )

    classifier_table.to_csv(
        output_dir / "table_c_classifier_comparison.csv",
        index=False,
    )

    hierarchical_table.to_csv(
        output_dir / "table_d_hierarchical_recognition.csv",
        index=False,
    )


# --------------------------------------------------
# Summary JSON
# --------------------------------------------------

def save_summary(
    metrics: dict,
    output_dir: Path,
) -> None:

    test = metrics["test"]

    summary = {
        "best_model": metrics["best_model"],
        "accuracy": test["accuracy"],
        "precision_macro": test["precision_macro"],
        "recall_macro": test["recall_macro"],
        "f1_macro": test["f1_macro"],
    }

    with open(
        output_dir / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate against baseline maritime papers."
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="metrics.json from train_classifier.py",
    )

    parser.add_argument(
        "--ais-count",
        type=int,
        default=9_547_322,
    )

    parser.add_argument(
        "--event-count",
        type=int,
        default=40_990,
    )

    parser.add_argument(
        "--output-dir",
        default="evaluation",
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    metrics = load_metrics(args.metrics)

    output_dir = Path(args.output_dir)

    event_table = build_event_table(
        args.ais_count,
        args.event_count,
    )

    trajectory_table = build_trajectory_table()

    classifier_table = build_classifier_table(metrics)

    hierarchical_table = build_hierarchical_table(metrics)

    save_tables(
        output_dir,
        event_table,
        trajectory_table,
        classifier_table,
        hierarchical_table,
    )

    save_summary(
        metrics,
        output_dir,
    )

    print("\n=== Evaluation Complete ===\n")

    print(event_table)
    print()

    print(trajectory_table)
    print()

    print(classifier_table)
    print()

    print(hierarchical_table)
    print()

    print(
        f"Results written to: {output_dir}"
    )


if __name__ == "__main__":
    main()