from __future__ import annotations
import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from shapely.geometry import LineString, Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed")

CLEANED_AIS_PATH = DATA_DIR / "cleaned_ais.parquet"
SEGMENTS_PATH = DATA_DIR / "segments.parquet"
EVENTS_PATH = DATA_DIR / "event_sequence.json"
FEATURES_PATH = DATA_DIR / "features.parquet"
LABELS_PATH = DATA_DIR / "situation_labels.parquet"
REASONED_SITUATIONS_PATH = DATA_DIR / "reasoned_situations.json"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

RANDOM_STATE = 42
TEST_FRACTION = 0.15   # fraction of vessels held out for test

# Features used for classifier training and evaluation.
# IMPORTANT: anchor_duration, loiter_duration, crossing_count, and
# distance_to_boundary are intentionally excluded because they are the direct
# operands of the weak-supervision label rules (label_generator.py). Exposing
# them to the classifier allows trivial threshold rediscovery (label leakage).
# All features below are behaviour-derived and do not appear in any label rule.
FEATURE_COLUMNS = [
    "window_minutes",
    "duration_seconds",
    "number_of_points",
    "avg_speed",
    "max_speed",
    "speed_variance",
    "path_length",
    "displacement",
    "trajectory_curvature",
]

# Canonical class order for confusion matrix display
CLASS_ORDER = [
    "Transit",
    "Anchorage",
    "Fishing Activity",
    "Boundary Crossing",
    "Prolonged Boundary Presence",
]

# Canonical classes for the dual tasks
ACTIVITY_CLASSES = [
    "Transit",
    "Anchorage",
    "Fishing Activity",
]

SITUATION_CLASSES = [
    "Boundary Crossing",
    "Prolonged Boundary Presence",
    "Repeated Crossing",
]

SITUATION_EVAL_CLASSES = [
    "Boundary Crossing",
    "Prolonged Boundary Presence",
    "None",
]

def _map_to_activity(label: str) -> str:
    if label == "Anchorage":
        return "Anchorage"
    if label == "Fishing Activity":
        return "Fishing Activity"
    return "Transit"

def _map_to_situation(label: str) -> str:
    if label == "Boundary Crossing":
        return "Boundary Crossing"
    if label == "Prolonged Boundary Presence":
        return "Prolonged Boundary Presence"
    return "None"

# Map from reasoner situation names → classifier class names
REASONER_TO_CLASSIFIER_MAP = {
    "Anchorage": "Anchorage",
    "Prolonged Boundary Presence": "Prolonged Boundary Presence",
    "Repeated Crossing": "Boundary Crossing",
    "Boundary Crossing": "Boundary Crossing",
    "Transit": "Transit",
    "Maneuvering": "Transit",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any],
    pd.DataFrame, pd.DataFrame, list[dict[str, Any]]
]:
    print("Loading datasets...")
    cleaned_ais = pd.read_parquet(CLEANED_AIS_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH)

    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    with open(REASONED_SITUATIONS_PATH, "r", encoding="utf-8") as f:
        reasoned_data = json.load(f)

    return cleaned_ais, segments, events_data, features, labels, reasoned_data["situations"]


# ---------------------------------------------------------------------------
# Part A – Event Abstraction
# ---------------------------------------------------------------------------

def run_part_a(cleaned_ais: pd.DataFrame, events_data: dict[str, Any]) -> dict[str, Any]:
    print("Evaluating Part A: Event Abstraction...")
    n_points = len(cleaned_ais)
    events = events_data["events"]
    n_events = len(events)

    reduction_ratio = 1.0 - (n_events / n_points)
    density = (n_events / n_points) * 1000.0

    event_types = [e["event_type"] for e in events]
    dist_counts = pd.Series(event_types).value_counts().to_dict()

    distribution = {}
    for etype in ["ENTRY", "EXIT", "ANCHOR", "LOITER", "MANEUVERING", "REPEATED_CROSSING"]:
        count = dist_counts.get(etype, 0)
        percentage = (count / n_events) * 100.0 if n_events > 0 else 0.0
        distribution[etype] = {
            "count": int(count),
            "percentage": float(percentage),
        }

    metrics = {
        "ais_point_count": int(n_points),
        "extracted_event_count": int(n_events),
        "event_reduction_ratio": float(reduction_ratio),
        "event_density_per_1000_points": float(density),
        "event_distribution": distribution,
    }

    with open(OUTPUT_DIR / "event_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Part A completed. Reduction ratio: {reduction_ratio:.2%}")
    return metrics


# ---------------------------------------------------------------------------
# Part B – Trajectory Simplification
# ---------------------------------------------------------------------------

def run_part_b(cleaned_ais: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    print("Evaluating Part B: Trajectory Simplification...")

    num_points = len(segments)
    unique_segments = segments["segment_id"].nunique()
    simplified_points = unique_segments * 2
    global_compression_ratio = 1.0 - (simplified_points / num_points)

    print("Calculating spatial and temporal deviations on a representative sample of segments...")
    np.random.seed(RANDOM_STATE)
    all_segment_ids = segments["segment_id"].unique()
    sample_segment_ids = np.random.choice(
        all_segment_ids,
        size=min(1000, len(all_segment_ids)),
        replace=False,
    )

    spatial_deviations: list[float] = []
    temporal_deviations: list[float] = []

    sampled_rows = segments[segments["segment_id"].isin(sample_segment_ids)].copy()
    sampled_rows["timestamp"] = pd.to_datetime(sampled_rows["timestamp"], utc=True)

    for seg_id, group in sampled_rows.groupby("segment_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        if n < 3:
            continue

        lats = group["latitude"].to_numpy()
        lons = group["longitude"].to_numpy()
        times = group["timestamp"].astype("int64").to_numpy() / 1e9

        a_lat, a_lon, a_t = lats[0], lons[0], times[0]
        b_lat, b_lon, b_t = lats[-1], lons[-1], times[-1]

        d_lon_total = b_lon - a_lon
        d_lat_total = b_lat - a_lat
        denom = d_lon_total ** 2 + d_lat_total ** 2

        for i in range(1, n - 1):
            p_lat, p_lon, p_t = lats[i], lons[i], times[i]

            if denom == 0:
                t = 0.0
            else:
                t = ((p_lon - a_lon) * d_lon_total + (p_lat - a_lat) * d_lat_total) / denom
                t = max(0.0, min(1.0, t))

            c_lat = a_lat + t * d_lat_total
            c_lon = a_lon + t * d_lon_total

            dx = (p_lon - c_lon) * 111.32 * math.cos(math.radians(a_lat))
            dy = (p_lat - c_lat) * 110.57
            dist_km = math.sqrt(dx ** 2 + dy ** 2)
            spatial_deviations.append(dist_km)

            interpolated_t = a_t + t * (b_t - a_t)
            time_err = abs(p_t - interpolated_t)
            temporal_deviations.append(time_err)

    mean_spatial = float(np.mean(spatial_deviations)) if spatial_deviations else 0.0
    max_spatial = float(np.max(spatial_deviations)) if spatial_deviations else 0.0
    mean_temporal = float(np.mean(temporal_deviations)) if temporal_deviations else 0.0
    max_temporal = float(np.max(temporal_deviations)) if temporal_deviations else 0.0

    print("Running Douglas-Peucker baseline comparison...")
    vessel_counts = cleaned_ais["MMSI"].value_counts()
    sample_mmsis = vessel_counts.index[:50]

    dp_tolerances = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    dp_compressions = {tol: [] for tol in dp_tolerances}
    dp_errors = {tol: [] for tol in dp_tolerances}

    for mmsi in sample_mmsis:
        v_data = cleaned_ais[cleaned_ais["MMSI"] == mmsi].sort_values("timestamp")
        if len(v_data) < 10:
            continue

        step = max(1, len(v_data) // 500)
        v_data_sampled = v_data.iloc[::step].copy()

        lats = v_data_sampled["latitude"].tolist()
        lons = v_data_sampled["longitude"].tolist()

        line = LineString(zip(lons, lats))
        total_pts = len(v_data_sampled)

        for tol in dp_tolerances:
            simplified = line.simplify(tol, preserve_topology=False)
            simplified_coords = list(simplified.coords)
            comp = 1.0 - (len(simplified_coords) / total_pts)
            dp_compressions[tol].append(comp)

            errors = []
            n_pts = len(lons)
            sample_size = min(100, n_pts)
            np.random.seed(RANDOM_STATE)
            indices = np.random.choice(n_pts, size=sample_size, replace=False)
            for idx in indices:
                pt = Point(lons[idx], lats[idx])
                dist_deg = simplified.distance(pt)
                dist_km = dist_deg * 111.0
                errors.append(dist_km)
            dp_errors[tol].append(np.mean(errors))

    dp_comp_curve = [float(np.mean(dp_compressions[tol])) for tol in dp_tolerances]
    dp_err_curve = [float(np.mean(dp_errors[tol])) for tol in dp_tolerances]

    metrics = {
        "compression_ratio": global_compression_ratio,
        "trajectory_complexity_reduction": global_compression_ratio,
        "spatial_deviation_mean_km": mean_spatial,
        "spatial_deviation_max_km": max_spatial,
        "temporal_deviation_mean_seconds": mean_temporal,
        "temporal_deviation_max_seconds": max_temporal,
        "dp_baseline": {
            "tolerances": dp_tolerances,
            "compression_ratios": dp_comp_curve,
            "mean_errors_km": dp_err_curve,
        },
    }

    with open(OUTPUT_DIR / "trajectory_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot 1: compression_vs_error.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dp_err_curve, dp_comp_curve, "o-", label="Douglas-Peucker (EPP Baseline)",
            color="#1f77b4", linewidth=2)
    ax.plot(mean_spatial, global_compression_ratio, "s",
            label="Proposed Event-Based Abstraction", color="#ff7f0e", markersize=10)
    ax.set_xlabel("Mean Spatial Deviation (km)", fontsize=11)
    ax.set_ylabel("Compression Ratio", fontsize=11)
    ax.set_title("Trajectory Simplification Trade-off: Compression vs Error",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("compression_vs_error.png", dpi=180)
    plt.close(fig)

    # Plot 2: trajectory_comparison.png
    best_mmsi = vessel_counts.index[0]
    v_data = cleaned_ais[cleaned_ais["MMSI"] == best_mmsi].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(v_data["longitude"], v_data["latitude"], "-",
            label="Raw AIS Trajectory", color="#cccccc", linewidth=1.5, alpha=0.8)

    line = LineString(zip(v_data["longitude"], v_data["latitude"]))
    dp_simplified = line.simplify(0.01, preserve_topology=False)
    dp_lons, dp_lats = zip(*dp_simplified.coords)
    ax.plot(dp_lons, dp_lats, "o-", label="Douglas-Peucker (EPP)", color="#1f77b4", linewidth=2)

    v_segs = segments[segments["MMSI"] == best_mmsi].sort_values("timestamp")
    seg_endpoints_lons: list[float] = []
    seg_endpoints_lats: list[float] = []
    for _, seg_group in v_segs.groupby("segment_id"):
        seg_group = seg_group.sort_values("timestamp")
        seg_endpoints_lons.extend([seg_group["longitude"].iloc[0], seg_group["longitude"].iloc[-1]])
        seg_endpoints_lats.extend([seg_group["latitude"].iloc[0], seg_group["latitude"].iloc[-1]])

    ax.plot(seg_endpoints_lons, seg_endpoints_lats, "s-",
            label="Proposed Abstraction", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(f"Trajectory Abstraction Comparison (Vessel MMSI: {best_mmsi})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("trajectory_comparison.png", dpi=180)
    plt.close(fig)

    print("Part B completed. Plots saved.")
    return metrics


# ---------------------------------------------------------------------------
# Part C & D – Situation Classification & Hierarchical Reasoning
# ---------------------------------------------------------------------------

def _vessel_independent_split(
    dataset: pd.DataFrame,
    test_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by unique vessel (MMSI), not by row.

    This is critical: random-row splits allow segments of the same vessel to
    appear in both train and test, inflating accuracy because the model
    memorises vessel-specific patterns.  Vessel-independent splits measure
    true generalisation to unseen vessels.
    """
    unique_mmsis = dataset["mmsi"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_mmsis)

    n_test = max(1, int(len(unique_mmsis) * test_fraction))
    test_mmsis = set(unique_mmsis[:n_test])

    test_mask = dataset["mmsi"].isin(test_mmsis)
    return dataset[~test_mask].copy(), dataset[test_mask].copy()


def _build_situations_index(
    situations: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Pre-index reasoned situations by MMSI with parsed timestamps."""
    index: dict[int, list[dict[str, Any]]] = {}
    for sit in situations:
        mmsi = sit["mmsi"]
        sit["start_dt"] = pd.to_datetime(sit["start_timestamp"], utc=True)
        sit["end_dt"] = pd.to_datetime(sit["end_timestamp"], utc=True)
        index.setdefault(mmsi, []).append(sit)
    return index


def _reasoner_label_for_segment(
    mmsi: int,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    situations_index: dict[int, list[dict[str, Any]]],
) -> str | None:
    """
    Return the highest-confidence reasoner class that overlaps this segment,
    or None if the reasoner has no opinion.
    """
    best_label: str | None = None
    best_conf = -1.0
    for sit in situations_index.get(mmsi, []):
        if seg_start <= sit["end_dt"] and seg_end >= sit["start_dt"]:
            clf_class = REASONER_TO_CLASSIFIER_MAP.get(sit["situation"], "Transit")
            if sit["confidence"] > best_conf:
                best_conf = sit["confidence"]
                best_label = clf_class
    return best_label


def _build_xgb_pipeline(rf_pipeline: Any, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
    """
    Train a fresh XGBoost pipeline that shares the same preprocessor as the
    loaded RF pipeline so the comparison is fair (same feature scaling).
    Note: We clone the preprocessor to avoid fitting it a second time on
    different data.
    """
    from sklearn.base import clone
    preprocessor = clone(rf_pipeline.named_steps["preprocessor"])
    
    n_classes = len(np.unique(y_train))
    xgb_objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    xgb_eval_metric = "mlogloss" if n_classes > 2 else "logloss"
    
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        objective=xgb_objective,
        eval_metric=xgb_eval_metric,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_clf),
    ])
    xgb_pipeline.fit(X_train, y_train)
    return xgb_pipeline


def _compute_clf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute all classification metrics against a fixed ground truth array and list of classes."""
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    per_class: dict[str, dict[str, float]] = {}
    for c in classes:
        c_true = (y_true == c)
        c_pred = (y_pred == c)
        per_class[c] = {
            "precision": float(precision_score(c_true, c_pred, zero_division=0)),
            "recall": float(recall_score(c_true, c_pred, zero_division=0)),
            "f1_score": float(f1_score(c_true, c_pred, zero_division=0)),
        }

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa),
        "mcc": float(mcc),
        "per_class": per_class,
    }


def _reasoner_situation_for_segment(
    mmsi: int,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    situations_index: dict[int, list[dict[str, Any]]],
) -> str | None:
    """
    Return the highest-confidence reasoner situation (of interest) that overlaps this segment,
    or None if no situation applies.
    """
    best_sit: str | None = None
    best_conf = -1.0
    sits_of_interest = {"Boundary Crossing", "Prolonged Boundary Presence", "Repeated Crossing"}
    for sit in situations_index.get(mmsi, []):
        situation_name = sit["situation"]
        if situation_name in sits_of_interest:
            if seg_start <= sit["end_dt"] and seg_end >= sit["start_dt"]:
                if sit["confidence"] > best_conf:
                    best_conf = sit["confidence"]
                    best_sit = situation_name
    return best_sit


def run_part_c_and_d(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    situations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate situation classification (Part C) and the hierarchical reasoning
    correction layer (Part D) in a truly hierarchical, dual-task framework.

    The pipeline is structured as follows:
    1. Task 1: Activity Recognition (predicting Transit, Anchorage, Fishing Activity).
       The classifiers (RF and XGBoost) are trained to predict these local activities.
    2. Task 2: Situation Detection (predicting Boundary Crossing, Prolonged Presence, Repeated Crossing).
       The Temporal Reasoner infers these contexts by matching event patterns.
    """
    print("Evaluating Part C & D: Hierarchical Maritime Situation Recognition...")

    # 1. Merge features + labels, drop Unknown
    dataset = features.merge(
        labels[["segment_id", "situation_label"]],
        on="segment_id",
        how="inner",
    )
    dataset = dataset.dropna(subset=["situation_label"])
    dataset = dataset.loc[dataset["situation_label"] != "Unknown"].copy()
    dataset = dataset.reset_index(drop=True)

    # Parse segment timestamps (needed for reasoner lookup)
    dataset["segment_start_dt"] = pd.to_datetime(dataset["segment_start"], utc=True)
    dataset["segment_end_dt"] = pd.to_datetime(dataset["segment_end"], utc=True)

    print(f"Dataset size after filtering: {len(dataset):,} segments across "
          f"{dataset['mmsi'].nunique():,} vessels.")

    # 2. Vessel-independent train / test split
    train_dataset, test_dataset = _vessel_independent_split(
        dataset, test_fraction=TEST_FRACTION, random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_dataset):,} segments ({train_dataset['mmsi'].nunique():,} vessels) | "
          f"Test: {len(test_dataset):,} segments ({test_dataset['mmsi'].nunique():,} vessels)")

    X_train = train_dataset[FEATURE_COLUMNS]
    X_test = test_dataset[FEATURE_COLUMNS]

    # Ground truth labels for evaluate
    y_true_activity = test_dataset["situation_label"].map(_map_to_activity).to_numpy()
    y_true_situation = test_dataset["situation_label"].map(_map_to_situation).to_numpy()
    y_true_original = test_dataset["situation_label"].to_numpy()

    # Train target is mapped to the 3 activity classes
    y_train_activity_text = train_dataset["situation_label"].map(_map_to_activity)

    # Encode labels for sklearn/xgboost
    le = LabelEncoder()
    le.fit(dataset["situation_label"].map(_map_to_activity))
    y_train_encoded = le.transform(y_train_activity_text)

    # ------------------------------------------------------------------
    # 3. Rule-Based baseline (evaluates Rule-Based's F1 on both tasks)
    # ------------------------------------------------------------------
    rule_predictions_activity = y_true_activity
    rule_predictions_situation = y_true_situation

    # ------------------------------------------------------------------
    # 4. Random Forest Activity Classifier
    # ------------------------------------------------------------------
    print("Loading pre-trained RF pipeline architecture and re-fitting on vessel-independent train split...")
    with open(BEST_MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)

    rf_pipeline = model_bundle["model"]

    # Re-train RF on 3 local activity targets
    from sklearn.base import clone
    rf_pipeline_refit = clone(rf_pipeline)
    rf_pipeline_refit.fit(X_train, y_train_encoded)

    rf_predictions_activity_encoded = rf_pipeline_refit.predict(X_test)
    rf_predictions_activity = le.inverse_transform(rf_predictions_activity_encoded)

    # ------------------------------------------------------------------
    # 5. XGBoost Activity Classifier
    # ------------------------------------------------------------------
    print("Training XGBoost baseline on vessel-independent train split...")
    xgb_pipeline = _build_xgb_pipeline(rf_pipeline_refit, X_train, y_train_encoded)
    xgb_predictions_activity_encoded = xgb_pipeline.predict(X_test)
    xgb_predictions_activity = le.inverse_transform(xgb_predictions_activity_encoded)

    # ------------------------------------------------------------------
    # 6. Proposed System (Situation Detection from Reasoner)
    # ------------------------------------------------------------------
    print("Inferring contextual situations using Temporal Reasoner...")
    situations_index = _build_situations_index(situations)

    context_predictions_raw: list[str | None] = []

    for row in test_dataset.itertuples(index=False):
        reasoner_situation = _reasoner_situation_for_segment(
            mmsi=row.mmsi,
            seg_start=row.segment_start_dt,
            seg_end=row.segment_end_dt,
            situations_index=situations_index,
        )
        context_predictions_raw.append(reasoner_situation)

    # For quantitative evaluation, map "Repeated Crossing" to "Boundary Crossing"
    context_predictions_eval = []
    for pred in context_predictions_raw:
        if pred == "Repeated Crossing":
            context_predictions_eval.append("Boundary Crossing")
        elif pred is None:
            context_predictions_eval.append("None")
        else:
            context_predictions_eval.append(pred)
    context_predictions_eval = np.array(context_predictions_eval)

    # ------------------------------------------------------------------
    # 7. Compute Metrics
    # ------------------------------------------------------------------
    print("Computing evaluation metrics...")
    
    # Task 1: Activity Recognition Metrics
    rule_activity_metrics = _compute_clf_metrics(y_true_activity, rule_predictions_activity, ACTIVITY_CLASSES)
    rf_activity_metrics = _compute_clf_metrics(y_true_activity, rf_predictions_activity, ACTIVITY_CLASSES)
    xgb_activity_metrics = _compute_clf_metrics(y_true_activity, xgb_predictions_activity, ACTIVITY_CLASSES)

    # Task 2: Situation Detection Metrics
    rule_situation_metrics = _compute_clf_metrics(y_true_situation, rule_predictions_situation, SITUATION_EVAL_CLASSES)
    proposed_situation_metrics = _compute_clf_metrics(y_true_situation, context_predictions_eval, SITUATION_EVAL_CLASSES)

    # Task 3: Context Enrichment Metrics
    total_segments = len(test_dataset)
    enriched_segments = sum(1 for p in context_predictions_raw if p is not None)
    coverage = enriched_segments / total_segments if total_segments > 0 else 0.0

    counts = pd.Series(context_predictions_raw).value_counts().to_dict()
    boundary_crossing_detected = int(counts.get("Boundary Crossing", 0))
    prolonged_presence_detected = int(counts.get("Prolonged Boundary Presence", 0))
    repeated_crossing_detected = int(counts.get("Repeated Crossing", 0))

    classification_payload = {
        "Rule-Based_Activity": rule_activity_metrics,
        "Random Forest_Activity": rf_activity_metrics,
        "XGBoost_Activity": xgb_activity_metrics,
        "Rule-Based_Situation": rule_situation_metrics,
        "Proposed_Situation": proposed_situation_metrics,
    }

    reasoning_payload = {
        "total_test_segments": total_segments,
        "enriched_segments": enriched_segments,
        "context_coverage_ratio": float(coverage),
        "boundary_crossing_situations_detected": boundary_crossing_detected,
        "prolonged_presence_situations_detected": prolonged_presence_detected,
        "repeated_crossing_situations_detected": repeated_crossing_detected,
    }

    # For dashboard/unified metrics loading backward compatibility
    # Write a dummy class report that mimics the old expected format
    dummy_class_report = rf_activity_metrics.copy()
    dummy_class_report["f1_score"] = rf_activity_metrics["f1_score"]

    with open(OUTPUT_DIR / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(classification_payload, f, indent=2)

    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(dummy_class_report, f, indent=2)

    with open(OUTPUT_DIR / "reasoning_metrics.json", "w", encoding="utf-8") as f:
        json.dump(reasoning_payload, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Save prediction audit table (with backwards compatibility)
    # ------------------------------------------------------------------
    proposed_predictions_compat = []
    for act, sit in zip(rf_predictions_activity, context_predictions_raw):
        if sit is not None and sit != "None":
            proposed_predictions_compat.append(sit)
        else:
            proposed_predictions_compat.append(act)

    audit = test_dataset[["segment_id", "mmsi", "situation_label"]].copy()
    audit["ground_truth"] = y_true_original
    audit["rule_predictions"] = rule_predictions_activity
    audit["rf_predictions"] = rf_predictions_activity
    audit["xgb_predictions"] = xgb_predictions_activity
    audit["proposed_predictions"] = proposed_predictions_compat
    
    # Task-specific columns
    audit["ground_truth_activity"] = y_true_activity
    audit["ground_truth_situation"] = y_true_situation
    audit["rf_predicted_activity"] = rf_predictions_activity
    audit["xgb_predicted_activity"] = xgb_predictions_activity
    audit["proposed_context_prediction"] = [p if p is not None else "None" for p in context_predictions_raw]
    audit.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)

    # ------------------------------------------------------------------
    # 9. Confusion Matrices
    # ------------------------------------------------------------------
    # 9a. Activity Recognition Confusion Matrix (RF)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_act = confusion_matrix(y_true_activity, rf_predictions_activity, labels=ACTIVITY_CLASSES)
    disp_act = ConfusionMatrixDisplay(confusion_matrix=cm_act, display_labels=ACTIVITY_CLASSES)
    disp_act.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("Activity Recognition Confusion Matrix (Random Forest)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=180)  # Overwrite confusion_matrix.png for dashboard
    plt.close(fig)

    # 9b. Situation Detection Confusion Matrix (Proposed)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_sit = confusion_matrix(y_true_situation, context_predictions_eval, labels=SITUATION_EVAL_CLASSES)
    disp_sit = ConfusionMatrixDisplay(confusion_matrix=cm_sit, display_labels=SITUATION_EVAL_CLASSES)
    disp_sit.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("Situation Detection Confusion Matrix (Proposed Reasoner)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("confusion_matrix_situation.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 10. Performance Plot
    # ------------------------------------------------------------------
    metrics_names = ["Activity Acc", "Activity F1", "Situation Acc", "Situation F1"]
    vals_baseline = [
        rule_activity_metrics["accuracy"],
        rule_activity_metrics["f1_score"],
        rule_situation_metrics["accuracy"],
        rule_situation_metrics["f1_score"]
    ]
    vals_rf_prop = [
        rf_activity_metrics["accuracy"],
        rf_activity_metrics["f1_score"],
        proposed_situation_metrics["accuracy"],
        proposed_situation_metrics["f1_score"]
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, vals_baseline, width, label="Rule-Based Baseline (Paper 3)", color="#1f77b4")
    ax.bar(x + width / 2, vals_rf_prop, width, label="Proposed Hierarchical Framework", color="#2ca02c")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Hierarchical Framework Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="lower left", fontsize=10)

    for i in range(len(metrics_names)):
        ax.text(i - width / 2, vals_baseline[i] + 0.02, f"{vals_baseline[i]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, vals_rf_prop[i] + 0.02, f"{vals_rf_prop[i]:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig("reasoning_improvement.png", dpi=180)  # Overwrite reasoning_improvement.png for dashboard
    plt.close(fig)

    print("Part C & D completed.")
    return classification_payload, reasoning_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cleaned_ais, segments, events_data, features, labels, situations = load_data()

    part_a_metrics = run_part_a(cleaned_ais, events_data)
    part_b_metrics = run_part_b(cleaned_ais, segments)
    classification_metrics, reasoning_metrics = run_part_c_and_d(features, labels, situations)

    unified_metrics = {
        "event_abstraction":       part_a_metrics,
        "trajectory_simplification": part_b_metrics,
        "situation_classification": classification_metrics,
        "hierarchical_reasoning":   reasoning_metrics,
    }

    performance_metrics_path = Path("data/processed/performance_metrics.json")
    if performance_metrics_path.exists():
        with open(performance_metrics_path, "r", encoding="utf-8") as f:
            unified_metrics["performance"] = json.load(f)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(unified_metrics, f, indent=2)

    print("All evaluations completed. Wrote metrics.json.")


if __name__ == "__main__":
    main()
