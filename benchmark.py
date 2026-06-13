from __future__ import annotations
import json
import time
import pickle
import os
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from event_extractor import extract_event_sequence, EventThresholds
from feature_engineering import calculate_segment_features
from train_classifier import build_models

DATA_DIR = Path("data/processed")
CLEANED_AIS_PATH = DATA_DIR / "cleaned_ais.parquet"
SEGMENTS_PATH = DATA_DIR / "segments.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
LABELS_PATH = DATA_DIR / "situation_labels.parquet"
BEST_MODEL_PATH = Path("models/best_model.pkl")

# Features used for classifier training.
# anchor_duration, loiter_duration, crossing_count, distance_to_boundary
# are excluded to prevent label leakage (see train_classifier.py for details).
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

def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_pipeline() -> dict[str, Any]:
    print("Starting pipeline benchmark...")
    
    results = {}
    mem_before = get_memory_usage_mb()
    
    # 1. Event Extraction Benchmark
    print("Benchmarking Event Extraction on a sample of 250,000 AIS points...")
    cleaned_ais = pd.read_parquet(CLEANED_AIS_PATH)
    sample_ais = cleaned_ais.iloc[:250000].copy()
    
    start_time = time.perf_counter()
    mem_start = get_memory_usage_mb()
    events = extract_event_sequence(sample_ais, thresholds=EventThresholds())
    duration_events = time.perf_counter() - start_time
    mem_end = get_memory_usage_mb()
    
    events_per_sec = len(sample_ais) / duration_events if duration_events > 0 else 0
    results["event_extraction"] = {
        "sample_points": len(sample_ais),
        "duration_seconds": float(duration_events),
        "events_per_second": float(events_per_sec),
        "memory_used_mb": float(mem_end - mem_start)
    }
    print(f"Event Extraction: {events_per_sec:.2f} points/sec in {duration_events:.3f}s")
    
    # 2. Feature Extraction Benchmark
    print("Benchmarking Feature Extraction on a sample of 50,000 segment rows...")
    segments = pd.read_parquet(SEGMENTS_PATH)
    sample_segments = segments.iloc[:50000].copy()
    
    start_time = time.perf_counter()
    mem_start = get_memory_usage_mb()
    # Mocking distance boundary calculation path to avoid shapefile loading overhead if not needed, 
    # but calculating motion, spatial, crossing and temporal features
    features = calculate_segment_features(
        segments=sample_segments,
        boundary_events=None,
        event_sequence=None,
        eez_path=None
    )
    duration_features = time.perf_counter() - start_time
    mem_end = get_memory_usage_mb()
    
    segments_per_sec = len(sample_segments) / duration_features if duration_features > 0 else 0
    results["feature_extraction"] = {
        "sample_rows": len(sample_segments),
        "duration_seconds": float(duration_features),
        "segments_per_second": float(segments_per_sec),
        "memory_used_mb": float(mem_end - mem_start)
    }
    print(f"Feature Extraction: {segments_per_sec:.2f} segments/sec in {duration_features:.3f}s")
    
    # 3. Training Time Benchmark
    print("Benchmarking Model Training...")
    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    
    dataset = features_df.merge(
        labels_df[["segment_id", "situation_label"]],
        on="segment_id",
        how="inner"
    )
    dataset = dataset.dropna(subset=["situation_label"])
    dataset = dataset.loc[dataset["situation_label"] != "Unknown"].copy()
    
    activity_map = {
        "Transit": "Transit",
        "Boundary Crossing": "Transit",
        "Prolonged Boundary Presence": "Transit",
        "Anchorage": "Anchorage",
        "Fishing Activity": "Fishing Activity",
    }
    dataset["situation_label"] = dataset["situation_label"].map(activity_map)
    
    X = dataset[FEATURE_COLUMNS]
    y = dataset["situation_label"]
    
    # Reconstruct training split
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    n_classes = len(np.unique(y_encoded))
    models = build_models(n_jobs=-1, n_classes=n_classes)
    rf_model = models["Random Forest"]
    
    start_time = time.perf_counter()
    mem_start = get_memory_usage_mb()
    rf_model.fit(X_train, y_train)
    duration_train_rf = time.perf_counter() - start_time
    mem_end = get_memory_usage_mb()
    
    results["training_rf"] = {
        "train_rows": len(X_train),
        "duration_seconds": float(duration_train_rf),
        "memory_used_mb": float(mem_end - mem_start)
    }
    print(f"Random Forest Training: {duration_train_rf:.3f}s")
    
    # Train XGBoost
    xgb_model = models["XGBoost"]
    start_time = time.perf_counter()
    mem_start = get_memory_usage_mb()
    xgb_model.fit(X_train, y_train)
    duration_train_xgb = time.perf_counter() - start_time
    mem_end = get_memory_usage_mb()
    
    results["training_xgb"] = {
        "train_rows": len(X_train),
        "duration_seconds": float(duration_train_xgb),
        "memory_used_mb": float(mem_end - mem_start)
    }
    print(f"XGBoost Training: {duration_train_xgb:.3f}s")
    
    # 4. Inference Time Benchmark
    print("Benchmarking Model Inference on test set...")
    start_time = time.perf_counter()
    mem_start = get_memory_usage_mb()
    predictions = rf_model.predict(X_val)
    duration_inference = time.perf_counter() - start_time
    mem_end = get_memory_usage_mb()
    
    inference_per_sec = len(X_val) / duration_inference if duration_inference > 0 else 0
    results["inference"] = {
        "inference_rows": len(X_val),
        "duration_seconds": float(duration_inference),
        "inferences_per_second": float(inference_per_sec),
        "memory_used_mb": float(mem_end - mem_start)
    }
    print(f"Inference: {inference_per_sec:.2f} rows/sec in {duration_inference:.3f}s")
    
    # Save metrics
    results["peak_memory_mb"] = float(get_memory_usage_mb() - mem_before)
    
    with open(DATA_DIR / "performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Generate Benchmark Plot
    generate_benchmark_plots(results)
    
    return results

def generate_benchmark_plots(results: dict[str, Any]) -> None:
    # Plot 1: Execution Times
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = ["Event Extr. (250k)", "Feature Extr. (50k)", "RF Train (80k)", "XGB Train (80k)", "Inference (17k)"]
    times = [
        results["event_extraction"]["duration_seconds"],
        results["feature_extraction"]["duration_seconds"],
        results["training_rf"]["duration_seconds"],
        results["training_xgb"]["duration_seconds"],
        results["inference"]["duration_seconds"]
    ]
    
    bars = ax.barh(phases, times, color="#17a2b8")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("Computational Execution Time by Pipeline Phase", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.3f}s", 
                ha="left", va="center", fontsize=9, fontweight="bold")
                
    fig.tight_layout()
    fig.savefig("performance_benchmark.png", dpi=180)
    plt.close(fig)
    print("Saved performance_benchmark.png.")

if __name__ == "__main__":
    benchmark_pipeline()
