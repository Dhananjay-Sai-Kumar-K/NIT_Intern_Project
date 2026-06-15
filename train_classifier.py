from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_LABELS_PATH = Path("data/processed/situation_labels.parquet")
DEFAULT_MODEL_PATH = Path("models/best_model.pkl")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_CONFUSION_MATRIX_PATH = Path("models/confusion_matrix.png")
RANDOM_STATE = 42

# Features used for classifier training.
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
    "path_tortuosity",
    "heading_variance",
    "mean_heading_change",
    "loop_count",
    "turn_density",
    "stationary_ratio",
    "crossing_count",
    "anchor_duration",
    "loiter_duration",
]


def train_maritime_classifier(
    features_path: str | Path,
    labels_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    confusion_matrix_path: str | Path,
    config: dict[str, Any] | None = None,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    n_jobs: int = 1,
    tune: bool = False,
) -> dict[str, Any]:
    # Ensure config fallback to avoid dictionary lookup crashes
    runtime_config = config if config is not None else {}
    
    # Corrected call to include the required config argument
    dataset = load_training_dataset(features_path, labels_path, runtime_config)
    
    # Map complex labels to local activities (Transit, Anchorage, Fishing Activity)
    activity_map = {
        "Transit": "Transit",
        "Boundary Crossing": "Boundary Crossing",
        "Prolonged Boundary Presence": "Prolonged Boundary Presence",
        "Anchorage": "Anchorage",
        "Fishing Activity": "Fishing Activity",
    }
    dataset["situation_label"] = dataset["situation_label"].map(activity_map)

    X = dataset[FEATURE_COLUMNS]
    y_text = dataset["situation_label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        validation_size=validation_size,
    )

    n_classes = len(np.unique(y_train))
    models = build_models(n_jobs=n_jobs, n_classes=n_classes)
    validation_scores: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Pipeline] = {}
    
    # Compute class weights for XGBoost
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    if tune:
        from sklearn.model_selection import RandomizedSearchCV
        print("Running RandomizedSearchCV on XGBoost...")
        
        xgb_pipeline = models["XGBoost"]
        param_grid = {
            "classifier__max_depth": [4, 6, 8, 10],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__n_estimators": [100, 250, 500],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0, 0.1, 0.2, 0.5]
        }
        
        search = RandomizedSearchCV(
            xgb_pipeline,
            param_distributions=param_grid,
            n_iter=30,
            scoring="f1_macro",
            cv=3,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            verbose=2
        )
        # Pass sample weights to XGBoost via the pipeline
        search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        
        print("Best params:", search.best_params_)
        models["XGBoost"] = search.best_estimator_

    for model_name, model in models.items():
        if model_name == "XGBoost" and not tune:
            model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        elif model_name == "XGBoost" and tune:
            # Already fitted by RandomizedSearchCV
            pass
        else:
            # Random Forest uses class_weight="balanced" inside its init
            model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)
        validation_scores[model_name] = calculate_metrics(
            y_validation,
            validation_predictions,
            labels=list(range(len(label_encoder.classes_))),
        )
        trained_models[model_name] = model

    best_model_name = max(
        validation_scores,
        key=lambda name: validation_scores[name]["f1_macro"],
    )
    best_model = trained_models[best_model_name]
    test_predictions = best_model.predict(X_test)
    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        labels=list(range(len(label_encoder.classes_))),
    )

    print("\n=== Fishing Activity Misclassification Breakdown ===")
    try:
        fishing_idx = list(label_encoder.classes_).index("Fishing Activity")
        conf_mat = confusion_matrix(y_test, test_predictions)
        fishing_row = conf_mat[fishing_idx]
        for idx, count in enumerate(fishing_row):
            if count > 0:
                pred_class = label_encoder.classes_[idx]
                if pred_class != "Fishing Activity":
                    print(f"Fishing → {pred_class}: {count}")
    except ValueError:
        print("Fishing Activity not found in test classes.")
        
    plot_feature_importance(best_model, FEATURE_COLUMNS, "models/feature_importance.png")

    metrics_payload = {
        "best_model": best_model_name,
        "label_classes": label_encoder.classes_.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "split": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "test_size": test_size,
            "validation_size": validation_size,
        },
        "validation": validation_scores,
        "test": test_metrics,
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    save_model(
        {
            "model": best_model,
            "label_encoder": label_encoder,
            "feature_columns": FEATURE_COLUMNS,
            "best_model_name": best_model_name,
        },
        model_path,
    )
    save_metrics(metrics_payload, metrics_path)
    save_confusion_matrix(
        y_test,
        test_predictions,
        label_encoder.classes_,
        confusion_matrix_path,
    )
    return metrics_payload


def load_training_dataset(features_path: str, labels_path: str, config: dict) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    dataset = features.merge(labels, on="segment_id", how="inner")

    # OPTION B: Map fine-grained Reasoner situations back to Base Activities for ML
    activity_map = {
        "Anchorage": "Anchorage",
        "Transit": "Transit",
        "Boundary Crossing": "Transit",
        "Prolonged Boundary Presence": "Transit",
        "Repeated Crossing": "Transit",
        "Fishing Activity": "Fishing Activity"
    }
    dataset["activity_label"] = dataset["situation_label"].map(activity_map)

    # Extract downsampling threshold from config
    max_majority = config.get("classifier_training", {}).get("max_majority_samples", 15000)

    # Perform Downsampling on the new Base Activities
    sampled_dfs = []
    for label, group in dataset.groupby("activity_label"):
        if len(group) > max_majority and label in ["Transit", "Anchorage"]:
            sampled = group.sample(n=max_majority, random_state=42)
            sampled_dfs.append(sampled)
        else:
            sampled_dfs.append(group)

    return pd.concat(sampled_dfs).reset_index(drop=True)


def split_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validation_fraction = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_models(n_jobs: int = 1, n_classes: int = 3) -> dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )
    xgb_objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    xgb_eval_metric = "mlogloss" if n_classes > 2 else "logloss"
    return {
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=n_jobs,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective=xgb_objective,
                        eval_metric=xgb_eval_metric,
                        tree_method="hist",
                        n_jobs=n_jobs,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    labels: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision_macro": float(
            precision_score(y_true, y_predicted, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_predicted, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_predicted, labels=labels, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_predicted,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_predicted,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_predicted,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_model(model_bundle: dict[str, Any], model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(model_bundle, model_file)


def save_metrics(metrics: dict[str, Any], metrics_path: str | Path) -> None:
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    class_names: np.ndarray,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_predicted, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    figure, axis = plt.subplots(figsize=(10, 8))
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=30)
    axis.set_title("Maritime Situation Classifier Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

def plot_feature_importance(model, feature_names, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classifier = model.named_steps.get("classifier")
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        
        print("\n=== Top Features ===")
        for f in range(min(20, len(importances))):
            print(f"{f + 1}. feature {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    else:
        print("Model does not support feature importances.")


def main(config_path: str | Path = "config.yaml") -> None:
    config = _load_config(config_path)
    training_config = config.get("classifier_training", {})
    train_maritime_classifier(
        features_path=config["output"].get("features_path", DEFAULT_FEATURES_PATH),
        labels_path=config["output"].get("situation_labels_path", DEFAULT_LABELS_PATH),
        model_path=config["output"].get("best_model_path", DEFAULT_MODEL_PATH),
        metrics_path=config["output"].get("metrics_path", DEFAULT_METRICS_PATH),
        confusion_matrix_path=config["output"].get(
            "confusion_matrix_path",
            DEFAULT_CONFUSION_MATRIX_PATH,
        ),
        config=config,
        test_size=training_config.get("test_size", 0.15),
        validation_size=training_config.get("validation_size", 0.15),
        n_jobs=training_config.get("n_jobs", 1),
        tune=training_config.get("tune", False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train maritime situation classifiers.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV hyperparameter tuning on XGBoost")
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.config:
        config = _load_config(args.config)
        training_config = config.get("classifier_training", {})
        metrics = train_maritime_classifier(
            features_path=config["output"].get("features_path", DEFAULT_FEATURES_PATH),
            labels_path=config["output"].get("situation_labels_path", DEFAULT_LABELS_PATH),
            model_path=config["output"].get("best_model_path", DEFAULT_MODEL_PATH),
            metrics_path=config["output"].get("metrics_path", DEFAULT_METRICS_PATH),
            confusion_matrix_path=config["output"].get(
                "confusion_matrix_path",
                DEFAULT_CONFUSION_MATRIX_PATH,
            ),
            config=config,
            test_size=training_config.get("test_size", args.test_size),
            validation_size=training_config.get("validation_size", args.validation_size),
            n_jobs=training_config.get("n_jobs", 1),
            tune=args.tune,
        )
    else:
        # Fallback dictionary if no config YAML file is supplied
        fallback_config = {}
        metrics = train_maritime_classifier(
            features_path=args.features,
            labels_path=args.labels,
            model_path=args.model_output,
            metrics_path=args.metrics_output,
            confusion_matrix_path=args.confusion_matrix_output,
            config=fallback_config,
            test_size=args.test_size,
            validation_size=args.validation_size,
            n_jobs=1,
            tune=args.tune,
        )

    print(f"Best model: {metrics['best_model']}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test F1 macro: {metrics['test']['f1_macro']:.4f}")
    print("Wrote model, metrics, and confusion matrix.")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    cli()