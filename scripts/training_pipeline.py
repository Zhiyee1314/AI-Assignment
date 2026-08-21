"""Shared training and feature-ablation pipeline for the diabetes app.

All three classifiers use the same train/test split, median imputer and
MinMaxScaler. Keeping this logic in one module prevents the ANN, SVM and KNN
scripts from accidentally using different preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "Data" / "diabetes.csv"
MODELS_DIR = REPO_ROOT / "models"
ABLATION_PATH = REPO_ROOT / "Data" / "feature_ablation_results.csv"
COMPARISON_PATH = REPO_ROOT / "Data" / "model_comparison_results.csv"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_clean_data() -> pd.DataFrame:
    """Load numeric data and remove exact duplicate patient rows."""
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data = data[required].apply(pd.to_numeric, errors="raise").drop_duplicates().reset_index(drop=True)
    for column in ZERO_AS_MISSING_COLS:
        data[column] = data[column].replace(0, np.nan)
    return data


def split_data(data: pd.DataFrame, features: list[str]):
    return train_test_split(
        data[features],
        data[TARGET],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )


def fit_preprocessors(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit on training data only and return values clipped to 0.00-1.00."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    return imputer, scaler, X_train_scaled, X_test_scaled


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tune_models(X_train: np.ndarray, y_train: pd.Series):
    """Tune each algorithm for accuracy on the same cross-validation folds."""
    searches = {
        "ANN": GridSearchCV(
            MLPClassifier(
                solver="lbfgs",
                max_iter=5000,
                max_fun=50000,
                random_state=RANDOM_STATE,
            ),
            {
                "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
            },
            cv=_cv(),
            scoring="accuracy",
            n_jobs=-1,
        ),
        "SVM": GridSearchCV(
            SVC(random_state=RANDOM_STATE),
            [
                {
                    "kernel": ["rbf"],
                    "C": [0.1, 0.3, 0.5, 1, 3, 10, 30],
                    "gamma": ["scale", 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    "class_weight": [None, "balanced"],
                },
                {
                    "kernel": ["linear"],
                    "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    "class_weight": [None, "balanced"],
                },
            ],
            cv=_cv(),
            scoring="accuracy",
            n_jobs=-1,
        ),
        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            cv=_cv(),
            scoring="accuracy",
            n_jobs=-1,
        ),
    }

    best_params: dict[str, dict[str, Any]] = {}
    cv_scores: dict[str, float] = {}
    for name, search in searches.items():
        print(f"Tuning {name}...")
        search.fit(X_train, y_train)
        best_params[name] = search.best_params_
        cv_scores[name] = float(search.best_score_)
        print(f"  best CV accuracy: {search.best_score_:.4f}")
        print(f"  parameters: {search.best_params_}")
    return best_params, cv_scores


def make_model(name: str, params: dict[str, Any]):
    if name == "ANN":
        return MLPClassifier(
            solver="lbfgs",
            max_iter=5000,
            max_fun=50000,
            random_state=RANDOM_STATE,
            **params,
        )
    if name == "SVM":
        return SVC(probability=True, random_state=RANDOM_STATE, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y_test, prediction)),
        "Precision": float(precision_score(y_test, prediction, zero_division=0)),
        "Recall": float(recall_score(y_test, prediction, zero_division=0)),
        "F1 Score": float(f1_score(y_test, prediction, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, probability)),
    }


def train_full_models():
    data = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(data, FEATURES)
    imputer, scaler, X_train_scaled, X_test_scaled = fit_preprocessors(X_train, X_test)

    best_params, cv_scores = tune_models(X_train_scaled, y_train)
    models = {}
    comparison_rows = []
    for name in ["ANN", "SVM", "KNN"]:
        model = make_model(name, best_params[name])
        model.fit(X_train_scaled, y_train)
        models[name] = model
        metrics = evaluate_model(model, X_test_scaled, y_test)
        comparison_rows.append({"Model": name, **metrics})
        print(f"{name} test accuracy: {metrics['Accuracy']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(models["ANN"], MODELS_DIR / "ann_model.pkl")
    joblib.dump(models["SVM"], MODELS_DIR / "svm_model.pkl")
    joblib.dump(models["KNN"], MODELS_DIR / "knn_model.pkl")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)

    raw_data = pd.read_csv(DATA_PATH)
    metadata = {
        "raw_rows": int(len(raw_data)),
        "unique_rows_used": int(len(data)),
        "duplicate_rows_removed": int(raw_data.duplicated().sum()),
        "features": FEATURES,
        "scaler": "MinMaxScaler(feature_range=(0, 1), clip=True)",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "best_parameters": best_params,
        "best_cv_accuracy": cv_scores,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data, best_params, comparison


def run_feature_ablation(data: pd.DataFrame, best_params: dict[str, dict[str, Any]]):
    """Retrain each model after removing one feature at a time."""
    rows = []
    for dropped_feature in [None] + FEATURES:
        selected = [feature for feature in FEATURES if feature != dropped_feature]
        X_train, X_test, y_train, y_test = split_data(data, selected)
        _, _, X_train_scaled, X_test_scaled = fit_preprocessors(X_train, X_test)

        for name in ["ANN", "SVM", "KNN"]:
            model = make_model(name, best_params[name])
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test)
            rows.append(
                {
                    "Model": name,
                    "Dropped Feature": dropped_feature or "None (All 8 Features)",
                    "Features Used": len(selected),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    baseline = (
        results[results["Dropped Feature"] == "None (All 8 Features)"]
        .set_index("Model")["Accuracy"]
        .to_dict()
    )
    results["Accuracy Change"] = results.apply(
        lambda row: row["Accuracy"] - baseline[row["Model"]], axis=1
    )

    tolerance = 1e-12
    results["Effect of Removal"] = np.select(
        [results["Accuracy Change"] < -tolerance, results["Accuracy Change"] > tolerance],
        ["Accuracy decreased", "Accuracy increased"],
        default="Accuracy unchanged",
    )
    results.to_csv(ABLATION_PATH, index=False)
    return results


def train_and_analyse():
    data, best_params, comparison = train_full_models()
    ablation = run_feature_ablation(data, best_params)
    return comparison, ablation

