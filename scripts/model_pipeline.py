"""Shared preprocessing helpers for every diabetes model and the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Data" / "diabetes.csv"
MODELS_DIR = ROOT_DIR / "models"
ABLATION_REPORT_PATH = ROOT_DIR / "Data" / "feature_ablation_results.csv"
MODEL_METRICS_PATH = ROOT_DIR / "Data" / "model_metrics.csv"

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
TARGET_COL = "Outcome"
ZERO_AS_MISSING_COLS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
RANDOM_STATE = 42
ORIGINAL_ROW_COUNT = 768


class SVMWithCalibratedProbability:
    """Keep the SVM decision boundary while adding non-deprecated probabilities."""

    def __init__(self, classifier, probability_model):
        self.classifier = classifier
        self.probability_model = probability_model

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.probability_model.predict_proba(X)

    def decision_function(self, X):
        return self.classifier.decision_function(X)


def load_clean_dataset() -> pd.DataFrame:
    """Load the dataset, validate its schema, and remove exact duplicate patients."""
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET_COL]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data = data[required].drop_duplicates().reset_index(drop=True)
    if not set(data[TARGET_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only 0 and 1.")
    return data


def prepare_raw_features(data: pd.DataFrame, features=None) -> pd.DataFrame:
    """Select numeric inputs and convert physiologically impossible zeroes to missing."""
    feature_names = list(features or FEATURES)
    prepared = data[feature_names].copy()
    prepared = prepared.apply(pd.to_numeric, errors="coerce")
    for column in ZERO_AS_MISSING_COLS:
        if column in prepared.columns:
            prepared[column] = prepared[column].replace(0, np.nan)
    return prepared


def split_raw_dataset(data=None, features=None):
    """Split original patients, then add synthetic rows to training only.

    Rows 0-767 are the original Pima records. Any later rows are synthetic and
    may improve training coverage, but they must never enter the held-out test
    set because that would make the reported accuracy misleading.
    """
    clean = load_clean_dataset() if data is None else data.copy()
    feature_names = list(features or FEATURES)
    original = clean.iloc[:ORIGINAL_ROW_COUNT].copy()
    synthetic = clean.iloc[ORIGINAL_ROW_COUNT:].copy()

    X_original = prepare_raw_features(original, feature_names)
    y_original = original[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_original,
        y_original,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_original,
    )

    if not synthetic.empty:
        X_synthetic = prepare_raw_features(synthetic, feature_names)
        y_synthetic = synthetic[TARGET_COL].astype(int)
        X_train = pd.concat([X_train, X_synthetic], axis=0)
        y_train = pd.concat([y_train, y_synthetic], axis=0)

    return X_train, X_test, y_train, y_test


def fit_preprocessors(X_train):
    """Fit median imputation and 0-1 scaling on training data only."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(X_train_imputed)
    return imputer, scaler


def transform_features(X, imputer, scaler):
    """Apply fitted preprocessing and guarantee model inputs stay in 0-1."""
    imputed = imputer.transform(X)
    transformed = scaler.transform(imputed)
    return np.clip(transformed, 0.0, 1.0)


def positive_probability(model, X):
    """Return the class-1 probability for models with different sklearn APIs."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -709, 709)))
    return np.asarray(model.predict(X), dtype=float)


def evaluate_model(model_label, model, X_test, y_test) -> pd.DataFrame:
    """Return one consistently formatted metrics row for one team member's model."""
    prediction = model.predict(X_test)
    return pd.DataFrame([{
        "Model": model_label,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1 Score": f1_score(y_test, prediction, zero_division=0),
    }])


def update_model_metrics(model_label, metrics: pd.DataFrame) -> None:
    """Replace only this model's row while preserving teammates' rows."""
    if MODEL_METRICS_PATH.exists():
        existing = pd.read_csv(MODEL_METRICS_PATH)
        existing = existing[existing["Model"] != model_label]
        metrics = pd.concat([existing, metrics], ignore_index=True)

    order = {"ANN": 0, "KNN": 1, "SVM": 2}
    metrics["_order"] = metrics["Model"].map(order)
    metrics = metrics.sort_values("_order").drop(columns="_order")
    metrics.round(4).to_csv(MODEL_METRICS_PATH, index=False)


def create_model_ablation_report(model_label, data, estimator) -> pd.DataFrame:
    """Measure one model after removing each feature, without training other models."""
    rows = []
    for removed_feature in [None] + FEATURES:
        used_features = [feature for feature in FEATURES if feature != removed_feature]
        X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(
            data=data, features=used_features
        )
        local_imputer, local_scaler = fit_preprocessors(X_train_raw)
        X_train = transform_features(X_train_raw, local_imputer, local_scaler)
        X_test = transform_features(X_test_raw, local_imputer, local_scaler)

        ablation_model = clone(estimator)
        ablation_model.fit(X_train, y_train)
        prediction = ablation_model.predict(X_test)
        rows.append({
            "Model": model_label,
            "Removed Feature": removed_feature or "None (Baseline)",
            "Features Used": len(used_features),
            "Accuracy": accuracy_score(y_test, prediction),
        })

    report = pd.DataFrame(rows)
    baseline = report.loc[
        report["Removed Feature"] == "None (Baseline)", "Accuracy"
    ].iloc[0]
    report["Baseline Accuracy"] = baseline
    report["Accuracy Change"] = report["Accuracy"] - baseline
    return report


def update_ablation_report(model_label, report: pd.DataFrame) -> None:
    """Replace only this model's ablation rows while preserving teammates' rows."""
    if ABLATION_REPORT_PATH.exists():
        existing = pd.read_csv(ABLATION_REPORT_PATH)
        existing = existing[existing["Model"] != model_label]
        report = pd.concat([existing, report], ignore_index=True)

    model_order = {"ANN": 0, "KNN": 1, "SVM": 2}
    feature_order = {"None (Baseline)": 0, **{
        feature: index + 1 for index, feature in enumerate(FEATURES)
    }}
    report["_model_order"] = report["Model"].map(model_order)
    report["_feature_order"] = report["Removed Feature"].map(feature_order)
    report = report.sort_values(["_feature_order", "_model_order"]).drop(
        columns=["_model_order", "_feature_order"]
    )
    report.round(4).to_csv(ABLATION_REPORT_PATH, index=False)
