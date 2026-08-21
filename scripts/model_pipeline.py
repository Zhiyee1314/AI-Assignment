"""Shared preprocessing helpers for every diabetes model and the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
