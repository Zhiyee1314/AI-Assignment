"""
SVM (Support Vector Machine) for Diabetes Prediction.

This file trains and evaluates only SVM. It uses the shared imputer and
scaler expected by the Streamlit application, tunes SVC on the training set,
and calibrates the selected SVC so svm_model.pkl provides predict_proba().
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC


print("RUNNING FILE:", __file__)


# ===============================================================
# 1. Paths and settings
# ===============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "Data" / "diabetes.csv"
MODELS_DIR = ROOT_DIR / "models"
IMPUTER_FILE = MODELS_DIR / "imputer.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
SVM_FILE = MODELS_DIR / "svm_model.pkl"

TARGET_COL = "Outcome"
RANDOM_STATE = 42

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# ===============================================================
# 2. Load and validate the dataset
# ===============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

raw = pd.read_csv(DATA_FILE)
required_columns = FEATURE_ORDER + [TARGET_COL]
missing_columns = [column for column in required_columns if column not in raw.columns]

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns: " + ", ".join(missing_columns)
    )

if raw[TARGET_COL].isna().any():
    raise ValueError("Outcome contains missing values.")

invalid_outcomes = set(raw[TARGET_COL].unique()) - {0, 1}
if invalid_outcomes:
    raise ValueError(
        "Outcome must contain only 0 and 1. Invalid values: "
        + ", ".join(map(str, sorted(invalid_outcomes)))
    )

print("Dataset rows:", len(raw))


# ===============================================================
# 3. Load the shared preprocessing artifacts
# ===============================================================

if not IMPUTER_FILE.exists() or not SCALER_FILE.exists():
    raise FileNotFoundError(
        "Run the shared preprocessing/training setup first. Required files: "
        f"{IMPUTER_FILE} and {SCALER_FILE}"
    )

imputer = joblib.load(IMPUTER_FILE)
scaler = joblib.load(SCALER_FILE)


# ===============================================================
# 4. Clean invalid physiological zero values
# ===============================================================

clean = raw.copy()

for column in ZERO_AS_MISSING_COLS:
    clean[column] = clean[column].replace(0, np.nan)

X = clean[FEATURE_ORDER]
y = clean[TARGET_COL].astype(int)


# ===============================================================
# 5. Apply the same saved preprocessing used by the app
# ===============================================================

# Keep arrays as NumPy arrays to avoid feature-name warnings when the
# preprocessing artifacts were originally fitted without DataFrame names.
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)


# ===============================================================
# 6. Fair, reproducible train/test split
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ===============================================================
# 7. Training-only cross-validation
# ===============================================================

search_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ===============================================================
# 8. SVM hyperparameter tuning
# ===============================================================

param_grid = [
    {
        "kernel": ["rbf"],
        "C": [0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30, 100],
        "gamma": [
            "scale",
            "auto",
            0.001,
            0.003,
            0.005,
            0.01,
            0.03,
            0.05,
            0.1,
            0.3,
        ],
        "class_weight": [None, "balanced"],
    },
    {
        "kernel": ["linear"],
        "C": [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 10],
        "class_weight": [None, "balanced"],
    },
]

grid = GridSearchCV(
    estimator=SVC(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring="accuracy",
    cv=search_cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)

print("\nSearching for best SVM parameters...")
grid.fit(X_train, y_train)

print("\n======================================")
print("BEST SVM SETTINGS")
print("======================================")
print("\nBest hyperparameters:")
print(grid.best_params_)
print(f"\nBest CV accuracy: {grid.best_score_:.4f}")


# ===============================================================
# 9. Calibrate the best SVC using TRAINING DATA ONLY
# ===============================================================

# CalibratedClassifierCV supplies a scientifically meaningful predict_proba()
# without enabling SVC(probability=True). Its internal calibration folds use
# only X_train/y_train; the unseen X_test is not used for tuning or calibration.
calibration_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

svm = CalibratedClassifierCV(
    estimator=grid.best_estimator_,
    method="sigmoid",
    cv=calibration_cv,
    n_jobs=-1,
)

print("\nCalibrating SVM probabilities using training folds...")
svm.fit(X_train, y_train)


# ===============================================================
# 10. Final unseen-test evaluation
# ===============================================================

y_pred = svm.predict(X_test)
y_probability = svm.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_probability)

print("\n======================================")
print("SVM (TUNED + CALIBRATED) — FINAL RESULTS")
print("======================================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))


# ===============================================================
# 11. Save the probability-capable final SVM
# ===============================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(svm, SVM_FILE)

# Reload and verify the exact artifact that Streamlit will use.
saved_svm = joblib.load(SVM_FILE)

if not hasattr(saved_svm, "predict_proba"):
    raise RuntimeError("Saved SVM does not provide predict_proba().")

verification_probability = saved_svm.predict_proba(X_test[:1])[:, 1]

print("\nSaved:", SVM_FILE)
print("Probability support: True")
print(
    "Verification diabetes probability:",
    f"{float(verification_probability[0]):.6f}",
)
