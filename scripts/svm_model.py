"""
SVM (Support Vector Machine) for Diabetes Prediction

Uses the SAME raw dataset, SAME imputer.pkl and SAME scaler.pkl that were
already fitted and saved by ann_model.py, so ANN / SVM / KNN are all
compared on identical preprocessing and an identical train/test split.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

RAW_PATH = "diabetes.csv"     # original, unprocessed dataset (same folder)
TARGET_COL = "Outcome"
RANDOM_STATE = 42             # must match ann_model.py's split

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------------------------------------------------------
# 1. Load raw data + reuse the SAME imputer/scaler already fitted
#    by ann_model.py (do NOT re-fit new ones here -> that would
#    make SVM's preprocessing inconsistent with ANN's/KNN's)
# ---------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
clean = raw.copy()
for c in cols_with_invalid_zero:
    clean[c] = clean[c].replace(0, np.nan)

X = clean[FEATURE_ORDER]
y = clean[TARGET_COL]

X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)

# ---------------------------------------------------------------
# 2. Train/test split -- SAME settings as ann_model.py
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 3. Hyperparameter tuning
# ---------------------------------------------------------------

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

param_grid = [
    {
        "kernel": ["rbf"],

        "C": [
            0.1,
            0.3,
            1,
            3,
            10,
            30,
            100
        ],

        "gamma": [
            "scale",
            "auto",
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3
        ],

        "class_weight": [
            None,
            "balanced"
        ]
    },

    {
        "kernel": ["linear"],

        "C": [
            0.01,
            0.03,
            0.1,
            0.3,
            1,
            3,
            10
        ],

        "class_weight": [
            None,
            "balanced"
        ]
    }
]

grid = GridSearchCV(
    SVC(
        probability=True,
        random_state=RANDOM_STATE
    ),

    param_grid,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1
)

grid.fit(X_train, y_train)

svm = grid.best_estimator_

print("\nBest SVM hyperparameters:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")
# ---------------------------------------------------------------
# 4. Evaluate on test set
# ---------------------------------------------------------------
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

print("\n===== SVM (Tuned SVC) — Final Results =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 5. Save model for Streamlit deployment
# ---------------------------------------------------------------
joblib.dump(svm, "svm_model.pkl")
print("\nSaved: svm_model.pkl")
