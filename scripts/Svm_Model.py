"""
SVM (Support Vector Machine) for Diabetes Prediction

Uses the SAME raw dataset, SAME imputer.pkl and SAME scaler.pkl
created by Ann_Model.py.

This version:
1. Removes StandardScaler feature-name warning
2. Removes deprecated SVC(probability=True)
3. Tunes SVM for ACCURACY
4. Uses decision_function for AUC
5. Saves the best SVM model
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


print("RUNNING FILE:", __file__)


# ===============================================================
# 1. Basic settings
# ===============================================================

RAW_PATH = "diabetes.csv"
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
    "Age"
]


# ===============================================================
# 2. Load dataset
# ===============================================================

raw = pd.read_csv(RAW_PATH)


print("Dataset rows:", len(raw))


# ===============================================================
# 3. Load SAME imputer and scaler created by ANN
# ===============================================================

imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")


# ===============================================================
# 4. Replace invalid zero values
# ===============================================================

cols_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


clean = raw.copy()


for column in cols_with_invalid_zero:
    clean[column] = clean[column].replace(
        0,
        np.nan
    )


X = clean[FEATURE_ORDER]
y = clean[TARGET_COL]


# ===============================================================
# 5. Apply shared preprocessing
# ===============================================================

# IMPORTANT:
# imputer.transform() returns NumPy array
# Keep it as NumPy array.
#
# Do NOT convert it back to DataFrame before scaler.transform().
# This prevents the feature-name warning.

X_imputed = imputer.transform(X)

X_scaled = scaler.transform(
    X_imputed
)


# ===============================================================
# 6. Train / test split
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ===============================================================
# 7. Cross validation
# ===============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ===============================================================
# 8. SVM hyperparameter tuning
# ===============================================================

param_grid = [

    # RBF kernel
    {
        "kernel": [
            "rbf"
        ],

        "C": [
            0.05,
            0.1,
            0.3,
            0.5,
            1,
            2,
            3,
            5,
            10,
            30,
            100
        ],

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
            0.3
        ],

        "class_weight": [
            None,
            "balanced"
        ]
    },


    # Linear kernel
    {
        "kernel": [
            "linear"
        ],

        "C": [
            0.001,
            0.003,
            0.01,
            0.03,
            0.05,
            0.1,
            0.3,
            0.5,
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


# ===============================================================
# 9. Grid Search
# ===============================================================

grid = GridSearchCV(

    estimator=SVC(
        random_state=RANDOM_STATE
    ),

    param_grid=param_grid,

    scoring="accuracy",

    cv=cv,

    n_jobs=-1,

    verbose=1
)


print(
    "\nSearching for best SVM parameters..."
)


grid.fit(
    X_train,
    y_train
)


# ===============================================================
# 10. Best model
# ===============================================================

svm = grid.best_estimator_


print(
    "\n======================================"
)

print(
    "BEST SVM SETTINGS"
)

print(
    "======================================"
)


print(
    "\nBest hyperparameters:"
)

print(
    grid.best_params_
)


print(
    f"\nBest CV accuracy: "
    f"{grid.best_score_:.4f}"
)


# ===============================================================
# 11. Test prediction
# ===============================================================

y_pred = svm.predict(
    X_test
)


# IMPORTANT:
#
# We do NOT use predict_proba().
#
# SVC decision_function gives a continuous score
# that can be used to calculate ROC AUC.

y_score = svm.decision_function(
    X_test
)


# ===============================================================
# 12. Evaluation
# ===============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_score
)


print(
    "\n======================================"
)

print(
    "SVM (Tuned SVC) — FINAL RESULTS"
)

print(
    "======================================"
)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)


print(
    "\nConfusion Matrix:\n",
    confusion_matrix(
        y_test,
        y_pred
    )
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ===============================================================
# 13. Save final SVM
# ===============================================================

joblib.dump(
    svm,
    "svm_model.pkl"
)


print(
    "\nSaved: svm_model.pkl"
)
