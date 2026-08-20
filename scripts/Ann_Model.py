"""
train_ann.py
------------
This is the script that PRODUCES the .pkl files your app.py loads.
Run this once locally after you finish training — it saves 3 files into
the same folder: imputer.pkl, scaler.pkl, ann_model.pkl

Your teammates copy THIS SAME imputer.pkl + scaler.pkl into their own
training script so all 3 models are trained on identical preprocessing.

CHANGES from your original version:
  1. RepeatedStratifiedKFold (5 splits x 3 repeats) instead of plain
     5-fold -> steadier CV estimate for grid search.
  2. Wider hidden_layer_sizes grid + alpha range.
  3. Added "adam" as a second solver option alongside "lbfgs" (adam
     often generalizes slightly better on small tabular data), with
     early_stopping so it doesn't overfit while searching.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------------------------------------------
# 1. Load your dataset
# ------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")  # <-- change to your actual CSV path

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"  # <-- change if your label column has a different name

X = df[FEATURE_ORDER].copy()
y = df[TARGET].copy()

# ------------------------------------------------------------------
# 2. Treat 0 as missing for these columns (0 is not physiologically
#    possible for these, so it really means "not recorded")
# ------------------------------------------------------------------
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in ZERO_AS_MISSING_COLS:
    X[c] = X[c].replace(0, np.nan)

# ------------------------------------------------------------------
# 3. Train/test split BEFORE fitting imputer/scaler (avoid leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. Fit imputer + scaler on the TRAIN split only, then transform both
#    -> These two objects are what gets shared with teammates
# ------------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
# Wrap back into DataFrames with column names preserved -- SimpleImputer
# and StandardScaler both return plain numpy arrays, so without this,
# the scaler gets fit WITHOUT feature names. Later, when app.py or
# svm_model.py calls scaler.transform() on a DataFrame (which DOES have
# names), sklearn raises the "X has feature names, but StandardScaler
# was fitted without feature names" warning. Keeping DataFrames end to
# end avoids that mismatch entirely.
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_ORDER)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_ORDER)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=FEATURE_ORDER)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=FEATURE_ORDER)

# ------------------------------------------------------------------
# 5. Tune and train ANN model
# ------------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

ann = MLPClassifier(
    max_iter=20000,           # raised from 5000 -- this is what was causing the
                              # "lbfgs failed to converge" warnings on the larger
                              # (64,32)-style networks. lbfgs is a batch solver;
                              # it needs a high iteration ceiling to actually finish.
    max_fun=100000,
    early_stopping=True,     # only kicks in for solver="adam"; ignored by lbfgs
    n_iter_no_change=20,
    random_state=42
)

param_grid = {
    # Trimmed the largest networks (64,32 / 64,32,16). With only ~614 training
    # rows, those have far more parameters than data can support -- they were
    # both the slowest to converge AND the most prone to overfitting the CV
    # folds (hence the CV-vs-test gap you saw: 0.822 CV vs 0.78 test).
    "hidden_layer_sizes": [
        (8,),
        (16,),
        (32,),
        (16, 8),
        (32, 16),
        (32, 16, 8),
    ],

    "activation": [
        "relu",
        "tanh"
    ],

    "alpha": [
        0.001,
        0.01,
        0.1,
        0.3,
        1.0,
        3.0,
    ],

    "solver": ["lbfgs", "adam"],
}

grid = GridSearchCV(
    ann,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

model = grid.best_estimator_

print("\nBest ANN parameters:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# ------------------------------------------------------------------
# 6. Evaluate ANN
# ------------------------------------------------------------------

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n===== ANN (Tuned MLP) — Final Results =====")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# 7. Save all 3 files with joblib (NOT plain pickle — joblib handles
#    numpy arrays inside sklearn objects more efficiently, and it's
#    what your app.py already expects via joblib.load(...))
# ------------------------------------------------------------------
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "ann_model.pkl")

print("Saved: imputer.pkl, scaler.pkl, ann_model.pkl")

# ------------------------------------------------------------------
# NOTE FOR TEAMMATES (SVM / KNN):
# Don't re-fit your own imputer/scaler. Instead:
#
#   imputer = joblib.load("imputer.pkl")   # <- the file you received
#   scaler  = joblib.load("scaler.pkl")    # <- the file you received
#
#   X_train_imputed = imputer.transform(X_train)   # transform, NOT fit_transform
#   X_train_scaled  = scaler.transform(X_train_imputed)
#
# This guarantees all 3 models see numbers on the same scale, trained
# on the same missing-value treatment, so accuracy/probability
# comparisons between ANN vs SVM vs KNN are actually fair.
# ------------------------------------------------------------------
