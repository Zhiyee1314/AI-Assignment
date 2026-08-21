"""
train_ann.py
------------
This is the script that PRODUCES the .pkl files your app.py loads.
Run this once locally after you finish training — it saves 3 files into
the same folder: imputer.pkl, scaler.pkl, ann_model.pkl

Your teammates copy THIS SAME imputer.pkl + scaler.pkl into their own
training script (see the note near the bottom) so all 3 models are
trained on identical preprocessing.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_pipeline import (
    MODELS_DIR,
    RANDOM_STATE,
    create_model_ablation_report,
    evaluate_model,
    fit_preprocessors,
    load_clean_dataset,
    split_raw_dataset,
    transform_features,
    update_ablation_report,
    update_model_metrics,
)
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
# 1. Load data and use the shared split
# ------------------------------------------------------------------
data = load_clean_dataset()

X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(
    data=data
)

# ANN creates the shared imputer and 0–1 scaler.
imputer, scaler = fit_preprocessors(X_train_raw)

X_train_scaled = transform_features(
    X_train_raw, imputer, scaler
)

X_test_scaled = transform_features(
    X_test_raw, imputer, scaler
)

print("Minimum normalized value:", X_train_scaled.min())
print("Maximum normalized value:", X_train_scaled.max())

# ------------------------------------------------------------------
# 5. Tune and train ANN model
# ------------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

ann = MLPClassifier(
    solver="lbfgs",
    max_iter=5000,
    max_fun=50000,
    random_state=42
)

param_grid = {
    "hidden_layer_sizes": [
        (8,),
        (16,),
        (32,),
        (16, 8),
        (32, 16),
        (64, 32)
    ],

    "activation": [
        "relu",
        "tanh"
    ],

    "alpha": [
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1
    ]
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
# Save ANN metrics without deleting KNN/SVM results
# ------------------------------------------------------------------
metrics = evaluate_model(
    "ANN", model, X_test_scaled, y_test
)

update_model_metrics("ANN", metrics)

# ------------------------------------------------------------------
# ANN leave-one-feature-out test
# ------------------------------------------------------------------
ablation = create_model_ablation_report(
    "ANN", data, model
)

update_ablation_report("ANN", ablation)

# ------------------------------------------------------------------
# 7. Save all 3 files with joblib (NOT plain pickle — joblib handles
#    numpy arrays inside sklearn objects more efficiently, and it's
#    what your app.py already expects via joblib.load(...))
# ------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(model, MODELS_DIR / "ann_model.pkl")


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
#   svm_model = SVC(probability=True, random_state=42)
#   svm_model.fit(X_train_scaled, y_train)
#   joblib.dump(svm_model, "svm_model.pkl")
#
# This guarantees all 3 models see numbers on the same scale, trained
# on the same missing-value treatment, so accuracy/probability
# comparisons between ANN vs SVM vs KNN are actually fair.
# ------------------------------------------------------------------
