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
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)
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
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

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
        (12, 6),
        (16, 8),
        (20, 10),
        (24, 12),
        (32, 16)
    ],

    "activation": [
        "relu",
        "tanh"
    ],

    "alpha": [
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30
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
# Find the best classification threshold using TRAINING data only
# ------------------------------------------------------------------

cv_prob = cross_val_predict(
    model,
    X_train_scaled,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

best_threshold = 0.50
best_threshold_accuracy = 0

for threshold in np.arange(0.30, 0.71, 0.01):

    cv_pred = (cv_prob >= threshold).astype(int)

    threshold_accuracy = accuracy_score(
        y_train,
        cv_pred
    )

    if threshold_accuracy > best_threshold_accuracy:
        best_threshold_accuracy = threshold_accuracy
        best_threshold = threshold

print(
    f"Best probability threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Threshold CV accuracy: "
    f"{best_threshold_accuracy:.4f}"
)

# ------------------------------------------------------------------
# 6. Evaluate ANN
# ------------------------------------------------------------------

y_prob = model.predict_proba(X_test_scaled)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)

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
#   svm_model = SVC(probability=True, random_state=42)
#   svm_model.fit(X_train_scaled, y_train)
#   joblib.dump(svm_model, "svm_model.pkl")
#
# This guarantees all 3 models see numbers on the same scale, trained
# on the same missing-value treatment, so accuracy/probability
# comparisons between ANN vs SVM vs KNN are actually fair.
# ------------------------------------------------------------------
