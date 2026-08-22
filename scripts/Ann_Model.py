"""
KNN (K-Nearest Neighbors) for Diabetes Prediction

Uses the SAME raw dataset, SAME imputer.pkl and SAME scaler.pkl that were
already fitted and saved by ann_model.py, so ANN / SVM / KNN are all
compared on identical preprocessing and an identical train/test split.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "models"

RAW_PATH = DATA_DIR / "diabetes.csv"
IMPUTER_FILE = MODELS_DIR / "imputer.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
KNN_MODEL_FILE = MODELS_DIR / "knn_model.pkl"

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_PATH}"
    )

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "Outcome"
RANDOM_STATE = 42             # must match ann_model.py's split

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------------------------------------------------------
# 1. Load raw data + reuse the SAME imputer/scaler already fitted
#    by ann_model.py (do NOT re-fit new ones here -> that would
#    make KNN's preprocessing inconsistent with ANN's/SVM's)
# ---------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
imputer = joblib.load(
    MODELS_DIR / "imputer.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
clean = raw.copy()
for c in cols_with_invalid_zero:
    clean[c] = clean[c].replace(0, np.nan)

X = clean[FEATURE_ORDER]
y = clean[TARGET_COL]

X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURE_ORDER)

# ---------------------------------------------------------------
# 2. Train/test split -- SAME settings as ann_model.py
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 3. Hyperparameter tuning
#    n_neighbors (K) controls the decision boundary. weights determines
#    whether closer neighbors have higher influence, and metric defines
#    the distance calculation algorithm.
# ---------------------------------------------------------------
param_grid = {
    "n_neighbors": range(1, 31),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid, cv=5, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
knn = grid.best_estimator_
print("Best hyperparameters:", grid.best_params_)

# ---------------------------------------------------------------
# 4. Evaluate on test set
# ---------------------------------------------------------------
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]

print("\n===== KNN (Tuned) — Final Results =====")
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
joblib.dump(
    knn,
    KNN_MODEL_FILE
)

print(f"\nSaved: {KNN_MODEL_FILE}")
