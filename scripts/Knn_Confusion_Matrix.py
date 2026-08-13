"""
knn_confusion_matrix.py
------------------------
Generates the Confusion Matrix heatmap for your KNN model.

Loads your already-trained knn_model.pkl + the shared imputer.pkl
and scaler.pkl, evaluates on the same test split, and saves the
heatmap as a PNG for your report.

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib

Run:
  python knn_confusion_matrix.py

Required files in the same folder:
  diabetes.csv, imputer.pkl, scaler.pkl, knn_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

RAW_PATH = "diabetes.csv"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
KNN_MODEL_PATH = "knn_model.pkl"
OUTPUT_FILE = "knn_confusion_matrix.png"

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------
    # 1. Load data + apply the SAME shared preprocessing
    # ------------------------------------------------------------
    raw = pd.read_csv(RAW_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    clean = raw.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)

    X = clean[FEATURE_ORDER]
    y = clean[TARGET_COL]

    X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURE_ORDER)

    # ------------------------------------------------------------
    # 2. SAME train/test split as your training script
    # ------------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ------------------------------------------------------------
    # 3. Load model and predict
    # ------------------------------------------------------------
    model = joblib.load(KNN_MODEL_PATH)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ------------------------------------------------------------
    # 4. Plot heatmap
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        annot_kws={"size": 18}
    )
    plt.title(f"KNN Confusion Matrix (Accuracy = {acc:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nRead as:")
    print(f"  True Negatives  (correctly predicted No Diabetes): {cm[0][0]}")
    print(f"  False Positives (wrongly predicted Diabetes)      : {cm[0][1]}")
    print(f"  False Negatives (wrongly predicted No Diabetes)   : {cm[1][0]}")
    print(f"  True Positives  (correctly predicted Diabetes)    : {cm[1][1]}")


if __name__ == "__main__":
    main()
