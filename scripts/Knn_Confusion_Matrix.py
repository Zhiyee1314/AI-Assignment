"""
knn_confusion_matrix.py
------------------------
Generates the Confusion Matrix heatmap for your KNN model, styled to
match a reference format:
  - Green-to-yellow color scale
  - Each cell shows percentage (column-normalized) + raw count
  - Y-axis = Predicted Labels, X-axis = Actual Labels
  - Accuracy shown in the title

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
CLASS_LABELS = ["No Diabetes", "Diabetes"]


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

    # cm[i][j] = actual class i, predicted class j
    cm = confusion_matrix(y_test, y_pred)

    # Transpose so rows = Predicted, columns = Actual (matches reference image)
    cm_t = cm.T

    # Column-normalized percentages (each column, i.e. each Actual class, sums to 100%)
    col_sums = cm_t.sum(axis=0, keepdims=True)
    cm_percent = (cm_t / col_sums) * 100

    # Build annotation text: "xx.x%\ncount" per cell
    annot = np.empty_like(cm_t).astype(str)
    for i in range(cm_t.shape[0]):
        for j in range(cm_t.shape[1]):
            annot[i, j] = f"{cm_percent[i, j]:.1f}%\n{cm_t[i, j]}"

    # ------------------------------------------------------------
    # 4. Plot heatmap (green-yellow style, matching the reference)
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm_t,
        annot=annot,
        fmt='',
        cmap='YlGn_r',
        cbar=True,
        linewidths=1,
        linecolor='black',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        annot_kws={"size": 13}
    )

    plt.title(f"Accuracy: {acc * 100:.2f}%", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=Predicted, columns=Actual):")
    print(cm_t)


if __name__ == "__main__":
    main()
