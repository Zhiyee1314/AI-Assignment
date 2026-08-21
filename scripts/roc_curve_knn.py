"""
roc_curve_knn.py
-----------------
Standalone Streamlit app that shows the ROC Curve + AUC for your KNN model.
This is separate from your main app.py -- run it on its own.

Requirements:
  pip install streamlit pandas numpy scikit-learn matplotlib joblib

Run from terminal:
  streamlit run scripts/roc_curve_knn.py

Required repository files:
  Data/diabetes.csv
  models/imputer.pkl, models/scaler.pkl, models/knn_model.pkl
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

# ------------------------------------------------------------------
# Repository paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "Data" / "diabetes.csv"
IMPUTER_PATH = REPO_ROOT / "models" / "imputer.pkl"
SCALER_PATH = REPO_ROOT / "models" / "scaler.pkl"
KNN_MODEL_PATH = REPO_ROOT / "models" / "knn_model.pkl"

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


st.set_page_config(page_title="KNN ROC Curve", page_icon="📈", layout="centered")
st.title("📈 KNN — ROC Curve & AUC")
st.caption("Evaluated on the same 80/20 test split used to train the KNN model.")


def check_files_exist():
    missing = [p for p in [RAW_PATH, IMPUTER_PATH, SCALER_PATH, KNN_MODEL_PATH] if not os.path.exists(p)]
    return missing


missing_files = check_files_exist()
if missing_files:
    st.error(
        "The following required file(s) were not found in this folder:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\nRun the script from a checkout that contains the Data/ and models/ folders."
    )
    st.stop()


@st.cache_data
def compute_roc():
    raw = pd.read_csv(RAW_PATH).drop_duplicates().reset_index(drop=True)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    clean = raw.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)

    X = clean[FEATURE_ORDER]
    y = clean[TARGET_COL]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(KNN_MODEL_PATH)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(y_test, y_pred)

    return fpr, tpr, roc_auc, acc


with st.spinner("Evaluating KNN model..."):
    fpr, tpr, roc_auc, acc = compute_roc()

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color='#2c7be5', linewidth=2, label=f"KNN (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess (AUC = 0.5)")
ax.set_title("ROC Curve - KNN Model", fontsize=14, fontweight='bold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
fig.tight_layout()

st.pyplot(fig, width="content")

# ------------------------------------------------------------------
# Summary metrics + interpretation
# ------------------------------------------------------------------
col1, col2 = st.columns(2)
col1.metric("AUC Score", f"{roc_auc:.4f}")
col2.metric("Accuracy", f"{acc:.4f}")

if roc_auc >= 0.9:
    interpretation = "Excellent discrimination between classes."
elif roc_auc >= 0.8:
    interpretation = "Good discrimination between classes."
elif roc_auc >= 0.7:
    interpretation = "Fair/acceptable discrimination between classes."
else:
    interpretation = "Poor discrimination -- close to random guessing."

st.info(f"**Interpretation:** AUC = {roc_auc:.4f} → {interpretation}")
