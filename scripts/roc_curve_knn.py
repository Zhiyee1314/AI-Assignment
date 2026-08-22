"""
roc_curve_knn.py
-----------------
Standalone Streamlit app that shows the ROC Curve + AUC for your KNN model.
This is separate from your main app.py -- run it on its own.

Requirements:
  pip install streamlit pandas numpy scikit-learn matplotlib joblib

Run from terminal:
  streamlit run roc_curve_knn.py

Required files in the same folder:
  diabetes.csv (or Data/diabetes.csv, see RAW_PATH below)
  imputer.pkl, scaler.pkl, knn_model.pkl
  (or models/imputer.pkl, models/scaler.pkl, models/knn_model.pkl)
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
# Config -- adjust these paths if your files sit in different folders
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT_DIR / "Data" / "knn_test_predictions.csv"

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
    missing = [p for p in [PREDICTIONS_PATH] if not os.path.exists(p)]
    return missing


missing_files = check_files_exist()
if missing_files:
    st.error(
        "The following required file(s) were not found in this folder:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\nMake sure this script sits in the same folder as your dataset and .pkl files, "
          "or edit the path variables at the top of `roc_curve_knn.py`."
    )
    st.stop()


@st.cache_data
def compute_roc():
    results = pd.read_csv(PREDICTIONS_PATH)
    y_test = results["Actual"]
    y_pred = results["Prediction"]
    y_prob = results["Probability"]

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
