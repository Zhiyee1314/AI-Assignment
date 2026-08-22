"""
confusion_matrix_knn.py
------------------------
Standalone Streamlit app that shows the Confusion Matrix heatmap for
your KNN model. This is separate from your main app.py -- run it on
its own.

Style:
  - Green-to-yellow color scale
  - Each cell shows percentage (column-normalized) + raw count
  - Y-axis = Predicted Labels, X-axis = Actual Labels
  - Accuracy shown in the title

Requirements:
  pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib

Run from terminal:
  streamlit run confusion_matrix_knn.py

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
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

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
CLASS_LABELS = ["No Diabetes", "Diabetes"]


st.set_page_config(page_title="KNN Confusion Matrix", page_icon="🟩", layout="centered")
st.title("🟩 KNN — Confusion Matrix")
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
          "or edit the path variables at the top of `confusion_matrix_knn.py`."
    )
    st.stop()


@st.cache_data
def compute_confusion_matrix():
    results = pd.read_csv(PREDICTIONS_PATH)
    y_test = results["Actual"]
    y_pred = results["Prediction"]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return cm, acc


with st.spinner("Evaluating KNN model..."):
    cm, acc = compute_confusion_matrix()

# ------------------------------------------------------------------
# Build the annotated, transposed matrix (rows=Predicted, cols=Actual)
# ------------------------------------------------------------------
cm_t = cm.T
col_sums = cm_t.sum(axis=0, keepdims=True)
cm_percent = (cm_t / col_sums) * 100

annot = np.empty_like(cm_t).astype(str)
for i in range(cm_t.shape[0]):
    for j in range(cm_t.shape[1]):
        annot[i, j] = f"{cm_percent[i, j]:.1f}%\n{cm_t[i, j]}"

# ------------------------------------------------------------------
# Plot (green-yellow style, matching the reference)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_t,
    annot=annot,
    fmt='',
    cmap='YlGn_r',
    cbar=True,
    linewidths=1,
    linecolor='black',
    xticklabels=CLASS_LABELS,
    yticklabels=CLASS_LABELS,
    annot_kws={"size": 13},
    ax=ax
)
ax.set_title(f"Accuracy: {acc * 100:.2f}%", fontsize=14, fontweight='bold')
ax.set_xlabel("Actual Labels")
ax.set_ylabel("Predicted Labels")
plt.setp(ax.get_xticklabels(), rotation=0)
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()

st.pyplot(fig, width="content")

# ------------------------------------------------------------------
# Summary metrics + interpretation
# ------------------------------------------------------------------
st.metric("Accuracy", f"{acc:.4f}")
