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
  streamlit run scripts/Knn_Confusion_Matrix.py

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
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

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
CLASS_LABELS = ["No Diabetes", "Diabetes"]


st.set_page_config(page_title="KNN Confusion Matrix", page_icon="🟩", layout="centered")
st.title("🟩 KNN — Confusion Matrix")
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
def compute_confusion_matrix():
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

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)  # cm[i][j] = actual class i, predicted class j

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
