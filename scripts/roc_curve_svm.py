"""Standalone Streamlit page for the SVM ROC curve and AUC."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, auc, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT_DIR / "Data" / "svm_test_predictions.csv"
REQUIRED_COLUMNS = ["Actual", "Prediction", "Probability"]


st.set_page_config(
    page_title="SVM ROC Curve",
    page_icon="📈",
    layout="centered",
)
st.title("📈 SVM — ROC Curve & AUC")
st.caption(
    "Evaluated using the calibrated probabilities and saved predictions "
    "from the same held-out test patients used by Svm_Model.py."
)


@st.cache_data
def compute_roc():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}. "
            "Run scripts/Svm_Model.py first."
        )

    results = pd.read_csv(PREDICTIONS_PATH)
    missing = [
        column for column in REQUIRED_COLUMNS if column not in results.columns
    ]
    if missing:
        raise ValueError(
            "SVM prediction CSV is missing columns: " + ", ".join(missing)
        )

    values = results[REQUIRED_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if values.isna().any().any():
        raise ValueError("SVM prediction CSV contains missing/non-numeric values.")
    if not values["Actual"].isin([0, 1]).all():
        raise ValueError("SVM Actual values must contain only 0 or 1.")
    if not values["Prediction"].isin([0, 1]).all():
        raise ValueError("SVM Prediction values must contain only 0 or 1.")
    if not values["Probability"].between(0, 1).all():
        raise ValueError("SVM Probability values must be between 0 and 1.")
    if values["Actual"].nunique() != 2:
        raise ValueError("SVM ROC-AUC requires both Outcome classes.")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        values["Actual"],
        values["Probability"],
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)
    accuracy = accuracy_score(values["Actual"], values["Prediction"])
    return false_positive_rate, true_positive_rate, roc_auc, accuracy


try:
    with st.spinner("Evaluating SVM model..."):
        fpr, tpr, roc_auc, accuracy = compute_roc()

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        fpr,
        tpr,
        color="#FF6B6B",
        linewidth=2,
        label=f"SVM (AUC = {roc_auc:.4f})",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        label="Random Guess (AUC = 0.5)",
    )
    axis.set_title("ROC Curve - SVM Model", fontsize=14, fontweight="bold")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    st.pyplot(figure, width="content")
    plt.close(figure)

    left, right = st.columns(2)
    left.metric("AUC Score", f"{roc_auc:.4f}")
    right.metric("Accuracy", f"{accuracy:.4f}")

    if roc_auc >= 0.9:
        interpretation = "Excellent discrimination between classes."
    elif roc_auc >= 0.8:
        interpretation = "Good discrimination between classes."
    elif roc_auc >= 0.7:
        interpretation = "Fair/acceptable discrimination between classes."
    else:
        interpretation = "Poor discrimination—close to random guessing."

    st.info(f"**Interpretation:** AUC = {roc_auc:.4f} → {interpretation}")

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(str(error))

