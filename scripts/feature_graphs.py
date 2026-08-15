"""
feature_graphs_app.py
----------------------
Standalone Streamlit app that shows ONE graph per feature (8 total),
each showing how that feature differs between diabetic and
non-diabetic patients (boxplot split by Outcome).

This is separate from your main app.py -- run it on its own.

Requirements:
  pip install streamlit pandas numpy matplotlib seaborn

Run from terminal:
  streamlit run feature_graphs_app.py

Required file in the same folder:
  diabetes.csv (or Data/diabetes.csv, see RAW_PATH below)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------
# Config -- adjust this path if diabetes.csv sits in a different folder
# ------------------------------------------------------------------
RAW_PATH = "diabetes.csv"
TARGET_COL = "Outcome"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Columns where 0 is not physiologically possible -> treat as missing
# (excluded from the raw 0s so they don't distort the boxplot)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Short, plain-English meaning for each feature -- used in graph titles
FEATURE_MEANING = {
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose concentration (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mmHg)",
    "SkinThickness": "Triceps skin fold thickness (mm)",
    "Insulin": "2-Hour serum insulin (mu U/ml)",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Genetic diabetes likelihood score",
    "Age": "Age (years)",
}

sns.set_style("whitegrid")

st.set_page_config(page_title="Feature Graphs", page_icon="📊", layout="wide")
st.title("📊 Feature Distributions by Diabetes Outcome")
st.caption(
    "Each graph shows whether a feature actually differs between diabetic "
    "and non-diabetic patients -- a bigger visible gap between the two boxes "
    "means that feature is likely a stronger predictor."
)

if not os.path.exists(RAW_PATH):
    st.error(
        f"Could not find `{RAW_PATH}`. Make sure this script sits in the same "
        f"folder as your dataset, or edit RAW_PATH at the top of "
        f"`feature_graphs_app.py`."
    )
    st.stop()


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(RAW_PATH)
    clean = df.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)
    return clean


def plot_feature(clean_df, feature):
    plot_df = clean_df[[feature, TARGET_COL]].dropna()

    mean_no = plot_df[plot_df[TARGET_COL] == 0][feature].mean()
    mean_yes = plot_df[plot_df[TARGET_COL] == 1][feature].mean()
    diff_pct = ((mean_yes - mean_no) / mean_no) * 100 if mean_no != 0 else float('nan')
    direction = "higher" if mean_yes > mean_no else "lower"

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.boxplot(
        data=plot_df, x=TARGET_COL, y=feature,
        hue=TARGET_COL, palette=["#22B07D", "#E5484D"], legend=False, ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_title(f"{feature}\n({FEATURE_MEANING.get(feature, '')})", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel(feature)
    fig.tight_layout()

    return fig, mean_no, mean_yes, diff_pct, direction


def plot_mean_comparison_bar(feature, mean_no, mean_yes):
    """Small bar chart comparing the two group means, with value labels,
    replacing the plain text caption with a visual."""
    fig, ax = plt.subplots(figsize=(5, 2.6))
    labels = ["No Diabetes", "Diabetes"]
    values = [mean_no, mean_yes]
    colors = ["#22B07D", "#E5484D"]

    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    ax.set_title(f"Mean {feature} Comparison", fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Value")
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig


clean_df = load_and_clean_data()

# Display 2 graphs per row
for row_start in range(0, len(FEATURES), 2):
    cols = st.columns(2)
    for col, feature in zip(cols, FEATURES[row_start:row_start + 2]):
        with col:
            fig, mean_no, mean_yes, diff_pct, direction = plot_feature(clean_df, feature)
            st.pyplot(fig, use_container_width=True)

            bar_fig = plot_mean_comparison_bar(feature, mean_no, mean_yes)
            st.pyplot(bar_fig, use_container_width=True)

            st.caption(
                f"Diabetic patients tend to have **{direction} {feature}** "
                f"({abs(diff_pct):.1f}% {direction})."
            )
            st.divider()
