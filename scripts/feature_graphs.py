"""
feature_graphs.py
------------------
Generates ONE graph per feature (8 total), each showing how that
feature differs between diabetic and non-diabetic patients.

Each graph = a boxplot split by Outcome (No Diabetes vs Diabetes),
which is meaningful because it visually shows whether that feature
is actually useful for predicting diabetes -- not just a random
distribution plot.

Also prints an automatic interpretation for each feature (mean
difference between groups) that you can paraphrase into your report.

Requirements:
  pip install pandas numpy matplotlib seaborn

Run:
  python feature_graphs.py

Output:
  Saves 8 PNG files into a folder called "feature_graphs/"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = "diabetes.csv"
OUTPUT_DIR = "feature_graphs"
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
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(RAW_PATH)

    # Replace invalid zeros with NaN so they don't skew the boxplots,
    # then drop those NaNs just for plotting (not modifying your CSV)
    clean = df.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)

    print("Generating one graph per feature...\n")

    for i, feature in enumerate(FEATURES, start=1):
        plot_df = clean[[feature, TARGET_COL]].dropna()

        mean_no = plot_df[plot_df[TARGET_COL] == 0][feature].mean()
        mean_yes = plot_df[plot_df[TARGET_COL] == 1][feature].mean()
        diff_pct = ((mean_yes - mean_no) / mean_no) * 100 if mean_no != 0 else float('nan')

        plt.figure(figsize=(6, 5))
        sns.boxplot(
            data=plot_df, x=TARGET_COL, y=feature,
            hue=TARGET_COL, palette=["#22B07D", "#E5484D"], legend=False
        )
        plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
        plt.title(f"{feature} by Diabetes Outcome\n({FEATURE_MEANING.get(feature, '')})",
                   fontsize=12, fontweight='bold')
        plt.xlabel("")
        plt.ylabel(feature)
        plt.tight_layout()

        filename = f"{OUTPUT_DIR}/{i}_{feature}.png"
        plt.savefig(filename, dpi=200)
        plt.close()

        direction = "higher" if mean_yes > mean_no else "lower"
        print(f"[{i}/8] Saved -> {filename}")
        print(f"       Mean (No Diabetes) = {mean_no:.2f} | Mean (Diabetes) = {mean_yes:.2f}")
        print(f"       Interpretation: Diabetic patients tend to have {direction} {feature} "
              f"({abs(diff_pct):.1f}% {direction}) than non-diabetic patients.\n")

    print(f"All 8 graphs saved in the '{OUTPUT_DIR}/' folder.")
    print("Use the printed interpretations above as a starting point for your report captions.")


if __name__ == "__main__":
    main()
