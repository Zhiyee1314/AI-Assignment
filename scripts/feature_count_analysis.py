"""
feature_count_analysis.py
---------------------------
Answers the report question: "Why did you choose this many features --
why not more, why not fewer?"

Tests EVERY possible number of features (from 1 up to all 8), using
SequentialFeatureSelector + KNN, and plots how F1 / Accuracy change as
you add more features. This is the same logic as your "Accuracy vs K"
graph, just with "Number of Features" on the X-axis instead of K.

Reuses the SAME shared imputer.pkl / scaler.pkl / FEATURE_ORDER /
RANDOM_STATE as Ann_Model.py, Svm_Model.py, and Knn_Model.py.

Requirements:
  pip install pandas numpy scikit-learn matplotlib joblib

Run:
  python feature_count_analysis.py

Required files in the same folder:
  diabetes.csv, imputer.pkl, scaler.pkl

Output:
  - Prints a table: number of features -> selected features -> CV F1 score
  - Saves 'feature_count_vs_score.png' -- the justification graph for your report
  - Saves 'selected_features.pkl' using the BEST number of features found
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector

RAW_PATH = "diabetes.csv"
TARGET_COL = "Outcome"
RANDOM_STATE = 42

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

OUTPUT_GRAPH = "feature_count_vs_score.png"
OUTPUT_SELECTED_FEATURES = "selected_features.pkl"


def main():
    # ---------------------------------------------------------------
    # 1. Load raw data + reuse the SAME shared imputer/scaler
    # ---------------------------------------------------------------
    raw = pd.read_csv(RAW_PATH)
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    clean = raw.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)

    X = clean[FEATURE_ORDER]
    y = clean[TARGET_COL]

    X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURE_ORDER)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    knn_for_selection = KNeighborsClassifier(n_neighbors=15)

    # ---------------------------------------------------------------
    # 2. Test every possible number of features (1 to 8)
    # ---------------------------------------------------------------
    n_options = range(1, len(FEATURE_ORDER) + 1)
    results = []

    print("Testing every feature count from 1 to 8...\n")

    for n in n_options:
        if n == len(FEATURE_ORDER):
            # No selection needed -- just use all features
            selected = FEATURE_ORDER
        else:
            sfs = SequentialFeatureSelector(
                knn_for_selection,
                n_features_to_select=n,
                direction="forward",
                scoring="f1",
                cv=cv,
                n_jobs=-1
            )
            sfs.fit(X_train, y_train)
            mask = sfs.get_support()
            selected = [str(f) for f in np.array(FEATURE_ORDER)[mask]]

        # Cross-validated F1 score using only the selected features
        scores = cross_val_score(
            knn_for_selection, X_train[selected], y_train,
            cv=cv, scoring="f1", n_jobs=-1
        )
        mean_f1 = scores.mean()

        results.append({
            "n_features": n,
            "selected_features": selected,
            "mean_cv_f1": mean_f1
        })

        print(f"[{n}/8] Features: {selected}")
        print(f"       Mean CV F1: {mean_f1:.4f}\n")

    # ---------------------------------------------------------------
    # 3. Find the best number of features
    # ---------------------------------------------------------------
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["mean_cv_f1"].idxmax()]
    best_n = int(best_row["n_features"])
    best_features = best_row["selected_features"]
    best_score = best_row["mean_cv_f1"]

    print("===== SUMMARY =====")
    print(f"Best number of features: {best_n}")
    print(f"Best features: {best_features}")
    print(f"Best mean CV F1: {best_score:.4f}")

    # ---------------------------------------------------------------
    # 4. Plot: Number of Features vs F1 Score (the justification graph)
    # ---------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["n_features"], results_df["mean_cv_f1"],
              marker='o', color='#2c7be5', linewidth=2)
    plt.scatter([best_n], [best_score], color='red', s=120, zorder=5,
                label=f"Best: {best_n} features (F1 = {best_score:.4f})")
    plt.title("Number of Features vs Cross-Validated F1 Score (KNN)",
               fontsize=13, fontweight='bold')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Mean Cross-Validated F1 Score")
    plt.xticks(list(n_options))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH, dpi=200)
    plt.close()
    print(f"\nSaved graph: {OUTPUT_GRAPH}")

    # ---------------------------------------------------------------
    # 5. Save the best feature set for teammates to reuse
    # ---------------------------------------------------------------
    joblib.dump(best_features, OUTPUT_SELECTED_FEATURES)
    print(f"Saved: {OUTPUT_SELECTED_FEATURES} (features: {best_features})")


if __name__ == "__main__":
    main()
