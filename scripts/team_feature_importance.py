"""
team_feature_importance.py
-----------------------------
Answers the report question: "Which features are most important /
increase accuracy the most?" -- shown for ALL 3 team models (ANN, SVM,
KNN), not just KNN.

Method: Permutation Importance
  For each feature, its values are randomly shuffled and the drop in
  accuracy is measured. A LARGER drop means that feature was MORE
  important to the model's predictions.

Why this method (and not something else):
  KNN has no built-in importance score (unlike Random Forest's
  feature_importances_ or SVM's linear coef_). Permutation importance
  works with ANY trained model, so it's the correct choice for a fair,
  model-agnostic comparison across ANN / SVM / KNN.

NOTE: the ANN/SVM/KNN models used HERE are quick, default-parameter
versions purely for this comparison -- they are NOT your teammates'
final tuned models (Ann_Model.py / Svm_Model.py / Knn_Model.py still
do their own separate GridSearchCV tuning).

Reuses the SAME shared imputer.pkl / scaler.pkl / FEATURE_ORDER /
RANDOM_STATE as the team's training scripts.

Requirements:
  pip install pandas numpy scikit-learn matplotlib joblib

Run:
  python team_feature_importance.py

Required files in the same folder:
  diabetes.csv, imputer.pkl, scaler.pkl

Output:
  - Prints an importance table per model
  - Saves 'team_feature_importance.png' (grouped bar chart)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT_DIR / "Data" / "diabetes.csv"
TARGET_COL = "Outcome"

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")
RANDOM_STATE = 42

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

OUTPUT_GRAPH = "team_feature_importance.png"
N_REPEATS = 30  # how many times each feature is shuffled, for a stable average


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

    # ---------------------------------------------------------------
    # 2. Train quick default versions of all 3 models
    # ---------------------------------------------------------------
    models = {
        "ANN": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }

    importance_results = {}

    print("Computing permutation importance for ANN, SVM, and KNN...\n")

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            scoring="accuracy",
            n_jobs=-1
        )

        importance_results[model_name] = result.importances_mean

        print(f"----- {model_name} -----")
        ranked = sorted(zip(FEATURE_ORDER, result.importances_mean), key=lambda x: x[1], reverse=True)
        for feature, importance in ranked:
            print(f"  {feature:<28} -> {importance:+.4f}")
        print()

    # ---------------------------------------------------------------
    # 3. Combine into one table, ranked by team-average importance
    # ---------------------------------------------------------------
    importance_df = pd.DataFrame(importance_results, index=FEATURE_ORDER)
    importance_df["Team Average"] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values("Team Average", ascending=False)

    print("===== SUMMARY (ranked by team-average importance) =====")
    print(importance_df.round(4).to_string())

    top_feature = importance_df.index[0]
    print(f"\nMost important feature overall: {top_feature}")

    # ---------------------------------------------------------------
    # 4. Plot: grouped bar chart, one group per feature, 3 bars each
    # ---------------------------------------------------------------
    features_sorted = importance_df.index.tolist()
    x = np.arange(len(features_sorted))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, importance_df.loc[features_sorted, "ANN"], width, label="ANN", color="#6C63FF")
    ax.bar(x, importance_df.loc[features_sorted, "SVM"], width, label="SVM", color="#FF6B6B")
    ax.bar(x + width, importance_df.loc[features_sorted, "KNN"], width, label="KNN", color="#22B07D")

    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=30, ha='right')
    ax.set_ylabel("Permutation Importance (Accuracy Drop)")
    ax.set_title("Feature Importance by Model (Permutation Importance)", fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUTPUT_GRAPH, dpi=200)
    plt.close()

    print(f"\nSaved graph: {OUTPUT_GRAPH}")


if __name__ == "__main__":
    main()
