"""
feature_count_analysis.py
---------------------------
Answers the report question: "Why did you choose this many features --
why not more, why not fewer?" -- in a way that is FAIR to all 3 team
models (ANN, SVM, KNN), not just KNN.
 
Method:
  1. Filter method (correlation with Outcome) is used to RANK features.
     This step is model-agnostic -- it doesn't favor any one algorithm,
     so it's a fair starting point for the whole team.
  2. For each possible number of features (1 to 8), take the TOP-N
     ranked features and test them on quick default versions of ALL
     THREE models (ANN, SVM, KNN) using cross-validation.
  3. Plot THREE lines on one graph (ANN / SVM / KNN), so you can show
     the chosen feature count works well across all three algorithms,
     not just yours.
 
NOTE: the ANN/SVM/KNN models used HERE are quick, default-parameter
versions purely for this feature-count comparison -- they are NOT
your teammates' final tuned models (Ann_Model.py / Svm_Model.py /
Knn_Model.py still do their own separate GridSearchCV tuning on
whichever final feature set the team agrees on).
 
Reuses the SAME shared imputer.pkl / scaler.pkl / FEATURE_ORDER /
RANDOM_STATE as the team's training scripts.
 
Requirements:
  pip install pandas numpy scikit-learn matplotlib joblib
 
Run:
  python team_feature_selection.py
 
Required files in the same folder:
  diabetes.csv, imputer.pkl, scaler.pkl
 
Output:
  - Prints a table: number of features -> features used -> F1 per model
  - Saves 'team_feature_count_vs_score.png' -- the justification graph
  - Saves 'selected_features.pkl' -- the recommended feature set to share
"""
 
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
 
RAW_PATH = "diabetes.csv"
TARGET_COL = "Outcome"
RANDOM_STATE = 42
 
FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
 
OUTPUT_GRAPH = "team_feature_count_vs_score.png"
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
 
    # ---------------------------------------------------------------
    # 2. STEP 1: Filter method -- rank features by correlation with
    #    Outcome (model-agnostic, fair starting point for the team)
    # ---------------------------------------------------------------
    corr_df = X_imputed.copy()
    corr_df[TARGET_COL] = y.values
    correlations = corr_df.corr()[TARGET_COL].drop(TARGET_COL).abs()
    ranked_features = correlations.sort_values(ascending=False).index.tolist()
 
    print("===== Feature ranking by |correlation| with Outcome =====")
    print(correlations.reindex(ranked_features).round(4).to_string())
    print()
 
    # ---------------------------------------------------------------
    # 3. STEP 2: For each N, test top-N features on ALL 3 model types
    # ---------------------------------------------------------------
    models = {
        "ANN": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }
 
    n_options = range(1, len(FEATURE_ORDER) + 1)
    results = []
 
    print("Testing every feature count (1 to 8) on ANN, SVM, and KNN...\n")
 
    for n in n_options:
        selected = ranked_features[:n]
        row = {"n_features": n, "features": selected}
 
        for model_name, model in models.items():
            scores = cross_val_score(
                model, X_train[selected], y_train,
                cv=cv, scoring="f1", n_jobs=-1
            )
            row[model_name] = scores.mean()
 
        results.append(row)
        print(f"[{n}/8] Features: {selected}")
        print(f"       ANN F1: {row['ANN']:.4f} | SVM F1: {row['SVM']:.4f} | KNN F1: {row['KNN']:.4f}\n")
 
    results_df = pd.DataFrame(results)
 
    # ---------------------------------------------------------------
    # 4. Find the best N based on the AVERAGE of all 3 models
    #    (fair to the whole team, not biased toward one model)
    # ---------------------------------------------------------------
    results_df["team_avg_f1"] = results_df[["ANN", "SVM", "KNN"]].mean(axis=1)
    best_row = results_df.loc[results_df["team_avg_f1"].idxmax()]
    best_n = int(best_row["n_features"])
    best_features = best_row["features"]
    best_avg = best_row["team_avg_f1"]
 
    print("===== SUMMARY =====")
    print(f"Best number of features (averaged across ANN/SVM/KNN): {best_n}")
    print(f"Selected features: {best_features}")
    print(f"Team-average F1 at this point: {best_avg:.4f}")
 
    # ---------------------------------------------------------------
    # 5. Plot: Number of Features vs F1 Score, ONE LINE PER MODEL
    # ---------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(results_df["n_features"], results_df["ANN"], marker='o', label="ANN", color="#6C63FF", linewidth=2)
    plt.plot(results_df["n_features"], results_df["SVM"], marker='s', label="SVM", color="#FF6B6B", linewidth=2)
    plt.plot(results_df["n_features"], results_df["KNN"], marker='^', label="KNN", color="#22B07D", linewidth=2)
    plt.plot(results_df["n_features"], results_df["team_avg_f1"], marker='D', label="Team Average",
              color="black", linewidth=2, linestyle='--')
 
    plt.axvline(x=best_n, color='red', linestyle=':', alpha=0.6)
    plt.scatter([best_n], [best_avg], color='red', s=150, zorder=5,
                label=f"Best: {best_n} features (avg F1 = {best_avg:.4f})")
 
    plt.title("Number of Features vs F1 Score (ANN, SVM, KNN)", fontsize=13, fontweight='bold')
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
    # 6. Save the recommended feature set for the whole team to share
    # ---------------------------------------------------------------
    joblib.dump(best_features, OUTPUT_SELECTED_FEATURES)
    print(f"Saved: {OUTPUT_SELECTED_FEATURES} (features: {best_features})")
    print("\nShare this file + the graph with your teammates as justification")
    print("for the final feature set used across ANN / SVM / KNN.")
 
 
if __name__ == "__main__":
    main()
