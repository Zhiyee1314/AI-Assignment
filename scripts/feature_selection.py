"""
feature_selection.py
---------------------
Performs feature selection for the group's Diabetes Prediction project.

Combines TWO methods:
  1. Filter method   -> correlation of each feature with Outcome
  2. Wrapper method   -> SequentialFeatureSelector using KNN

IMPORTANT NOTE ON METHOD CHOICE:
  RFE (Recursive Feature Elimination) is NOT used here because it
  requires the model to expose coef_ or feature_importances_, which
  KNN does not have. SequentialFeatureSelector works with ANY estimator
  (including KNN) by repeatedly training + cross-validating, so it is
  the correct wrapper method to use for a KNN-based selection.

Reuses the SAME shared imputer.pkl / scaler.pkl / FEATURE_ORDER /
RANDOM_STATE / train-test split settings as Ann_Model.py, Svm_Model.py,
and Knn_Model.py, so the selected feature subset is valid for comparing
ANN / SVM / KNN fairly -- not just for your own KNN model.

Requirements:
  pip install pandas numpy scikit-learn joblib

Run:
  python feature_selection.py

Required files in the same folder:
  diabetes.csv, imputer.pkl, scaler.pkl

Output:
  - Prints the correlation table (filter method)
  - Prints the selected features (wrapper method)
  - Saves 'selected_features.pkl' -- share this file with your teammates
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# How many features to keep in the wrapper method step.
# Adjust this if you want a smaller/larger subset.
N_FEATURES_TO_SELECT = 5


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

    # ---------------------------------------------------------------
    # STEP 1: Filter method -- correlation with Outcome
    # (computed on imputed, UNSCALED data so correlation values stay
    # interpretable in original feature units)
    # ---------------------------------------------------------------
    corr_df = X_imputed.copy()
    corr_df[TARGET_COL] = y.values
    correlations = corr_df.corr()[TARGET_COL].drop(TARGET_COL)
    correlations_sorted = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    print("===== STEP 1: Filter Method (Correlation with Outcome) =====")
    print(correlations_sorted.round(4).to_string())
    print()
    print("Interpretation: features with correlation closer to +1 or -1 have")
    print("a stronger linear relationship with diabetes Outcome. Values near 0")
    print("suggest the feature is weakly related on its own.\n")

    # ---------------------------------------------------------------
    # STEP 2: Wrapper method -- Sequential Feature Selector using KNN
    # ---------------------------------------------------------------
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURE_ORDER)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # A reasonable fixed K just for the selection process itself
    # (this is NOT your final tuned KNN model -- that still comes from
    # Knn_Model.py's own GridSearchCV)
    knn_for_selection = KNeighborsClassifier(n_neighbors=15)

    sfs = SequentialFeatureSelector(
        knn_for_selection,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction="forward",
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X_train, y_train)

    selected_mask = sfs.get_support()
    selected_features = [str(f) for f in np.array(FEATURE_ORDER)[selected_mask]]
    dropped_features = [str(f) for f in np.array(FEATURE_ORDER)[~selected_mask]]

    print("===== STEP 2: Wrapper Method (SequentialFeatureSelector + KNN) =====")
    print(f"Selected {N_FEATURES_TO_SELECT} out of {len(FEATURE_ORDER)} features:")
    print(f"  KEEP    -> {selected_features}")
    print(f"  DROPPED -> {dropped_features}")
    print()

    # ---------------------------------------------------------------
    # Save the selected feature list so ANN / SVM / KNN scripts can
    # ALL reuse the SAME subset for a fair comparison
    # ---------------------------------------------------------------
    joblib.dump(selected_features, "selected_features.pkl")
    print("Saved: selected_features.pkl")
    print("\nShare this file with your teammates. Everyone should load it and")
    print("subset their FEATURE_ORDER / X using these columns BEFORE training,")
    print("so ANN / SVM / KNN are all compared on the same reduced feature set.")
    print("\nExample for a teammate's script:")
    print('    selected_features = joblib.load("selected_features.pkl")')
    print('    X = clean[selected_features]   # instead of clean[FEATURE_ORDER]')


if __name__ == "__main__":
    main()
