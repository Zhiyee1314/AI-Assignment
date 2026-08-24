"""Train, evaluate, ablate, and save the KNN model only."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocessing import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    ZERO_AS_MISSING_COLS,
    load_dataset,
    new_imputer,
    new_scaler,
)

MODEL_NAME = "KNN"

# Target feature subset
SELECTED_FEATURES = ["Glucose", "BMI", "Age", "Pregnancies"]


def clean_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 with NaN only for features that exist in df and treat 0 as missing."""
    df_clean = df.copy()
    zero_cols = [c for c in df_clean.columns if c in ZERO_AS_MISSING_COLS]
    if zero_cols:
        df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    return df_clean


def run_feature_ablation(base_estimator, X_train, X_test, y_train, y_test):
    """Retrain KNN after removing each feature, one experiment at a time."""
    rows = []
    for removed in [None] + SELECTED_FEATURES:
        kept = [f for f in SELECTED_FEATURES if f != removed]
        train_subset = clean_zeros(X_train[kept])
        test_subset = clean_zeros(X_test[kept])

        imputer = new_imputer()
        train_imputed = imputer.fit_transform(train_subset)
        test_imputed = imputer.transform(test_subset)

        scaler = new_scaler()
        train_scaled = scaler.fit_transform(train_imputed)
        test_scaled = scaler.transform(test_imputed)

        candidate = clone(base_estimator)
        candidate.fit(train_scaled, y_train)
        prediction = candidate.predict(test_scaled)
        rows.append({
            "Model": MODEL_NAME,
            "Removed Feature": "None" if removed is None else removed,
            "Accuracy": accuracy_score(y_test, prediction),
        })

    report = pd.DataFrame(rows)
    baseline = report.loc[
        report["Removed Feature"] == "None", "Accuracy"
    ].iloc[0]
    report["Accuracy Change"] = report["Accuracy"] - baseline
    return report


def main():
    data = load_dataset()

    # Extract 4 features and target variable
    X = data[SELECTED_FEATURES]
    y = data["Outcome"]  # Update if your target column name is different

    # Train/test split directly
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Clean zero values safely
    X_train_clean = clean_zeros(X_train_raw)
    X_test_clean = clean_zeros(X_test_raw)

    print("Total rows:", len(data))
    print("KNN training patients:", len(y_train))
    print("KNN testing patients:", len(y_test))
    print("Selected Features:", SELECTED_FEATURES)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search_pipeline = Pipeline([
        ("imputer", new_imputer()),
        ("scaler", new_scaler()),
        ("model", KNeighborsClassifier()),
    ])
    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }
    grid = GridSearchCV(
        estimator=search_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )

    print("\nSearching for best KNN parameters...")
    grid.fit(X_train_clean, y_train)
    best_model_params = {
        key.removeprefix("model__"): value
        for key, value in grid.best_params_.items()
        if key.startswith("model__")
    }

    # Fit preprocessors on the clean 4-feature dataset
    imputer = new_imputer()
    scaler = new_scaler()

    X_train_imp = imputer.fit_transform(X_train_clean)
    X_test_imp = imputer.transform(X_test_clean)

    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)

    knn = KNeighborsClassifier(**best_model_params)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\nBest KNN parameters:", best_model_params)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    print("\n===== KNN (Tuned) — Final Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\n", classification_report(y_test, y_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE: saved as knn_imputer.pkl / knn_scaler.pkl (NOT imputer.pkl /
    # scaler.pkl) so the team's shared 8-feature imputer.pkl / scaler.pkl
    # used by ANN and SVM are never overwritten or broken by this script.
    joblib.dump(knn, MODELS_DIR / "knn_model.pkl")

    pd.DataFrame([{
        "Model": MODEL_NAME,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "CV Accuracy Mean": grid.best_score_,
        "CV Accuracy Std": grid.cv_results_["std_test_score"][grid.best_index_],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Best Parameters": json.dumps(best_model_params, default=str),
        "Training Patients": len(y_train),
        "Test Patients": len(y_test),
    }]).to_csv(DATA_DIR / "knn_metrics.csv", index=False)

    pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Prediction": y_pred,
        "Probability": y_prob,
    }).to_csv(DATA_DIR / "knn_test_predictions.csv", index=False)

    print("Running KNN leave-one-feature-out evaluation...")
    ablation = run_feature_ablation(
        knn, X_train_raw, X_test_raw, y_train, y_test
    )
    ablation.to_csv(DATA_DIR / "knn_ablation.csv", index=False)

    print("\nSaved:")
    print(" - models/knn_model.pkl")
    print(" - Data/knn_metrics.csv")
    print(" - Data/knn_ablation.csv")
    print(" - Data/knn_test_predictions.csv")


if __name__ == "__main__":
    main()
