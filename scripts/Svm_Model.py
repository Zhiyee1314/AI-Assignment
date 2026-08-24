"""
Train, evaluate, ablate, and save the SVM model only.
"""

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocessing import (  # noqa: E402
    DATA_DIR,
    FEATURE_ORDER,
    MODELS_DIR,
    RANDOM_STATE,
    ZERO_AS_MISSING_COLS,
    load_dataset,
    new_imputer,
    new_scaler,
    split_raw_dataset,
    transform_features,
)

MODEL_NAME = "SVM"


def run_feature_ablation(base_estimator, X_train, X_test, y_train, y_test):
    """Retrain SVM after removing each feature, one experiment at a time."""
    rows = []
    for removed in [None] + FEATURE_ORDER:
        kept = [f for f in FEATURE_ORDER if f != removed]
        train_subset = X_train[kept].copy()
        test_subset = X_test[kept].copy()
        zero_columns = [c for c in kept if c in ZERO_AS_MISSING_COLS]
        train_subset[zero_columns] = train_subset[zero_columns].replace(0, np.nan)
        test_subset[zero_columns] = test_subset[zero_columns].replace(0, np.nan)

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
    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data)

    print("Dataset rows:", len(data))
    print("Selected features:", FEATURE_ORDER)
    print("SVM training patients:", len(y_train))
    print("SVM testing patients:", len(y_test))

    # ---------------------------------------------------------------
    # Load the SHARED imputer/scaler already fitted by Ann_Model.py.
    # Do NOT call fit() or fit_transform() on them here.
    # ---------------------------------------------------------------
    imputer_path = MODELS_DIR / "imputer.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not imputer_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "models/imputer.pkl and models/scaler.pkl were not found. "
            "Run Ann_Model.py first -- it fits and saves the shared "
            "4-feature preprocessors that this script reuses."
        )
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    X_train = transform_features(X_train_raw, imputer, scaler)
    X_test = transform_features(X_test_raw, imputer, scaler)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = [
        # RBF kernel
        {
            "kernel": ["rbf"],
            "C": [0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30, 100],
            "gamma": ["scale", "auto", 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3],
            "class_weight": [None, "balanced"],
        },
        # Linear kernel
        {
            "kernel": ["linear"],
            "C": [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 10],
            "class_weight": [None, "balanced"],
        },
    ]

    grid = GridSearchCV(
        estimator=SVC(probability=True, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("\nSearching for best SVM parameters...")
    grid.fit(X_train, y_train)
    svm = grid.best_estimator_

    print("\nBest SVM parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n===== SVM (Tuned SVC) — Final Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\n", classification_report(y_test, y_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")

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
        "Best Parameters": json.dumps(grid.best_params_, default=str),
        "Training Patients": len(y_train),
        "Test Patients": len(y_test),
    }]).to_csv(DATA_DIR / "svm_metrics.csv", index=False)

    pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Prediction": y_pred,
        "Probability": y_prob,
    }).to_csv(DATA_DIR / "svm_test_predictions.csv", index=False)

    print("Running SVM leave-one-feature-out evaluation...")
    ablation = run_feature_ablation(
        svm, X_train_raw, X_test_raw, y_train, y_test
    )
    ablation.to_csv(DATA_DIR / "svm_ablation.csv", index=False)

    print("\nSaved:")
    print(" - models/svm_model.pkl")
    print(" - Data/svm_metrics.csv")
    print(" - Data/svm_ablation.csv")
    print(" - Data/svm_test_predictions.csv")
    print("\n(models/imputer.pkl and models/scaler.pkl were reused, not overwritten)")


if __name__ == "__main__":
    main()
