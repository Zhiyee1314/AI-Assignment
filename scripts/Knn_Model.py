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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocessing import (  # noqa: E402
    DATA_DIR,
    FEATURE_ORDER,
    MODELS_DIR,
    RANDOM_STATE,
    ZERO_AS_MISSING_COLS,
    fit_preprocessors,
    load_dataset,
    new_imputer,
    new_scaler,
    replace_invalid_zeros,
    split_raw_dataset,
    transform_features,
)

MODEL_NAME = "KNN"

# Define the 4 features you want to use (e.g., FEATURE_ORDER[:4] or explicit column names)
SELECTED_FEATURES = FEATURE_ORDER[:4]


def run_feature_ablation(base_estimator, X_train, X_test, y_train, y_test):
    """Retrain KNN after removing each feature, one experiment at a time."""
    rows = []
    for removed in [None] + SELECTED_FEATURES:
        kept = [f for f in SELECTED_FEATURES if f != removed]
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

    # Filter raw datasets to only the 4 selected features
    X_train_raw = X_train_raw[SELECTED_FEATURES]
    X_test_raw = X_test_raw[SELECTED_FEATURES]

    X_train_clean = replace_invalid_zeros(X_train_raw)

    print("Dataset rows:", len(data))
    print("Evaluation patients (all dataset rows):", len(y_train) + len(y_test))
    print("KNN training patients:", len(y_train))
    print("KNN testing patients:", len(y_test))
    print("Selected Features (4):", SELECTED_FEATURES)

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
        "model__n_neighbors": list(range(1, 32)),
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

    imputer, scaler = fit_preprocessors(X_train_raw)
    X_train = transform_features(X_train_raw, imputer, scaler)
    X_test = transform_features(X_test_raw, imputer, scaler)

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
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(knn, MODELS_DIR / "knn_model.pkl")

    saved_model = joblib.load(MODELS_DIR / "knn_model.pkl")
    saved_prediction = saved_model.predict(X_test[:1])
    saved_probability = saved_model.predict_proba(X_test[:1])[:, 1]
    if len(saved_prediction) != 1 or len(saved_probability) != 1:
        raise RuntimeError("Saved KNN model failed prediction verification.")

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
    print(" - models/imputer.pkl")
    print(" - models/scaler.pkl")
    print(" - models/knn_model.pkl")
    print(" - KNN saved-model prediction verification: OK")
    print(" - Data/knn_metrics.csv")
    print(" - Data/knn_ablation.csv")


if __name__ == "__main__":
    main()
