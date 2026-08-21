"""Train ANN, KNN and SVM with one fair 0-1 preprocessing pipeline.

Run from the repository root:
    python scripts/train_all_models.py

The script removes exact duplicate rows, performs one shared stratified split,
fits the imputer and MinMaxScaler on training data only, tunes all three models,
saves their artifacts, and creates the leave-one-feature-out CSV report.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_pipeline import (
    ABLATION_REPORT_PATH,
    FEATURES,
    MODEL_METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SVMWithCalibratedProbability,
    fit_preprocessors,
    load_clean_dataset,
    positive_probability,
    split_raw_dataset,
    transform_features,
)


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tune_models(X_train, y_train):
    searches = {
        "ANN": GridSearchCV(
            MLPClassifier(
                solver="adam", max_iter=1500, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=30,
                random_state=RANDOM_STATE,
            ),
            {
                "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.005],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=1,
        ),
        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": range(3, 32, 2),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=1,
        ),
        "SVM": GridSearchCV(
            SVC(random_state=RANDOM_STATE),
            [
                {
                    "kernel": ["rbf"],
                    "C": [0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30],
                    "gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1],
                    "class_weight": [None, "balanced"],
                },
                {
                    "kernel": ["linear"],
                    "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    "class_weight": [None, "balanced"],
                },
            ],
            cv=CV,
            scoring="accuracy",
            n_jobs=1,
        ),
    }

    tuned = {}
    for label, search in searches.items():
        print(f"\nTuning {label}...")
        search.fit(X_train, y_train)
        print(f"Best {label} parameters: {search.best_params_}")
        print(f"Best {label} CV accuracy: {search.best_score_:.4f}")
        tuned[label] = search.best_estimator_
    return tuned


def calibrated_models(tuned, X_train, y_train):
    """Fit final estimators; calibrate SVM without deprecated SVC probability mode."""
    svm_classifier = clone(tuned["SVM"])
    svm_classifier.fit(X_train, y_train)
    svm_probability_model = CalibratedClassifierCV(
        estimator=clone(tuned["SVM"]), method="sigmoid", cv=CV
    )
    svm_probability_model.fit(X_train, y_train)
    models = {
        "ANN": clone(tuned["ANN"]),
        "KNN": clone(tuned["KNN"]),
        "SVM": SVMWithCalibratedProbability(svm_classifier, svm_probability_model),
    }
    models["ANN"].fit(X_train, y_train)
    models["KNN"].fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    rows = []
    for label, model in models.items():
        prediction = model.predict(X_test)
        rows.append({
            "Model": label,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1 Score": f1_score(y_test, prediction, zero_division=0),
        })
    return pd.DataFrame(rows)


def create_ablation_report(data, tuned_models):
    """Retrain each algorithm after removing one feature and report test accuracy."""
    rows = []
    for removed_feature in [None] + FEATURES:
        used_features = [f for f in FEATURES if f != removed_feature]
        X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(
            data=data, features=used_features
        )
        imputer, scaler = fit_preprocessors(X_train_raw)
        X_train = transform_features(X_train_raw, imputer, scaler)
        X_test = transform_features(X_test_raw, imputer, scaler)

        models = {
            "ANN": clone(tuned_models["ANN"]),
            "KNN": clone(tuned_models["KNN"]),
            "SVM": clone(tuned_models["SVM"]),
        }
        for label, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            rows.append({
                "Model": label,
                "Removed Feature": removed_feature or "None (Baseline)",
                "Features Used": len(used_features),
                "Accuracy": accuracy_score(y_test, prediction),
            })

    report = pd.DataFrame(rows)
    baseline = report[report["Removed Feature"] == "None (Baseline)"].set_index("Model")["Accuracy"]
    report["Baseline Accuracy"] = report["Model"].map(baseline)
    report["Accuracy Change"] = report["Accuracy"] - report["Baseline Accuracy"]
    return report[[
        "Model", "Removed Feature", "Features Used", "Accuracy",
        "Baseline Accuracy", "Accuracy Change",
    ]]


def main():
    data = load_clean_dataset()
    print(f"Unique dataset rows available: {len(data)}")

    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data=data)
    imputer, scaler = fit_preprocessors(X_train_raw)
    X_train = transform_features(X_train_raw, imputer, scaler)
    X_test = transform_features(X_test_raw, imputer, scaler)

    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert X_test.min() >= 0.0 and X_test.max() <= 1.0

    tuned = tune_models(X_train, y_train)
    models = calibrated_models(tuned, X_train, y_train)
    metrics = evaluate_models(models, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    for label, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{label.lower()}_model.pkl")

    metrics.round(4).to_csv(MODEL_METRICS_PATH, index=False)
    report = create_ablation_report(data, tuned)
    report.round(4).to_csv(ABLATION_REPORT_PATH, index=False)

    print("\nFinal held-out metrics:")
    print(metrics.round(4).to_string(index=False))
    print(f"\nSaved models to: {MODELS_DIR}")
    print(f"Saved metrics to: {MODEL_METRICS_PATH}")
    print(f"Saved ablation report to: {ABLATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
