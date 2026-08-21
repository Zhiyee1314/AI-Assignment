"""Train, evaluate, and save the ANN diabetes prediction model.

This script trains ANN only. It creates the shared imputer and 0-1 scaler
that KNN and SVM must load so all three models use identical preprocessing.
"""

import sys
from pathlib import Path

import joblib
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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
)
from sklearn.neural_network import MLPClassifier


# Allow this file to be run with: python scripts\Ann_Model.py
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_pipeline import (
    MODELS_DIR,
    RANDOM_STATE,
    create_model_ablation_report,
    evaluate_model,
    fit_preprocessors,
    load_clean_dataset,
    split_raw_dataset,
    transform_features,
    update_ablation_report,
    update_model_metrics,
)


def main():
    # ---------------------------------------------------------------
    # 1. Load data and create the shared, fair train/test split
    # ---------------------------------------------------------------
    data = load_clean_dataset()

    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data=data)

    print("Dataset patients:", len(data))
    print("ANN training patients:", len(y_train))
    print("ANN testing patients:", len(y_test))

    # ---------------------------------------------------------------
    # 2. Fit preprocessing on training data only
    # ---------------------------------------------------------------
    imputer, scaler = fit_preprocessors(X_train_raw)

    X_train_scaled = transform_features(X_train_raw, imputer, scaler)
    X_test_scaled = transform_features(X_test_raw, imputer, scaler)

    print("Minimum normalized value:", X_train_scaled.min())
    print("Maximum normalized value:", X_train_scaled.max())

    if X_train_scaled.min() < 0.0 or X_train_scaled.max() > 1.0:
        raise ValueError("Training inputs are outside the required 0-1 range.")

    if X_test_scaled.min() < 0.0 or X_test_scaled.max() > 1.0:
        raise ValueError("Testing inputs are outside the required 0-1 range.")

    # ---------------------------------------------------------------
    # 3. Tune the ANN using training cross-validation only
    # ---------------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    ann = MLPClassifier(
        solver="lbfgs",
        max_iter=15000,
        max_fun=150000,
        tol=0.001,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "hidden_layer_sizes": [
            (4,),
            (8,),
            (16,),
            (32,),
            (8, 4),
            (16, 8),
            (32, 16),
        ],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    grid = GridSearchCV(
        estimator=ann,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        return_train_score=True,
    )

    print("\nTuning ANN hyperparameters...")
    grid.fit(X_train_scaled, y_train)

    print("Best ANN parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # ---------------------------------------------------------------
    # 4. Tune the decision threshold using training data only
    # ---------------------------------------------------------------
    threshold_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = TunedThresholdClassifierCV(
        estimator=clone(grid.best_estimator_),
        scoring="accuracy",
        response_method="predict_proba",
        thresholds=100,
        cv=threshold_cv,
        refit=True,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    print("\nTuning ANN probability threshold...")
    model.fit(X_train_scaled, y_train)
    print(f"Best probability threshold: {model.best_threshold_:.4f}")

    # ---------------------------------------------------------------
    # 5. Evaluate once on the held-out original patients
    # ---------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    print("\n===== ANN (Tuned MLP) - Final Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred, zero_division=0))

    if accuracy < 0.80:
        print(
            "NOTE: Test accuracy is below 80%. The result is still valid; "
            "0-1 scaling and parameter tuning cannot guarantee 80% on "
            "unseen patients."
        )

    # ---------------------------------------------------------------
    # 6. Save preprocessing and ANN before the longer ablation test
    # ---------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(model, MODELS_DIR / "ann_model.pkl")

    print("\nSaved:")
    print(" - models/imputer.pkl")
    print(" - models/scaler.pkl")
    print(" - models/ann_model.pkl")

    # ---------------------------------------------------------------
    # 7. Update ANN metrics without deleting teammate results
    # ---------------------------------------------------------------
    metrics = evaluate_model("ANN", model, X_test_scaled, y_test)
    update_model_metrics("ANN", metrics)

    # ---------------------------------------------------------------
    # 8. ANN leave-one-feature-out evaluation
    # ---------------------------------------------------------------
    print("\nRunning ANN leave-one-feature-out evaluation...")
    ablation = create_model_ablation_report("ANN", data, model)
    update_ablation_report("ANN", ablation)

    print("Saved ANN results to:")
    print(" - Data/model_metrics.csv")
    print(" - Data/feature_ablation_results.csv")


if __name__ == "__main__":
    main()