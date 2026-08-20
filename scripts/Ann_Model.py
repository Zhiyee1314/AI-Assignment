"""
ANN Model for Diabetes Prediction
Tuned version using Adam + Early Stopping + Feature Selection
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ==============================================================
# 1. Load dataset
# ==============================================================

df = pd.read_csv("diabetes.csv")

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

X = df[FEATURE_ORDER].copy()
y = df[TARGET].copy()


print("Dataset rows:", len(df))
print("\nClass distribution:")
print(y.value_counts())


# ==============================================================
# 2. Replace invalid zero values with NaN
# ==============================================================

ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in ZERO_AS_MISSING_COLS:
    X[column] = X[column].replace(0, np.nan)


# ==============================================================
# 3. Train / Test Split
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================================================
# 4. Missing Value Imputation
# ==============================================================

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# ==============================================================
# 5. Feature Scaling
# ==============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# ==============================================================
# 6. ANN + Feature Selection
# ==============================================================

# SelectKBest allows Grid Search to determine whether
# using 5, 6, 7 or all 8 features gives better performance.

ann_pipeline = Pipeline([
    (
        "select",
        SelectKBest(score_func=f_classif)
    ),

    (
        "ann",
        MLPClassifier(
            solver="adam",

            # Automatically stop when validation accuracy
            # stops improving
            early_stopping=True,

            validation_fraction=0.15,

            n_iter_no_change=50,

            max_iter=3000,

            tol=0.00001,

            random_state=42
        )
    )
])


# ==============================================================
# 7. Parameters to test
# ==============================================================

param_distributions = {

    # Test how many features should be used
    "select__k": [
        5,
        6,
        7,
        8
    ],

    # ANN architectures
    "ann__hidden_layer_sizes": [
        (8,),
        (12,),
        (16,),
        (24,),
        (32,),
        (12, 6),
        (16, 8),
        (24, 12),
        (32, 16)
    ],

    # Activation function
    "ann__activation": [
        "relu",
        "tanh"
    ],

    # Regularization
    "ann__alpha": [
        0.0001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.2
    ],

    # Learning rate
    "ann__learning_rate_init": [
        0.0005,
        0.001,
        0.002,
        0.003,
        0.005
    ],

    # Batch size
    "ann__batch_size": [
        32,
        64,
        128
    ]
}


# ==============================================================
# 8. Cross Validation
# ==============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==============================================================
# 9. Hyperparameter Search
# ==============================================================

search = RandomizedSearchCV(
    estimator=ann_pipeline,

    param_distributions=param_distributions,

    # Try 80 different combinations
    n_iter=80,

    scoring="accuracy",

    cv=cv,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


print("\nTraining and tuning ANN...")

search.fit(
    X_train_scaled,
    y_train
)


# ==============================================================
# 10. Best ANN
# ==============================================================

model = search.best_estimator_

print("\n========================================")
print("BEST ANN SETTINGS")
print("========================================")

print("\nBest parameters:")
print(search.best_params_)

print(
    f"\nBest CV accuracy: "
    f"{search.best_score_:.4f}"
)


# ==============================================================
# 11. Display selected features
# ==============================================================

selected_mask = (
    model
    .named_steps["select"]
    .get_support()
)

selected_features = np.array(
    FEATURE_ORDER
)[selected_mask]

print("\nSelected features:")

for feature in selected_features:
    print("-", feature)


# ==============================================================
# 12. Evaluate on TEST set
# ==============================================================

y_pred = model.predict(
    X_test_scaled
)

y_prob = model.predict_proba(
    X_test_scaled
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n========================================")
print("ANN (Tuned Adam) — FINAL RESULTS")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")


print(
    "\nConfusion Matrix:\n",
    confusion_matrix(
        y_test,
        y_pred
    )
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==============================================================
# 13. Information about final ANN
# ==============================================================

final_ann = model.named_steps["ann"]

print(
    "ANN training iterations:",
    final_ann.n_iter_
)

print(
    f"ANN final loss: "
    f"{final_ann.loss_:.6f}"
)


# ==============================================================
# 14. Save models
# ==============================================================

joblib.dump(
    imputer,
    "imputer.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    model,
    "ann_model.pkl"
)


print(
    "\nSaved: "
    "imputer.pkl, "
    "scaler.pkl, "
    "ann_model.pkl"
)
