"""
ANN (Artificial Neural Network) for Diabetes Prediction — FINAL VERSION
Member B's task | Dataset: diabetes__1_.csv (Pima Indians Diabetes, 768 rows)

Pipeline:
  raw csv -> replace invalid 0s with NaN -> median imputation -> StandardScaler
  -> train/test split (80/20, stratified) -> tuned MLPClassifier -> evaluate

NOTE for teammates: use RANDOM_STATE=42 and the SAME diabetes_clean.csv /
imputer.pkl / scaler.pkl for SVM and KNN so all 3 models are compared on
identical data and identical preprocessing.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

RAW_PATH = "diabetes.csv"   # original, unprocessed dataset (same folder as this script)
TARGET_COL = "Outcome"
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load + clean missing values
#    Glucose/BloodPressure/SkinThickness/Insulin/BMI = 0 is medically
#    impossible -> treat as missing, impute with column median.
# ---------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
clean = raw.copy()
for c in cols_with_invalid_zero:
    clean[c] = clean[c].replace(0, np.nan)

X = clean.drop(columns=[TARGET_COL])
y = clean[TARGET_COL]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------------------------------------------------------------
# 2. Feature scaling
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

clean_out = X_scaled.copy()
clean_out[TARGET_COL] = y.values
clean_out.to_csv("diabetes_clean.csv", index=False)
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------------------------------
# 3. Train/test split (SAME split settings must be used by SVM & KNN)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 4. Hyperparameter tuning
# ---------------------------------------------------------------
param_grid = {
    "hidden_layer_sizes": [(8,), (16,), (16, 8), (32, 16), (32, 16, 8)],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "activation": ["relu", "tanh"],
}
grid = GridSearchCV(
    MLPClassifier(max_iter=3000, early_stopping=True, random_state=RANDOM_STATE),
    param_grid, cv=5, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
ann = grid.best_estimator_
print("Best hyperparameters:", grid.best_params_)

# ---------------------------------------------------------------
# 5. Evaluate on test set
# ---------------------------------------------------------------
y_pred = ann.predict(X_test)
y_prob = ann.predict_proba(X_test)[:, 1]

print("\n===== ANN (Tuned MLPClassifier) — Final Results =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 6. Save final model for Streamlit deployment
# ---------------------------------------------------------------
joblib.dump(ann, "ann_model.pkl")
print("\nSaved: diabetes_clean.csv, imputer.pkl, scaler.pkl, ann_model.pkl")
