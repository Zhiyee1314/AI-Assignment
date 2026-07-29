"""
train_ann.py
------------
This is the script that PRODUCES the .pkl files your app.py loads.
Run this once locally after you finish training — it saves 3 files into
the same folder: imputer.pkl, scaler.pkl, ann_model.pkl

Your teammates copy THIS SAME imputer.pkl + scaler.pkl into their own
training script (see the note near the bottom) so all 3 models are
trained on identical preprocessing.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ------------------------------------------------------------------
# 1. Load your dataset
# ------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")  # <-- change to your actual CSV path

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"  # <-- change if your label column has a different name

X = df[FEATURE_ORDER].copy()
y = df[TARGET].copy()

# ------------------------------------------------------------------
# 2. Treat 0 as missing for these columns (0 is not physiologically
#    possible for these, so it really means "not recorded")
# ------------------------------------------------------------------
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in ZERO_AS_MISSING_COLS:
    X[c] = X[c].replace(0, np.nan)

# ------------------------------------------------------------------
# 3. Train/test split BEFORE fitting imputer/scaler (avoid leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. Fit imputer + scaler on the TRAIN split only, then transform both
#    -> These two objects are what gets shared with teammates
# ------------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ------------------------------------------------------------------
# 5. Train the model
# ------------------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=1000,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.3f}")

# ------------------------------------------------------------------
# 6. Save all 3 files with joblib (NOT plain pickle — joblib handles
#    numpy arrays inside sklearn objects more efficiently, and it's
#    what your app.py already expects via joblib.load(...))
# ------------------------------------------------------------------
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "ann_model.pkl")

print("Saved: imputer.pkl, scaler.pkl, ann_model.pkl")

# ------------------------------------------------------------------
# NOTE FOR TEAMMATES (SVM / KNN):
# Don't re-fit your own imputer/scaler. Instead:
#
#   imputer = joblib.load("imputer.pkl")   # <- the file you received
#   scaler  = joblib.load("scaler.pkl")    # <- the file you received
#
#   X_train_imputed = imputer.transform(X_train)   # transform, NOT fit_transform
#   X_train_scaled  = scaler.transform(X_train_imputed)
#
#   svm_model = SVC(probability=True, random_state=42)
#   svm_model.fit(X_train_scaled, y_train)
#   joblib.dump(svm_model, "svm_model.pkl")
#
# This guarantees all 3 models see numbers on the same scale, trained
# on the same missing-value treatment, so accuracy/probability
# comparisons between ANN vs SVM vs KNN are actually fair.
# ------------------------------------------------------------------
