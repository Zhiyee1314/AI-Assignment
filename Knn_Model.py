import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

X = df[FEATURE_ORDER].copy()
y = df[TARGET].copy()

# ------------------------------------------------------------------
# 2. Treat 0 as missing for invalid medical columns
# ------------------------------------------------------------------
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in ZERO_AS_MISSING_COLS:
    X[c] = X[c].replace(0, np.nan)

# ------------------------------------------------------------------
# 3. Train/test split (identical split matching team settings)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. Load your teammate's existing Imputer & Scaler
#    (CRITICAL: Use .transform(), NOT .fit_transform())
# ------------------------------------------------------------------
imputer = joblib.load("imputer.pkl")
scaler  = joblib.load("scaler.pkl")

X_train_imputed = imputer.transform(X_train)
X_test_imputed  = imputer.transform(X_test)

X_train_scaled = scaler.transform(X_train_imputed)
X_test_scaled  = scaler.transform(X_test_imputed)

# ------------------------------------------------------------------
# 5. Tune K & Train KNN Model
# ------------------------------------------------------------------
param_grid = {"n_neighbors": range(1, 31)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

best_k = grid.best_params_["n_neighbors"]
print(f"Optimal K found: {best_k}")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

acc = knn_model.score(X_test_scaled, y_test)
print(f"KNN Test accuracy: {acc:.3f}")

# ------------------------------------------------------------------
# 6. Save ONLY your knn_model.pkl
# ------------------------------------------------------------------
joblib.dump(knn_model, "knn_model.pkl")
print("Saved: knn_model.pkl successfully!")
