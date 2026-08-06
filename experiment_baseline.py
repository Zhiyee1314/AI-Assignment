import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

raw = pd.read_csv("/mnt/user-data/uploads/diabetes__1_.csv")
print("Shape:", raw.shape)
print("Outcome balance:\n", raw["Outcome"].value_counts(normalize=True))

clean = raw.copy()
for c in ZERO_AS_MISSING:
    clean[c] = clean[c].replace(0, np.nan)
print("\nMissing values per column after treating 0 as NaN:")
print(clean[ZERO_AS_MISSING].isna().sum())

X = clean[FEATURES]
y = clean[TARGET]

# ============================================================
# BASELINE: exactly replicate the current app pipeline
# (mean imputer + standard scaler + default hyperparams)
# ============================================================
print("\n" + "="*60)
print("BASELINE (current app pipeline, default hyperparameters)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

baseline_models = {
    "SVM (default)": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN (default)": KNeighborsClassifier(),
    "ANN/MLP (default)": MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
}

for name, model in baseline_models.items():
    model.fit(X_train_sc, y_train)
    pred = model.predict(X_test_sc)
    print(f"{name:20s} acc={accuracy_score(y_test, pred):.4f}")
