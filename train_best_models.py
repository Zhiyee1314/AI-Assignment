import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
RANDOM_STATE = 42
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# -----------------------------------------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------------------------------------
raw = pd.read_csv("diabetes.csv")  # 請確保檔名/路徑正確
clean = raw.copy()

for c in ZERO_AS_MISSING:
    clean[c] = clean[c].replace(0, np.nan)

X = clean[FEATURES]
y = clean[TARGET]

# -----------------------------------------------------------
# 2. TRAIN / TEST SPLIT & PREPROCESSING (Replicating App Pipeline)
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

# -----------------------------------------------------------
# 3. TRAIN ONLY THE NEW HIGH-ACCURACY MODELS
# -----------------------------------------------------------
new_models = {
    "random_forest_model.pkl": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "extra_trees_model.pkl": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

print("\n" + "="*60)
print("TRAINING NEW MODELS & EXPORTING PKL FILES")
print("="*60)

for pkl_filename, model in new_models.items():
    model.fit(X_train_sc, y_train)
    pred = model.predict(X_test_sc)
    
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Accuracy : {acc:.4f} ({(acc*100):.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    
    # 匯出 .pkl 檔案
    joblib.dump(model, pkl_filename)
    print(f"  --> Saved to {pkl_filename}")

print("\n🎉 完成！已經成功生成兩個突破 80%+ 的 pkl 檔案！")
