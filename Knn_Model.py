import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
 
RAW_PATH = "diabetes.csv"     # original, unprocessed dataset (same folder)
TARGET_COL = "Outcome"
RANDOM_STATE = 42             # must match ann_model.py's split
FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
OUTPUT_DIR = "graphs"
 
sns.set_style("whitegrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ---------------------------------------------------------------
# 1. Load raw data + reuse the SAME imputer/scaler already fitted
#    by ann_model.py (do NOT re-fit new ones here -> that would
#    make KNN's preprocessing inconsistent with ANN's/SVM's)
# ---------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
 
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
clean = raw.copy()
for c in cols_with_invalid_zero:
    clean[c] = clean[c].replace(0, np.nan)
 
X = clean[FEATURE_ORDER]
y = clean[TARGET_COL]
 
X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURE_ORDER)
 
# ---------------------------------------------------------------
# 2. Train/test split -- SAME settings as ann_model.py
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
 
# ---------------------------------------------------------------
# 3. Hyperparameter tuning
#    n_neighbors (K) controls the decision boundary. weights determines
#    whether closer neighbors have higher influence, and metric defines
#    the distance calculation algorithm.
# ---------------------------------------------------------------
param_grid = {
    "n_neighbors": range(1, 31),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid, cv=5, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
knn = grid.best_estimator_
best_params = grid.best_params_
print("Best hyperparameters:", best_params)
 
# ---------------------------------------------------------------
# 4. Evaluate on test set
# ---------------------------------------------------------------
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]
 
print("\n===== KNN (Tuned) — Final Results =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
 
# ---------------------------------------------------------------
# 5. Save model for Streamlit deployment
# ---------------------------------------------------------------
joblib.dump(knn, "knn_model.pkl")
print("\nSaved: knn_model.pkl")
 
 
# =================================================================
# 6. GRAPHS FOR ASSIGNMENT REPORT (Section C)
# =================================================================
print("\nGenerating report graphs...")
 
# ---- Graph 1: Accuracy vs K value ------------------------------
# weights/metric are fixed to your best found values, only K varies,
# so this graph reflects the actual tuned model's behaviour.
k_range = range(1, 31)
accuracies = []
for k in k_range:
    temp_model = KNeighborsClassifier(
        n_neighbors=k,
        weights=best_params["weights"],
        metric=best_params["metric"]
    )
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    accuracies.append(accuracy_score(y_test, temp_pred))
 
best_k = best_params["n_neighbors"]
best_k_acc = accuracies[best_k - 1]
 
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), accuracies, marker='o', color='#2c7be5', linewidth=2)
plt.scatter([best_k], [best_k_acc], color='red', s=100, zorder=5,
            label=f"Chosen K = {best_k} (Accuracy = {best_k_acc:.4f})")
plt.title(
    f"KNN Accuracy vs K Value (weights={best_params['weights']}, metric={best_params['metric']})",
    fontsize=12, fontweight='bold'
)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_accuracy_vs_k.png", dpi=200)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/1_accuracy_vs_k.png")
 
# ---- Graph 2: Confusion Matrix Heatmap --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
    annot_kws={"size": 16}
)
plt.title("KNN Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_confusion_matrix.png", dpi=200)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/2_confusion_matrix.png")
 
# ---- Graph 3: ROC Curve + AUC -----------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)
 
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='#2c7be5', linewidth=2, label=f"KNN (AUC = {roc_auc_value:.4f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess (AUC = 0.5)")
plt.title("ROC Curve - KNN Model", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_roc_curve.png", dpi=200)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/3_roc_curve.png")
 
# ---- Graph 4: Precision-Recall Curve ----------------------------
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
 
plt.figure(figsize=(6, 6))
plt.plot(recall_vals, precision_vals, color='#e74c3c', linewidth=2)
plt.title("Precision-Recall Curve - KNN Model", fontsize=14, fontweight='bold')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_precision_recall_curve.png", dpi=200)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/4_precision_recall_curve.png")
 
# ---- Graph 5: Decision Boundary (2D, Glucose vs BMI) ------------
# Retrained on just 2 features purely for visualization -- will NOT
# match the full 8-feature model's accuracy, shown for intuition only.
X_2d = X_train[["Glucose", "BMI"]].values
model_2d = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=best_params["weights"],
    metric=best_params["metric"]
)
model_2d.fit(X_2d, y_train)
 
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='coolwarm',
    edgecolor='k', s=25
)
plt.title(f"KNN Decision Boundary (K={best_k}, Glucose vs BMI)", fontsize=13, fontweight='bold')
plt.xlabel("Glucose (scaled)")
plt.ylabel("BMI (scaled)")
plt.legend(handles=scatter.legend_elements()[0], labels=["No Diabetes", "Diabetes"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_decision_boundary.png", dpi=200)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/5_decision_boundary.png")
print("  NOTE: Graph 5 uses only 2 of the 8 features (visualization only),")
print("        so it will NOT match your full model's reported accuracy.")
 
print(f"\nAll graphs saved in the '{OUTPUT_DIR}/' folder.")
