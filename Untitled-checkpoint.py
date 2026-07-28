import warnings
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore", category=UserWarning)
 
DOUBLE_LINE = "=" * 60
 
FEATURE_LABELS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Glucose level (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mmHg)",
    "SkinThickness": "Skin thickness (mm)",
    "Insulin": "Insulin level (mu U/ml)",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function (e.g. 0.5)",
    "Age": "Age (years)",
}
 
 
def print_header(title):
    print("\n" + DOUBLE_LINE)
    print(title)
    print(DOUBLE_LINE)

# ============================================
# Part 1: Train the model
# ============================================
def train_model(csv_path="diabetes.csv"):
    """Load data, preprocess it, train a KNN model, and print evaluation metrics."""
 
    print_header("STEP 1: LOADING & PREPROCESSING DATA")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully -> {df.shape[0]} rows, {df.shape[1]} columns")
 
    # These features cannot biologically be 0, so treat 0 as a missing value
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Missing values handled (zeros replaced with median values)")
 
    # Separate features (X) and target variable (y)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()
 
    # Train-test split (stratified to keep class ratio balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # Feature scaling (critical for KNN since it relies on distance)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling applied (StandardScaler)")
 
    print_header("STEP 2: TRAINING THE KNN MODEL")
    print("Searching for the best K value (testing K = 1 to 30)...")
    param_grid = {"n_neighbors": range(1, 31)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)
    best_k = grid.best_params_["n_neighbors"]
    print(f"Best K value found: {best_k}")
 
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    print("Final model trained successfully")
 
    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
 
    print_header("STEP 3: MODEL PERFORMANCE ON TEST SET")
 
    # --- Metrics table ---
    metrics_table = [
        ["Accuracy", f"{accuracy_score(y_test, y_pred):.4f}"],
        ["Precision", f"{precision_score(y_test, y_pred):.4f}"],
        ["Recall", f"{recall_score(y_test, y_pred):.4f}"],
        ["F1 Score", f"{f1_score(y_test, y_pred):.4f}"],
    ]
    print("\nOverall Metrics:")
    print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="fancy_grid"))
 
    # --- Confusion matrix table ---
    cm = confusion_matrix(y_test, y_pred)
    cm_table = [
        ["Actual: No Diabetes", cm[0][0], cm[0][1]],
        ["Actual: Diabetes", cm[1][0], cm[1][1]],
    ]
    print("\nConfusion Matrix:")
    print(tabulate(
        cm_table,
        headers=["", "Predicted: No Diabetes", "Predicted: Diabetes"],
        tablefmt="fancy_grid"
    ))
 
    # --- Classification report table ---
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_table = []
    for label in ["0", "1"]:
        row = report_dict[label]
        class_name = "No Diabetes (0)" if label == "0" else "Diabetes (1)"
        report_table.append([
            class_name,
            f"{row['precision']:.2f}",
            f"{row['recall']:.2f}",
            f"{row['f1-score']:.2f}",
            int(row['support']),
        ])
    report_table.append([
        "Weighted Avg",
        f"{report_dict['weighted avg']['precision']:.2f}",
        f"{report_dict['weighted avg']['recall']:.2f}",
        f"{report_dict['weighted avg']['f1-score']:.2f}",
        int(report_dict['weighted avg']['support']),
    ])
    print("\nClassification Report:")
    print(tabulate(
        report_table,
        headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
        tablefmt="fancy_grid"
    ))
    print(DOUBLE_LINE)
 
    return model, scaler, feature_names

# ============================================
# Part 2: Get user input
# ============================================
def get_user_input(feature_names):
    """Prompt the user to enter their health details."""
 
    print_header("ENTER PATIENT DETAILS")
 
    user_data = []
    for feature in feature_names:
        label = FEATURE_LABELS.get(feature, feature)
        while True:
            try:
                value = float(input(f"{label:<45}: "))
                user_data.append(value)
                break
            except ValueError:
                print("  -> Invalid input, please enter a number.")
 
    return pd.DataFrame([user_data], columns=feature_names)

# ============================================
# Part 3: Prediction
# ============================================
def predict_diabetes(model, scaler, user_input_df):
    """Scale the user's input and predict whether they are diabetic."""
 
    user_input_scaled = scaler.transform(user_input_df)
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0]
 
    print_header("PREDICTION RESULT")
 
    result_text = "HIGH likelihood of diabetes" if prediction == 1 else "LOW likelihood of diabetes"
 
    result_table = [
        ["Prediction", result_text],
        ["Probability of Diabetes", f"{probability[1] * 100:.2f}%"],
        ["Probability of No Diabetes", f"{probability[0] * 100:.2f}%"],
    ]
    print(tabulate(result_table, tablefmt="fancy_grid"))
 
    print(DOUBLE_LINE)

# ============================================
# Main Program
# ============================================
def main():
    model, scaler, feature_names = train_model("diabetes.csv")
 
    while True:
        user_input_df = get_user_input(feature_names)
        predict_diabetes(model, scaler, user_input_df)
 
        again = input("\nDo you want to Predict again? (y/n): ").strip().lower()
        if again != "y":
            print("\nThank you for using the system!")
            break
 
 
if __name__ == "__main__":
    main()
