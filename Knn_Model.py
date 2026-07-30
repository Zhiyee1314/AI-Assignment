import warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pickle  # ADDED: Imported pickle module
 
warnings.filterwarnings("ignore", category=UserWarning)
 
CSV_PATH = "diabetes.csv"
 
# (field_key, display_label, min_value, max_value, default_value, step, help_text)
FIELDS = [
    ("Pregnancies", "Number of Pregnancies", 0, 20, 1, 1, "How many times the patient has been pregnant"),
    ("Glucose", "Glucose Level (mg/dL)", 0, 300, 120, 1, "Plasma glucose concentration"),
    ("BloodPressure", "Diastolic Blood Pressure (mmHg)", 0, 200, 70, 1, "Diastolic blood pressure"),
    ("SkinThickness", "Skin Thickness (mm)", 0, 100, 20, 1, "Triceps skin fold thickness"),
    ("Insulin", "Insulin Level (mu U/ml)", 0, 900, 80, 1, "2-Hour serum insulin"),
    ("BMI", "Body Mass Index (BMI)", 0.0, 70.0, 24.5, 0.1, "Weight in kg / (height in m)^2"),
    ("DiabetesPedigreeFunction", "Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01, "Genetic diabetes likelihood score"),
    ("Age", "Age (years)", 1, 120, 30, 1, "Age in years"),
]
 
 
# ============================================
# Model training (cached so it only runs once)
# ============================================
@st.cache_resource
def train_model(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
 
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
 
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    param_grid = {"n_neighbors": range(1, 31)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)
    best_k = grid.best_params_["n_neighbors"]
 
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)

    # ADDED: Save model to knn_model.pkl and scaler to scaler.pkl
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
 
    y_pred = model.predict(X_test_scaled)
 
    results = {
        "best_k": best_k,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "dataset_shape": df.shape,
    }
 
    return model, scaler, feature_names, results
 
 
# ============================================
# Page setup
# ============================================
st.set_page_config(page_title="Diabetes Prediction - KNN", page_icon="🩺", layout="centered")
 
st.title("🩺 Diabetes Prediction System")
st.caption("Using K-Nearest Neighbors (KNN) trained on the Pima Indians Diabetes Dataset")
 
try:
    model, scaler, feature_names, results = train_model()
except FileNotFoundError:
    st.error(
        f"Could not find '{CSV_PATH}'. Please make sure the dataset file is "
        f"in the same folder as this app (and pushed to your GitHub repo)."
    )
    st.stop()
 
tab1, tab2 = st.tabs(["📊 Model Performance", "🔍 Predict"])
 
# ============================================
# Tab 1: Model Performance
# ============================================
with tab1:
    st.subheader("Dataset & Model Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", results["dataset_shape"][0])
    col2.metric("Features", results["dataset_shape"][1] - 1)
    col3.metric("Best K", results["best_k"])
 
    st.subheader("Overall Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [
            f"{results['accuracy']:.4f}",
            f"{results['precision']:.4f}",
            f"{results['recall']:.4f}",
            f"{results['f1']:.4f}",
        ]
    })
    st.table(metrics_df.set_index("Metric"))
 
    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Diabetes", "Actual: Diabetes"],
        columns=["Predicted: No Diabetes", "Predicted: Diabetes"]
    )
    st.table(cm_df)
 
    st.subheader("Classification Report")
    report = results["report"]
    report_rows = []
    for label, name in [("0", "No Diabetes (0)"), ("1", "Diabetes (1)")]:
        row = report[label]
        report_rows.append({
            "Class": name,
            "Precision": f"{row['precision']:.2f}",
            "Recall": f"{row['recall']:.2f}",
            "F1-Score": f"{row['f1-score']:.2f}",
            "Support": int(row["support"]),
        })
    wavg = report["weighted avg"]
    report_rows.append({
        "Class": "Weighted Avg",
        "Precision": f"{wavg['precision']:.2f}",
        "Recall": f"{wavg['recall']:.2f}",
        "F1-Score": f"{wavg['f1-score']:.2f}",
        "Support": int(wavg["support"]),
    })
    report_df = pd.DataFrame(report_rows)
    st.table(report_df.set_index("Class"))
 
# ============================================
# Tab 2: Predict
# ============================================
with tab2:
    st.subheader("Enter Patient Details")
 
    user_values = {}
    with st.form("prediction_form"):
        for key, label, min_val, max_val, default_val, step, help_text in FIELDS:
            user_values[key] = st.number_input(
                label, min_value=min_val, max_value=max_val,
                value=default_val, step=step, help=help_text
            )
        submitted = st.form_submit_button("Predict", use_container_width=True)
 
    if submitted:
        user_df = pd.DataFrame([[user_values[k] for k, *_ in FIELDS]], columns=feature_names)
        user_scaled = scaler.transform(user_df)
 
        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0]
 
        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ HIGH likelihood of diabetes")
        else:
            st.success(f"### ✅ LOW likelihood of diabetes")
 
        col1, col2 = st.columns(2)
        col1.metric("Probability of Diabetes", f"{probability[1] * 100:.2f}%")
        col2.metric("Probability of No Diabetes", f"{probability[0] * 100:.2f}%")
 
        st.progress(float(probability[1]))
        st.caption("Note: For academic purposes only. This is NOT a medical diagnosis. Please consult a doctor for any concerns.")