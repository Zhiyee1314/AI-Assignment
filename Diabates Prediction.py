import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------
# Shared preprocessors (same for every model)
# ------------------------------------------------------------------
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ------------------------------------------------------------------
# Auto-detect which model files exist in the repo.
# Teammates just need to drop their .pkl file with this exact name
# in the repo root -- no code changes needed here.
# ------------------------------------------------------------------
MODEL_FILES = {
    "ANN (Neural Network)": "ann_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
}
available_models = {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}

# ------------------------------------------------------------------
# Session state: prediction history (resets when the app restarts,
# but persists while users click around during a live demo)
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's health information below.")

if not available_models:
    st.error("No model files found. Make sure at least ann_model.pkl / svm_model.pkl / knn_model.pkl is in the repo.")
    st.stop()

model_name = st.selectbox("Choose a model", list(available_models.keys()))
model = joblib.load(available_models[model_name])
if len(available_models) < len(MODEL_FILES):
    missing = [n for n in MODEL_FILES if n not in available_models]
    st.caption(f"Not yet added: {', '.join(missing)}")

# ------------------------------------------------------------------
# Static evaluation results from training (test set metrics).
# Update these numbers if you retrain / retune a model.
# ------------------------------------------------------------------
MODEL_METRICS = {
    "ANN (Neural Network)": {"Accuracy": 0.72, "Precision": 0.60, "Recall": 0.61, "F1-score": 0.61, "AUC": 0.80},
    "SVM":                  {"Accuracy": 0.71, "Precision": 0.60, "Recall": 0.48, "F1-score": 0.54, "AUC": 0.81},
}
MODEL_COLOR = {
    "ANN (Neural Network)": "#6C63FF",
    "SVM": "#FF6B6B",
}
MODEL_DESC = {
    "ANN (Neural Network)": "Learns complex non-linear relationships between features via layers of neurons. Best Recall here — misses fewer diabetic patients.",
    "SVM": "Finds the optimal boundary that separates diabetic vs non-diabetic cases. Highest AUC — best at ranking overall risk.",
}

color = MODEL_COLOR.get(model_name, "#888888")
st.markdown(
    f"<div style='padding:10px 16px;border-left:6px solid {color};background:{color}15;border-radius:6px;margin-bottom:8px'>"
    f"<b>{model_name}</b> — {MODEL_DESC.get(model_name, '')}</div>",
    unsafe_allow_html=True
)

if model_name in MODEL_METRICS:
    m = MODEL_METRICS[model_name]
    mcols = st.columns(5)
    for mcol, (label, value) in zip(mcols, m.items()):
        mcol.metric(label, f"{value:.2f}")

st.divider()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=FEATURE_ORDER)

    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if input_df.loc[0, c] == 0:
            input_df.loc[0, c] = None

    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=FEATURE_ORDER)
    input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=FEATURE_ORDER)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    result_label = "High Risk" if prediction == 1 else "Low Risk"

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes  (probability: {probability:.1%})  — model: {model_name}")
    else:
        st.success(f"✅ Low Risk of Diabetes  (probability: {probability:.1%})  — model: {model_name}")
    st.caption("This is a machine learning prediction for educational purposes only, not a medical diagnosis.")

    # Save this prediction into history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Model": model_name,
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Result": result_label,
        "Probability": round(probability, 3),
    })

# ------------------------------------------------------------------
# Prediction History
# ------------------------------------------------------------------
st.divider()
st.subheader("📋 Prediction History")

if not st.session_state.history:
    st.info("No predictions yet. Make a prediction above to see it here.")
else:
    hist_df = pd.DataFrame(st.session_state.history)

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    with right:
        counts = hist_df["Result"].value_counts()
        st.bar_chart(counts)

    st.caption(f"Total predictions this session: {len(hist_df)}")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
