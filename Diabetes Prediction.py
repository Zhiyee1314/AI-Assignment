import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load trained model + preprocessors (must match the ones saved by
# ann_model.py — same imputer, same scaler, same feature order)
# ------------------------------------------------------------------
model = joblib.load("ann_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's health information below.")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    # 1. Build a single-row DataFrame in the exact same column order as training
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=FEATURE_ORDER)

    # 2. Treat 0s in these columns as missing, same as training preprocessing
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if input_df.loc[0, c] == 0:
            input_df.loc[0, c] = None

    # 3. Apply the SAME imputer and scaler fitted during training
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=FEATURE_ORDER)
    input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=FEATURE_ORDER)

    # 4. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes  (probability: {probability:.1%})")
    else:
        st.success(f"✅ Low Risk of Diabetes  (probability: {probability:.1%})")

    st.caption("This is a machine learning prediction for educational purposes only, not a medical diagnosis.")
