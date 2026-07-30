import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Map: dropdown label -> model file on disk
# Add new members' models here (or auto-detected below)
MODEL_FILES = {
    "ANN": "ann_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
}

IMPUTER_FILE = "imputer.pkl"
SCALER_FILE = "scaler.pkl"
HISTORY_FILE = "prediction_history.csv"


# Columns where a 0 actually means "missing" (matches your training script)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------
@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def get_available_models():
    """Only list models whose .pkl file actually exists in the folder.
    This is what lets teammates just drop svm_model.pkl / knn_model.pkl
    into the same folder with zero code changes."""
    available = {}
    for label, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            available[label] = filename
    return available


def get_shared_preprocessors():
    imputer = load_pickle(IMPUTER_FILE) if os.path.exists(IMPUTER_FILE) else None
    scaler = load_pickle(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
    return imputer, scaler


# ----------------------------------------------------------------------
# HISTORY HELPERS (persisted to a local CSV so it survives app restarts)
# ----------------------------------------------------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "model", "probability", "prediction"] + FEATURES)


def save_history_row(row: dict):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)
    return df


# ----------------------------------------------------------------------
# SIDEBAR: MODEL SELECTION
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

available_models = get_available_models()

if not available_models:
    st.sidebar.error(
        "No model .pkl files found in this folder.\n\n"
        "Expected one or more of: " + ", ".join(MODEL_FILES.values())
    )
    st.stop()

model_choice = st.sidebar.selectbox(
    "Choose prediction model",
    options=list(available_models.keys()),
    help="Only models whose .pkl file is present in the app folder show up here."
)

st.sidebar.caption(
    f"Detected models: {', '.join(available_models.keys())}"
)

page = st.sidebar.radio("Page", ["Predict", "Prediction History"])


# ----------------------------------------------------------------------
# HEADER (changes title/description depending on the selected model)
# ----------------------------------------------------------------------
MODEL_TITLES = {
    "ANN": "🧠 ANN Diabetes Risk Predictor",
    "SVM": "📈 SVM Diabetes Risk Predictor",
    "KNN": "👥 KNN Diabetes Risk Predictor",
}
MODEL_SUBTITLES = {
    "ANN": "Powered by a Multilayer Perceptron (Neural Network) that learns non-linear patterns between health features.",
    "SVM": "Powered by a Support Vector Machine that finds the optimal boundary separating diabetic vs non-diabetic cases.",
    "KNN": "Powered by K-Nearest Neighbors, which predicts based on the most similar past patients in the dataset.",
}
MODEL_ACCENT = {
    "ANN": "#6C63FF",
    "SVM": "#FF6B6B",
    "KNN": "#22B07D",
}

title_text = MODEL_TITLES.get(model_choice, "🩺 Diabetes Risk Predictor")
subtitle_text = MODEL_SUBTITLES.get(model_choice, "Enter the patient's health information below.")
accent = MODEL_ACCENT.get(model_choice, "#888888")

st.markdown(
    f"<h2 style='color:{accent};margin-bottom:0'>{title_text}</h2>",
    unsafe_allow_html=True
)
st.caption(subtitle_text)

imputer, scaler = get_shared_preprocessors()

if imputer is None or scaler is None:
    st.warning(
        "`imputer.pkl` and/or `scaler.pkl` not found. Predictions will run on raw "
        "input values without the shared preprocessing your team agreed on. "
        "Make sure both files sit in the same folder as app.py."
    )


# ----------------------------------------------------------------------
# PAGE: PREDICT
# ----------------------------------------------------------------------
if page == "Predict":

    if model_choice == "SVM":
        # ---- Similar structure to ANN (two-column number inputs), but
        # ---- with a different field grouping/order and result style ----
        col1, col2 = st.columns(2)

        with col1:
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=162, step=1)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=70, step=1)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.01, step=0.01, format="%.2f")
            age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

        with col2:
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.51, step=0.01, format="%.2f")

        run_clicked = st.button("🔎 Assess Risk", type="primary")

    else:
        # ---- Original layout for ANN / KNN ----
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=162, step=1)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=70, step=1)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.01, step=0.01, format="%.2f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.51, step=0.01, format="%.2f")
            age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

        run_clicked = st.button("Predict", type="primary")

    if run_clicked:
        model_path = available_models[model_choice]
        model = load_pickle(model_path)

        input_df = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
            columns=FEATURES
        )

        # Treat 0s as missing in the same columns used during training
        for c in ZERO_AS_MISSING_COLS:
            if input_df.loc[0, c] == 0:
                input_df.loc[0, c] = None

        if imputer is not None:
            input_df = pd.DataFrame(imputer.transform(input_df), columns=FEATURES)
        if scaler is not None:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

        X = input_df

        prediction = model.predict(X)[0]
        # Probability of the positive (diabetic) class
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0][1]
        else:
            # Fallback for models without predict_proba (e.g. some SVM configs)
            proba = float(prediction)

        if model_choice == "SVM":
            # ---- Distinct SVM result: gauge-style progress bar instead
            # ---- of the st.error/st.success banner used by ANN/KNN ----
            st.write("")
            st.markdown("#### Risk Assessment Result")
            risk_pct = proba * 100
            bar_color = "#FF6B6B" if prediction == 1 else "#22B07D"
            st.markdown(
                f"""
                <div style="background:#eee;border-radius:8px;height:28px;width:100%;overflow:hidden;">
                  <div style="background:{bar_color};height:100%;width:{risk_pct:.1f}%;
                       display:flex;align-items:center;justify-content:flex-end;padding-right:8px;
                       color:white;font-weight:600;font-size:13px;">
                       {risk_pct:.1f}%
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            label = "HIGH RISK" if prediction == 1 else "LOW RISK"
            st.markdown(f"**Classification:** :{'red' if prediction == 1 else 'green'}[{label}]")
        else:
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes (probability: {proba*100:.1f}%)")
            else:
                st.success(f"✅ Low Risk of Diabetes (probability: {proba*100:.1f}%)")

        st.caption("This is a machine learning prediction for educational purposes only, not a medical diagnosis.")

        save_history_row({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_choice,
            "probability": round(float(proba) * 100, 2),
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        })


# ----------------------------------------------------------------------
# PAGE: PREDICTION HISTORY
# ----------------------------------------------------------------------
else:
    st.markdown("### 📊 Prediction History")

    history = load_history()

    if history.empty:
        st.info("No predictions yet. Go to the Predict page and run one first.")
    else:
        colA, colB = st.columns([1, 1])
        with colA:
            model_filter = st.multiselect(
                "Filter by model", options=sorted(history["model"].unique()),
                default=sorted(history["model"].unique())
            )
        with colB:
            if st.button("🗑️ Clear history"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()

        filtered = history[history["model"].isin(model_filter)]

        st.markdown("#### Risk outcome count per model")
        chart_data = (
            filtered.groupby(["model", "prediction"]).size().unstack(fill_value=0)
        )
        st.bar_chart(chart_data)

        st.markdown("#### Probability per prediction (most recent last)")
        st.bar_chart(filtered.set_index(filtered.index)["probability"])

        st.markdown("#### Raw history table")
        st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download history as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

