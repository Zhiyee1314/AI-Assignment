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
    layout="wide",
    initial_sidebar_state="expanded"
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
# STYLE — design tokens + injected CSS
# ----------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg:            #F3F6FC;
            --card:          #FFFFFF;
            --border:        #E3E8F0;
            --text:          #1E2433;
            --muted:         #667085;
            --primary:       #4338CA;
            --primary-dark:  #362DA8;
            --primary-light: #6D63F0;
            --accent:        #0D9488;
            --risk-high:     #D0272B;
            --risk-high-bg:  #FDECEC;
            --risk-low:      #0F8A5F;
            --risk-low-bg:   #E9F8F1;
        }

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Sora', sans-serif;
            color: var(--text);
        }

        /* ---------- Headings ---------- */
        h1, h2, h3, .hero-title {
            font-family: 'Sora', sans-serif;
            color: var(--text);
            letter-spacing: -0.01em;
        }

        /* ---------- Hero banner ---------- */
        .hero-card {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 28px;
            box-shadow: 0 10px 30px -12px rgba(67, 56, 202, 0.45);
        }
        .hero-eyebrow {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.75);
            margin-bottom: 6px;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: rgba(255,255,255,0.85);
            margin-top: 6px;
        }

        /* ---------- Section / card panels ---------- */
        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px 26px 8px 26px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .panel-title {
            font-family: 'Sora', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel-caption {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 14px;
        }

        /* ---------- Inputs ---------- */
        [data-testid="stNumberInput"] label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text);
        }
        [data-testid="stNumberInput"] input {
            border-radius: 10px;
        }

        /* ---------- Predict button ---------- */
        div.stButton > button, div.stDownloadButton > button {
            width: 100%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: #FFFFFF;
            font-weight: 700;
            font-family: 'Sora', sans-serif;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            box-shadow: 0 6px 16px -6px rgba(67, 56, 202, 0.55);
            transition: transform 0.05s ease-in-out, box-shadow 0.15s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            box-shadow: 0 8px 20px -6px rgba(67, 56, 202, 0.7);
            color: #FFFFFF;
        }
        div.stButton > button:active {
            transform: scale(0.99);
        }

        /* ---------- Result card ---------- */
        .result-card {
            display: flex;
            align-items: center;
            gap: 26px;
            border-radius: 16px;
            padding: 22px 28px;
            margin-top: 6px;
            border: 1px solid var(--border);
        }
        .result-card.high { background: var(--risk-high-bg); }
        .result-card.low  { background: var(--risk-low-bg); }

        .result-label {
            font-family: 'Sora', sans-serif;
            font-weight: 800;
            font-size: 1.25rem;
            margin-bottom: 2px;
        }
        .result-label.high { color: var(--risk-high); }
        .result-label.low  { color: var(--risk-low); }

        .result-sub {
            font-size: 0.88rem;
            color: var(--muted);
        }

        .gauge-wrap {
            position: relative;
            width: 108px;
            height: 108px;
            flex-shrink: 0;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .gauge-inner {
            position: absolute;
            width: 82px;
            height: 82px;
            border-radius: 50%;
            background: #FFFFFF;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .gauge-pct {
            font-family: 'Sora', sans-serif;
            font-weight: 800;
            font-size: 1.15rem;
            line-height: 1;
        }
        .gauge-pct-label {
            font-size: 0.62rem;
            color: var(--muted);
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        /* ---------- KPI metric cards (history page) ---------- */
        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px 20px;
            text-align: left;
        }
        .kpi-label {
            font-size: 0.78rem;
            color: var(--muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-value {
            font-family: 'Sora', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text);
            margin-top: 2px;
        }

        .disclaimer {
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def risk_gauge_html(proba: float, is_high: bool) -> str:
    """Builds a small donut gauge (pure CSS conic-gradient) showing probability %."""
    pct = max(0, min(100, proba * 100))
    color = "var(--risk-high)" if is_high else "var(--risk-low)"
    return f"""
    <div class="gauge-wrap" style="background: conic-gradient({color} {pct}%, #E3E8F0 0);">
        <div class="gauge-inner">
            <div class="gauge-pct" style="color:{color};">{pct:.0f}%</div>
            <div class="gauge-pct-label">risk</div>
        </div>
    </div>
    """


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
# APP START
# ----------------------------------------------------------------------
inject_css()

# ---------------- SIDEBAR: MODEL SELECTION ----------------
st.sidebar.markdown("### ⚙️ Settings")

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

st.sidebar.caption(f"Detected models: {', '.join(available_models.keys())}")

st.sidebar.markdown("---")
page = st.sidebar.radio("Page", ["Predict", "Prediction History"])

# ---------------- HERO HEADER ----------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Clinical decision support · demo</div>
        <div class="hero-title">🩺 Diabetes Risk Predictor</div>
        <div class="hero-subtitle">Enter the patient's health information below to estimate diabetes risk.</div>
    </div>
    """,
    unsafe_allow_html=True
)

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

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📋 Patient information</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-caption">Model in use: <strong>{model_choice}</strong></div>',
        unsafe_allow_html=True
    )

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

    st.write("")
    predict_clicked = st.button("Predict", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
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

        is_high = prediction == 1
        risk_class = "high" if is_high else "low"
        risk_word = "High Risk of Diabetes" if is_high else "Low Risk of Diabetes"
        risk_sub = (
            "The model flagged elevated risk based on the values entered — recommend clinical follow-up."
            if is_high else
            "The model did not detect elevated risk from the values entered."
        )

        st.markdown(
            f"""
            <div class="result-card {risk_class}">
                {risk_gauge_html(proba, is_high)}
                <div>
                    <div class="result-label {risk_class}">{'⚠️' if is_high else '✅'} {risk_word}</div>
                    <div class="result-sub">{risk_sub}</div>
                </div>
            </div>
            <div class="disclaimer">This is a machine learning prediction for educational purposes only, not a medical diagnosis.</div>
            """,
            unsafe_allow_html=True
        )

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
    history = load_history()

    if history.empty:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.info("No predictions yet. Go to the Predict page and run one first.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        total = len(history)
        high_pct = (history["prediction"] == "High Risk").mean() * 100
        top_model = history["model"].mode()[0] if not history["model"].mode().empty else "—"

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Total predictions</div>'
                f'<div class="kpi-value">{total}</div></div>',
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">High risk rate</div>'
                f'<div class="kpi-value">{high_pct:.0f}%</div></div>',
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Most-used model</div>'
                f'<div class="kpi-value">{top_model}</div></div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Prediction history</div>', unsafe_allow_html=True)

        colA, colB = st.columns([3, 1])
        with colA:
            model_filter = st.multiselect(
                "Filter by model", options=sorted(history["model"].unique()),
                default=sorted(history["model"].unique())
            )
        with colB:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear history"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()

        filtered = history[history["model"].isin(model_filter)]

        st.markdown("**Risk outcome count per model**")
        chart_data = (
            filtered.groupby(["model", "prediction"]).size().unstack(fill_value=0)
        )
        st.bar_chart(chart_data, color=["#D0272B", "#0F8A5F"])

        st.markdown("**Probability per prediction (most recent last)**")
        st.bar_chart(filtered.set_index(filtered.index)["probability"], color="#4338CA")

        st.markdown("**Raw history table**")
        st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download history as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)
