import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# 1. 修改 CSS 匯入路徑 (指向 CSS/ 資料夾)
from CSS.styles import inject_global_css


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

TARGET_COL = "Outcome"

# 2. 修改數據集與模型的相對路徑 (指向 data/ 與 models/ 資料夾)
RAW_PATH = "Data/diabetes.csv"
RANDOM_STATE = 42

# Map: dropdown label -> model file on disk
MODEL_FILES = {
    "ANN": "models/ann_model.pkl",
    "SVM": "models/svm_model.pkl",
    "KNN": "models/knn_model.pkl",
}

IMPUTER_FILE = "models/imputer.pkl"
SCALER_FILE = "models/scaler.pkl"
HISTORY_FILE = "prediction_history.csv"

ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

MODEL_TITLES = {
    "ANN": "ANN Diabetes Risk Predictor",
    "SVM": "SVM Diabetes Risk Predictor",
    "KNN": "KNN Diabetes Risk Predictor",
}
MODEL_ICONS = {"ANN": "🧠", "SVM": "📈", "KNN": "👥"}
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


def render_header(model_choice: str):
    accent = MODEL_ACCENT.get(model_choice, "#888888")
    icon = MODEL_ICONS.get(model_choice, "🩺")
    title_text = MODEL_TITLES.get(model_choice, "Diabetes Risk Predictor")
    subtitle_text = MODEL_SUBTITLES.get(model_choice, "Enter the patient's health information below.")
    st.markdown(
        f"""
        <div class="app-header">
            <div class="icon">{icon}</div>
            <div>
                <h2>{title_text}</h2>
                <p>{subtitle_text}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    return accent


def render_result_card(model_choice: str, prediction: int, proba: float, accent: str):
    risk_pct = proba * 100
    is_high = prediction == 1
    bar_color = "#E5484D" if is_high else "#22B07D"
    label = "⚠️ High Risk of Diabetes" if is_high else "✅ Low Risk of Diabetes"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Risk Assessment · {model_choice}</div>
            <div class="result-label" style="color:{bar_color}">{label}</div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{risk_pct:.1f}%;background:{bar_color};">
                    {risk_pct:.1f}%
                </div>
            </div>
            <div class="disclaimer">This is a machine learning prediction for educational purposes only, not a medical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------
@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def get_available_models():
    """Only list models whose .pkl file actually exists in models/ folder."""
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
# MODEL COMPARISON HELPERS
# ----------------------------------------------------------------------
@st.cache_data
def compute_model_comparison(_available_models_tuple):
    """_available_models_tuple: tuple of (label, filepath) pairs so it's hashable for caching."""
    available_models = dict(_available_models_tuple)

    raw = pd.read_csv(RAW_PATH)
    imputer = load_pickle(IMPUTER_FILE)
    scaler = load_pickle(SCALER_FILE)

    clean = raw.copy()
    for c in ZERO_AS_MISSING_COLS:
        clean[c] = clean[c].replace(0, np.nan)

    X = clean[FEATURES]
    y = clean[TARGET_COL]

    X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=FEATURES)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    for label, filepath in available_models.items():
        model = load_pickle(filepath)
        y_pred = model.predict(X_test)
        rows.append({
            "Model": label,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        })

    return pd.DataFrame(rows).set_index("Model")


# ----------------------------------------------------------------------
# HISTORY HELPERS
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
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    available_models = get_available_models()

    if not available_models:
        st.error(
            "No model .pkl files found in models/ folder.\n\n"
            "Expected one or more of: " + ", ".join(MODEL_FILES.values())
        )
        st.stop()

    model_choice = st.selectbox(
        "Choose prediction model",
        options=list(available_models.keys()),
        format_func=lambda m: f"{MODEL_ICONS.get(m, '')}  {m}",
        help="Only models whose .pkl file is present in models/ show up here."
    )

    st.caption(f"Detected models: {', '.join(available_models.keys())}")
    st.divider()

    page = st.radio(
        "Page",
        ["Predict", "Compare Models", "Prediction History"],
        format_func=lambda p: {
            "Predict": "🩺  Predict",
            "Compare Models": "📊  Compare Models",
            "Prediction History": "📋  Prediction History",
        }[p]
    )


# ----------------------------------------------------------------------
# GLOBAL CSS + HEADER
# ----------------------------------------------------------------------
accent = MODEL_ACCENT.get(model_choice, "#888888")
inject_global_css(accent)

if page == "Predict":
    render_header(model_choice)

imputer, scaler = get_shared_preprocessors()

if imputer is None or scaler is None:
    st.warning(
        "`imputer.pkl` and/or `scaler.pkl` not found in `models/`. Predictions will run on raw "
        "input values without the shared preprocessing your team agreed on."
    )


# ----------------------------------------------------------------------
# PAGE: PREDICT
# ----------------------------------------------------------------------
if page == "Predict":

    st.markdown("#### 📝 Patient Information")

    if model_choice == "SVM":
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

        run_clicked = st.button("🔎  Assess Risk", type="primary", use_container_width=True)

    else:
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

        run_clicked = st.button("🩺  Predict", type="primary", use_container_width=True)

    if run_clicked:
        model_path = available_models[model_choice]
        model = load_pickle(model_path)

        input_df = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
            columns=FEATURES
        )

        for c in ZERO_AS_MISSING_COLS:
            if input_df.loc[0, c] == 0:
                input_df.loc[0, c] = None

        if imputer is not None:
            input_df = pd.DataFrame(imputer.transform(input_df), columns=FEATURES)
        if scaler is not None:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

        X = input_df

        prediction = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0][1]
        else:
            proba = float(prediction)

        render_result_card(model_choice, int(prediction), float(proba), accent)

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

if page == "Predict":
    render_header(model_choice)

    st.markdown("#### 📝 Patient Information")
    st.caption(
        "Enter normal medical values. Missing medical measurements may be "
        "entered as 0; the backend performs median imputation and 0-1 scaling."
    )

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", 0, 20, 3, 1
        )

        glucose = st.number_input(
            "Glucose", 0, 300, 162, 1
        )

        blood_pressure = st.number_input(
            "Blood Pressure", 0, 200, 80, 1
        )

        skin_thickness = st.number_input(
            "Skin Thickness", 0, 100, 20, 1
        )

    with col2:
        insulin = st.number_input(
            "Insulin", 0, 900, 70, 1
        )

        bmi = st.number_input(
            "BMI",
            0.0,
            70.0,
            30.0,
            0.1,
            format="%.1f",
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            0.0,
            3.0,
            0.51,
            0.01,
            format="%.2f",
        )

        age = st.number_input(
            "Age", 1, 120, 35, 1
        )
        if st.button(
        f"🩺 Predict with {model_choice}",
        type="primary",
        width="stretch",
    ):
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])

        try:
            # Validate all eight raw medical values.
            validated = validate_patient_frame(patient)

            # Apply the saved imputer and scaler.
            scaled = transform_features(
                validated,
                imputer,
                scaler,
            )

            # Load only the model selected in the sidebar.
            selected_model = load_artifact(
                available_models[model_choice]
            )

            # Run only the selected model.
            selected_prediction = int(
                selected_model.predict(scaled)[0]
            )

            selected_probabilities = get_positive_probability(
                selected_model,
                scaled,
            )

            selected_probability = (
                None
                if selected_probabilities is None
                else float(selected_probabilities[0])
            )

            # Display only the selected model's result.
            render_result_card(
                model_choice,
                selected_prediction,
                selected_probability,
            )

            # Optional: show values after 0–1 normalization.
            with st.expander(
                "View the 0.00–1.00 values sent to the model"
            ):
                normalized = pd.DataFrame(
                    scaled,
                    columns=FEATURES,
                    index=validated.index,
                )

                st.dataframe(
                    normalized.style.format("{:.4f}"),
                    width="stretch",
                    hide_index=True,
                )

            # Save the selected prediction to history.
            save_history_row({
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "model": model_choice,
                "probability": (
                    np.nan
                    if selected_probability is None
                    else round(selected_probability * 100, 2)
                ),
                "prediction": (
                    "High Risk"
                    if selected_prediction == 1
                    else "Low Risk"
                ),
                **validated.iloc[0].to_dict(),
            })

        except ValueError as error:
            st.error(str(error))


# ----------------------------------------------------------------------
# PAGE: COMPARE MODELS
# ----------------------------------------------------------------------
elif page == "Compare Models":
    st.markdown("## 📊 Model Comparison")
    st.caption(
        "All available models are evaluated on the SAME held-out test set "
        "(same 80/20 split, same shared imputer.pkl and scaler.pkl) so the "
        "comparison below is fair."
    )

    if imputer is None or scaler is None:
        st.error("imputer.pkl and scaler.pkl are required in models/ to run a fair comparison.")
    else:
        with st.spinner("Evaluating all available models..."):
            comparison_df = compute_model_comparison(tuple(available_models.items()))

        st.markdown("#### Metrics Table")
        st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)

        st.markdown("#### Metrics Bar Chart")
        st.bar_chart(comparison_df)

        st.markdown("#### Accuracy Only")
        st.bar_chart(comparison_df[["Accuracy"]])

        best_model = comparison_df["Accuracy"].idxmax()
        st.success(
            f"**{best_model}** has the highest accuracy "
            f"({comparison_df.loc[best_model, 'Accuracy']:.4f})."
        )

        st.download_button(
            "⬇️ Download comparison table as CSV",
            data=comparison_df.round(4).to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )


# ----------------------------------------------------------------------
# PAGE: PREDICTION HISTORY
# ----------------------------------------------------------------------
else:
    st.markdown("#### 📋 Prediction History")

    history = load_history()

    if history.empty:
        st.info("No predictions yet. Go to the Predict page and run one first.")
    else:
        colA, colB = st.columns([3, 1])
        with colA:
            model_filter = st.multiselect(
                "Filter by model", options=sorted(history["model"].unique()),
                default=sorted(history["model"].unique())
            )
        with colB:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear history", use_container_width=True):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()

        filtered = history[history["model"].isin(model_filter)]

        st.markdown("##### Raw History Table")
        st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download history as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
