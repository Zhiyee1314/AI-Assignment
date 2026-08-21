"""Streamlit diabetes-risk app for ANN, KNN and SVM."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from CSS.styles import inject_global_css
from scripts.model_pipeline import (
    ABLATION_REPORT_PATH,
    FEATURES,
    load_clean_dataset,
    positive_probability,
    prepare_raw_features,
    split_raw_dataset,
    transform_features,
)


st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "ANN": ROOT_DIR / "models" / "ann_model.pkl",
    "SVM": ROOT_DIR / "models" / "svm_model.pkl",
    "KNN": ROOT_DIR / "models" / "knn_model.pkl",
}
IMPUTER_FILE = ROOT_DIR / "models" / "imputer.pkl"
SCALER_FILE = ROOT_DIR / "models" / "scaler.pkl"
HISTORY_FILE = ROOT_DIR / "prediction_history.csv"

MODEL_TITLES = {
    "ANN": "ANN Diabetes Risk Predictor",
    "SVM": "SVM Diabetes Risk Predictor",
    "KNN": "KNN Diabetes Risk Predictor",
}
MODEL_ICONS = {"ANN": "🧠", "SVM": "📈", "KNN": "👥"}
MODEL_SUBTITLES = {
    "ANN": "A neural network that learns non-linear patterns between patient features.",
    "SVM": "A support vector machine that separates diabetic and non-diabetic cases.",
    "KNN": "A nearest-neighbours model that compares a patient with similar past cases.",
}
MODEL_ACCENT = {"ANN": "#6C63FF", "SVM": "#FF6B6B", "KNN": "#22B07D"}

INPUT_BOUNDS = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0.0, 70.0),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1, 120),
}


@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def get_available_models():
    return {label: path for label, path in MODEL_FILES.items() if path.exists()}


def get_shared_preprocessors():
    if not IMPUTER_FILE.exists() or not SCALER_FILE.exists():
        return None, None
    return load_pickle(IMPUTER_FILE), load_pickle(SCALER_FILE)


def render_header(model_choice):
    st.markdown(
        f"""
        <div class="app-header">
            <div class="icon">{MODEL_ICONS[model_choice]}</div>
            <div>
                <h2>{MODEL_TITLES[model_choice]}</h2>
                <p>{MODEL_SUBTITLES[model_choice]}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(model_choice, prediction, probability):
    risk_pct = float(probability) * 100
    high_risk = int(prediction) == 1
    color = "#E5484D" if high_risk else "#22B07D"
    label = "⚠️ High Risk of Diabetes" if high_risk else "✅ Low Risk of Diabetes"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Risk Assessment · {model_choice}</div>
            <div class="result-label" style="color:{color}">{label}</div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{risk_pct:.1f}%;background:{color};">
                    {risk_pct:.1f}%
                </div>
            </div>
            <div class="disclaimer">Educational machine-learning result only; not a medical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_patient_rows(raw_features, imputer, scaler):
    prepared = prepare_raw_features(raw_features, FEATURES)
    normalized = transform_features(prepared, imputer, scaler)
    return pd.DataFrame(normalized, columns=FEATURES, index=raw_features.index)


def validate_patient_csv(uploaded):
    missing = [column for column in FEATURES if column not in uploaded.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"

    raw_features = uploaded[FEATURES].copy()
    converted = pd.DataFrame(index=raw_features.index)
    invalid_messages = []
    for column in FEATURES:
        original = raw_features[column]
        numeric = pd.to_numeric(original, errors="coerce")
        nonblank = original.notna() & original.astype(str).str.strip().ne("")
        invalid = nonblank & numeric.isna()
        if invalid.any():
            rows = ", ".join(str(int(i) + 2) for i in invalid[invalid].index[:5])
            invalid_messages.append(f"{column} (CSV row {rows})")
        converted[column] = numeric

    if invalid_messages:
        return None, "Non-numeric values found in: " + "; ".join(invalid_messages)

    range_messages = []
    for column, (minimum, maximum) in INPUT_BOUNDS.items():
        outside = converted[column].notna() & ~converted[column].between(minimum, maximum)
        if outside.any():
            rows = ", ".join(str(int(i) + 2) for i in outside[outside].index[:5])
            range_messages.append(f"{column} (CSV row {rows}; allowed {minimum}–{maximum})")
    if range_messages:
        return None, "Values outside the accepted medical-input ranges: " + "; ".join(range_messages)

    return converted, None


def load_history():
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "model", "probability", "prediction"] + FEATURES)


def save_history_row(row):
    history = pd.concat([load_history(), pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


@st.cache_data
def compute_model_comparison(model_paths_tuple):
    model_paths = dict(model_paths_tuple)
    data = load_clean_dataset()
    _, X_test_raw, _, y_test = split_raw_dataset(data=data)
    imputer = load_pickle(IMPUTER_FILE)
    scaler = load_pickle(SCALER_FILE)
    X_test = transform_features(X_test_raw, imputer, scaler)

    rows = []
    for label, path in model_paths.items():
        prediction = load_pickle(path).predict(X_test)
        rows.append({
            "Model": label,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1 Score": f1_score(y_test, prediction, zero_division=0),
        })
    return pd.DataFrame(rows).set_index("Model")


available_models = get_available_models()
if not available_models:
    st.error("No trained model files were found in the models folder.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Single-patient model",
        options=list(available_models),
        format_func=lambda label: f"{MODEL_ICONS[label]}  {label}",
    )
    st.caption(f"Detected models: {', '.join(available_models)}")
    st.divider()
    page = st.radio(
        "Page",
        ["Predict", "Bulk CSV Prediction", "Compare Models", "Feature Ablation", "Prediction History"],
        format_func=lambda item: {
            "Predict": "🩺  Predict",
            "Bulk CSV Prediction": "📁  Bulk CSV Prediction",
            "Compare Models": "📊  Compare Models",
            "Feature Ablation": "🧪  Feature Ablation",
            "Prediction History": "📋  Prediction History",
        }[item],
    )

inject_global_css(MODEL_ACCENT[model_choice])
imputer, scaler = get_shared_preprocessors()
if imputer is None or scaler is None:
    st.error("The shared models/imputer.pkl and models/scaler.pkl files are required.")
    st.stop()


if page == "Predict":
    render_header(model_choice)
    st.markdown("#### 📝 Patient Information")
    st.caption("Enter normal medical values. The app converts them to 0.00–1.00 before prediction.")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 3, 1)
        glucose = st.number_input("Glucose", 0, 300, 162, 1)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 80, 1)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20, 1)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 70, 1)
        bmi = st.number_input("BMI", 0.0, 70.0, 30.01, 0.01, format="%.2f")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.51, 0.01, format="%.2f")
        age = st.number_input("Age", 1, 120, 35, 1)

    if st.button("🩺 Predict", type="primary", width="stretch"):
        raw_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])
        normalized = normalize_patient_rows(raw_input, imputer, scaler)
        model = load_pickle(available_models[model_choice])
        prediction = int(model.predict(normalized.to_numpy())[0])
        probability = float(positive_probability(model, normalized.to_numpy())[0])
        render_result_card(model_choice, prediction, probability)

        with st.expander("View the 0.00–1.00 values sent to the model"):
            st.dataframe(normalized.style.format("{:.4f}"), width="stretch", hide_index=True)

        save_history_row({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_choice,
            "probability": round(probability, 4),
            "prediction": "High Risk" if prediction else "Low Risk",
            **raw_input.iloc[0].to_dict(),
        })


elif page == "Bulk CSV Prediction":
    st.markdown("## 📁 Bulk Patient CSV Prediction")
    st.caption("Upload normal medical values. Every patient is processed by ANN, KNN and SVM.")
    template = pd.DataFrame(columns=FEATURES)
    st.download_button(
        "⬇️ Download patient CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "patient_upload_template.csv",
        "text/csv",
    )
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
        else:
            raw_features, validation_error = validate_patient_csv(uploaded)
            if validation_error:
                st.error(validation_error)
            elif uploaded.empty:
                st.warning("The uploaded CSV has no patient rows.")
            else:
                missing_models = sorted({"ANN", "KNN", "SVM"} - set(available_models))
                if missing_models:
                    st.error("Bulk prediction requires these missing models: " + ", ".join(missing_models))
                else:
                    normalized = normalize_patient_rows(raw_features, imputer, scaler)
                    output = uploaded.copy()
                    for feature in FEATURES:
                        output[f"Normalized_{feature}"] = normalized[feature].round(4)

                    votes = []
                    for label in ["ANN", "KNN", "SVM"]:
                        model = load_pickle(available_models[label])
                        prediction = model.predict(normalized.to_numpy()).astype(int)
                        probability = positive_probability(model, normalized.to_numpy())
                        output[f"{label}_Prediction"] = prediction
                        output[f"{label}_Risk"] = np.where(prediction == 1, "High Risk", "Low Risk")
                        output[f"{label}_Probability"] = np.round(probability, 4)
                        votes.append(prediction)

                    majority = (np.vstack(votes).sum(axis=0) >= 2).astype(int)
                    output["Majority_Vote_Prediction"] = majority
                    output["Majority_Vote_Risk"] = np.where(majority == 1, "High Risk", "Low Risk")

                    st.success(f"Predicted {len(output)} patient(s) with ANN, KNN and SVM.")
                    st.dataframe(output, width="stretch", hide_index=True)
                    st.download_button(
                        "⬇️ Download all predictions as CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        "bulk_diabetes_predictions.csv",
                        "text/csv",
                    )


elif page == "Compare Models":
    st.markdown("## 📊 Model Comparison")
    st.caption("All models use the same 768 unique patients, split, imputer and 0–1 scaler.")
    comparison = compute_model_comparison(tuple((label, str(path)) for label, path in available_models.items()))
    st.dataframe(comparison.style.format("{:.4f}"), width="stretch")
    st.bar_chart(comparison)
    best_model = comparison["Accuracy"].idxmax()
    st.success(f"{best_model} has the highest held-out accuracy: {comparison.loc[best_model, 'Accuracy']:.4f}.")
    st.download_button(
        "⬇️ Download comparison CSV",
        comparison.round(4).to_csv().encode("utf-8"),
        "model_comparison.csv",
        "text/csv",
    )


elif page == "Feature Ablation":
    st.markdown("## 🧪 Leave-One-Feature-Out Accuracy")
    st.caption(
        "Each model is retrained after removing one feature. A negative Accuracy Change means "
        "the model performed worse without that feature, so the feature was useful on this test split."
    )
    if not ABLATION_REPORT_PATH.exists():
        st.error("The ablation report is missing. Run: python scripts/train_all_models.py")
    else:
        ablation = pd.read_csv(ABLATION_REPORT_PATH)
        st.dataframe(
            ablation.style.format({
                "Accuracy": "{:.4f}",
                "Baseline Accuracy": "{:.4f}",
                "Accuracy Change": "{:+.4f}",
            }),
            width="stretch",
            hide_index=True,
        )
        removed_only = ablation[ablation["Removed Feature"] != "None (Baseline)"]
        chart_data = removed_only.pivot(
            index="Removed Feature", columns="Model", values="Accuracy Change"
        )
        st.markdown("#### Accuracy change after removing each feature")
        st.bar_chart(chart_data)
        st.download_button(
            "⬇️ Download feature ablation CSV",
            ablation.to_csv(index=False).encode("utf-8"),
            "feature_ablation_results.csv",
            "text/csv",
        )


else:
    st.markdown("## 📋 Prediction History")
    history = load_history()
    if history.empty:
        st.info("No single-patient predictions have been saved yet.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.multiselect(
                "Filter by model",
                sorted(history["model"].unique()),
                default=sorted(history["model"].unique()),
            )
        with col2:
            st.write("")
            if st.button("🗑️ Clear history", width="stretch"):
                HISTORY_FILE.unlink(missing_ok=True)
                st.rerun()
        filtered = history[history["model"].isin(selected)]
        st.dataframe(filtered.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download history CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
        )
