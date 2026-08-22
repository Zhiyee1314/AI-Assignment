"""Streamlit frontend for the trained ANN, KNN, and SVM models.

This file never retrains a model. It accepts raw medical values, validates
them, applies the saved imputer and 0-1 scaler, and loads existing .pkl files.
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError

from CSS.styles import inject_global_css
from scripts.preprocessing import (
    DATA_DIR,
    FEATURE_ORDER as FEATURES,
    MODELS_DIR,
    transform_features,
    validate_patient_frame,
)


ROOT_DIR = Path(__file__).resolve().parent
HISTORY_FILE = ROOT_DIR / "prediction_history.csv"

MODEL_FILES = {
    "ANN": MODELS_DIR / "ann_model.pkl",
    "KNN": MODELS_DIR / "knn_model.pkl",
    "SVM": MODELS_DIR / "svm_model.pkl",
}
IMPUTER_FILE = MODELS_DIR / "imputer.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

MODEL_TITLES = {
    "ANN": "ANN Diabetes Risk Predictor",
    "KNN": "KNN Diabetes Risk Predictor",
    "SVM": "SVM Diabetes Risk Predictor",
}
MODEL_ICONS = {"ANN": "🧠", "KNN": "👥", "SVM": "📈"}
MODEL_ACCENT = {"ANN": "#6C63FF", "KNN": "#22B07D", "SVM": "#FF6B6B"}
MODEL_SUBTITLES = {
    "ANN": "A tuned multilayer perceptron for non-linear diabetes patterns.",
    "KNN": "A tuned nearest-neighbor model using similar training patients.",
    "SVM": "A tuned and probability-calibrated support vector machine.",
}


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_pickle(path_string, modified_time_ns, file_size):
    """Cache an artifact while invalidating the cache when the file changes."""
    return joblib.load(path_string)


def load_artifact(path):
    path = Path(path)
    stat = path.stat()
    return load_pickle(str(path), stat.st_mtime_ns, stat.st_size)


def get_available_models():
    return {
        label: path
        for label, path in MODEL_FILES.items()
        if path.exists()
    }


def get_positive_probability(model, X):
    """Return class-1 probabilities only when the fitted model supports them."""
    try:
        values = model.predict_proba(X)
        return np.asarray(values[:, 1], dtype=float)
    except (AttributeError, NotFittedError):
        return None


def predict_all_models(raw_patients, available_models, imputer, scaler):
    """Validate and predict one or many raw patient rows with every model."""
    validated = validate_patient_frame(raw_patients)
    scaled = transform_features(validated, imputer, scaler)
    predictions = {}

    for label, path in available_models.items():
        model = load_artifact(path)
        predicted = np.asarray(model.predict(scaled), dtype=int)
        probabilities = get_positive_probability(model, scaled)
        predictions[label] = {
            "prediction": predicted,
            "probability": probabilities,
        }

    return validated, predictions


def render_header(model_choice):
    accent = MODEL_ACCENT[model_choice]
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
    return accent


def render_result_card(model_choice, prediction, probability):
    is_high = int(prediction) == 1
    color = "#E5484D" if is_high else "#22B07D"
    label = "⚠️ High Risk of Diabetes" if is_high else "✅ Low Risk of Diabetes"

    if probability is None or np.isnan(probability):
        st.markdown(
            f"### {label}\n\n{model_choice} does not provide calibrated probability."
        )
        return

    risk_pct = float(probability) * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Selected Model: {model_choice}</div>
            <div class="result-label" style="color:{color}">{label}</div>
            <div style="margin:12px 0;font-size:1.05rem;">
                Diabetes Probability: <strong>{risk_pct:.2f}%</strong>
            </div>
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


def prediction_rows(predictions, row_index=0):
    rows = []
    for label, output in predictions.items():
        predicted = int(output["prediction"][row_index])
        probabilities = output["probability"]
        probability = None if probabilities is None else float(probabilities[row_index])
        rows.append({
            "Model": label,
            "Prediction": "High Risk" if predicted == 1 else "Low Risk",
            "Diabetes Probability": probability,
        })
    return rows


def load_history():
    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except (OSError, pd.errors.EmptyDataError):
            pass
    return pd.DataFrame(
        columns=["timestamp", "model", "probability", "prediction"] + FEATURES
    )


def save_history_row(row):
    existing = load_history()
    new_row = pd.DataFrame([row])
    history = new_row if existing.empty else pd.concat(
        [existing, new_row], ignore_index=True
    )
    try:
        history.to_csv(HISTORY_FILE, index=False)
    except OSError:
        st.warning(
            "Prediction succeeded, but history could not be saved on this host."
        )


def load_model_comparison():
    reports = []
    for label in ["ANN", "KNN", "SVM"]:
        path = DATA_DIR / f"{label.lower()}_metrics.csv"
        if path.exists():
            reports.append(pd.read_csv(path))
    if not reports:
        raise FileNotFoundError(
            "No model metric reports exist. Run each model script first."
        )
    return pd.concat(reports, ignore_index=True).set_index("Model")


def load_ablation_comparison():
    combined = None
    missing = []
    for label in ["ANN", "KNN", "SVM"]:
        path = DATA_DIR / f"{label.lower()}_ablation.csv"
        if not path.exists():
            missing.append(label)
            continue
        report = pd.read_csv(path)[
            ["Removed Feature", "Accuracy", "Accuracy Change"]
        ].rename(columns={
            "Accuracy": f"{label} Accuracy",
            "Accuracy Change": f"{label} Change",
        })
        combined = report if combined is None else combined.merge(
            report, on="Removed Feature", how="outer"
        )
    return combined, missing


available_models = get_available_models()
if not available_models:
    st.error("No model .pkl files were found in the models folder.")
    st.stop()
if not IMPUTER_FILE.exists() or not SCALER_FILE.exists():
    st.error(
        "Prediction is disabled because models/imputer.pkl or "
        "models/scaler.pkl is missing."
    )
    st.stop()

imputer = load_artifact(IMPUTER_FILE)
scaler = load_artifact(SCALER_FILE)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Primary prediction model",
        options=list(available_models),
        format_func=lambda name: f"{MODEL_ICONS[name]}  {name}",
    )
    st.caption("Available models: " + ", ".join(available_models))
    st.divider()
    page = st.radio(
        "Page",
        [
            "Predict",
            "Batch CSV",
            "Compare Models",
            "Feature Ablation",
            "Prediction History",
        ],
        format_func=lambda value: {
            "Predict": "🩺  Predict",
            "Batch CSV": "📄  Batch CSV",
            "Compare Models": "📊  Compare Models",
            "Feature Ablation": "🔬  Feature Ablation",
            "Prediction History": "📋  Prediction History",
        }[value],
    )

inject_global_css(MODEL_ACCENT[model_choice])


if page == "Predict":
    render_header(model_choice)
    st.markdown("#### 📝 Patient Information")
    st.caption(
        "Enter normal medical values. Missing medical measurements may be "
        "entered as 0; the backend performs median imputation and 0-1 scaling."
    )

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 3, 1)
        glucose = st.number_input("Glucose", 0, 300, 162, 1)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 80, 1)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20, 1)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 70, 1)
        bmi = st.number_input("BMI", 0.0, 70.0, 30.0, 0.1, format="%.1f")
        dpf = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.51, 0.01, format="%.2f"
        )
        age = st.number_input("Age", 1, 120, 35, 1)

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
            # Validate and preprocess the raw medical values once.
            validated = validate_patient_frame(patient)
            scaled = transform_features(validated, imputer, scaler)

            # Load and run only the model chosen in the sidebar.
            selected_model = load_artifact(available_models[model_choice])
            selected_prediction = int(selected_model.predict(scaled)[0])
            selected_probabilities = get_positive_probability(
                selected_model, scaled
            )
            selected_probability = (
                None
                if selected_probabilities is None
                else float(selected_probabilities[0])
            )
            render_result_card(
                model_choice, selected_prediction, selected_probability
            )

            with st.expander("View the 0.00–1.00 values sent to the model"):
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

            save_history_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_choice,
                "probability": (
                    np.nan
                    if selected_probability is None
                    else round(selected_probability * 100, 2)
                ),
                "prediction": (
                    "High Risk" if selected_prediction == 1 else "Low Risk"
                ),
                **validated.iloc[0].to_dict(),
            })
        except ValueError as error:
            st.error(str(error))


elif page == "Batch CSV":
    st.markdown("## 📄 Batch Patient Prediction")
    st.write(
        "Upload a CSV containing the eight raw medical-value columns. Every "
        "patient will be evaluated by every available model."
    )
    template = pd.DataFrame(columns=FEATURES)
    st.download_button(
        "⬇️ Download empty CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_patient_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
            extra = [column for column in uploaded.columns if column not in FEATURES]
            if extra:
                st.info("Extra columns ignored: " + ", ".join(extra))

            patients, predictions = predict_all_models(
                uploaded, available_models, imputer, scaler
            )
            duplicate_count = int(patients.duplicated().sum())
            if duplicate_count:
                st.warning(
                    f"{duplicate_count} duplicate patient row(s) detected. They "
                    "are preserved to keep the uploaded row order."
                )

            results = patients.copy()
            results.insert(0, "Patient Row", np.arange(1, len(results) + 1))
            for label, output in predictions.items():
                results[f"{label} Prediction"] = np.where(
                    output["prediction"] == 1, "High Risk", "Low Risk"
                )
                probabilities = output["probability"]
                results[f"{label} Probability"] = (
                    np.nan if probabilities is None else probabilities
                )

            st.success(f"Processed {len(results)} patient(s).")
            st.dataframe(results, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download batch prediction results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_batch_predictions.csv",
                mime="text/csv",
            )
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.error(f"Unable to process CSV: {error}")


elif page == "Compare Models":
    st.markdown("## 📊 Fair Model Comparison")
    st.caption(
        "These saved metrics come from the same 614 original training patients "
        "and 154 held-out original test patients. Streamlit does not retrain."
    )
    try:
        comparison = load_model_comparison()
        metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        st.dataframe(
            comparison[metric_columns].style.format("{:.4f}"),
            width="stretch",
        )
        st.bar_chart(comparison[metric_columns])

        selected_report = st.selectbox(
            "Show confusion matrix for", comparison.index.tolist()
        )
        row = comparison.loc[selected_report]
        matrix = pd.DataFrame(
            [[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(matrix, width="stretch")
        st.download_button(
            "⬇️ Download model comparison",
            comparison.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
    except FileNotFoundError as error:
        st.warning(str(error))


elif page == "Feature Ablation":
    st.markdown("## 🔬 Leave-One-Feature-Out Testing")
    st.caption(
        "Each row is produced by retraining that model with the named feature "
        "removed. No feature is deleted only at prediction time."
    )
    ablation, missing = load_ablation_comparison()
    if missing:
        st.warning("Missing ablation reports for: " + ", ".join(missing))
    if ablation is None:
        st.info("Run the three model scripts to generate ablation reports.")
    else:
        numeric_columns = [c for c in ablation.columns if c != "Removed Feature"]
        st.dataframe(
            ablation.style.format({c: "{:.4f}" for c in numeric_columns}),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ Download feature-ablation comparison",
            ablation.to_csv(index=False).encode("utf-8"),
            file_name="feature_ablation_comparison.csv",
            mime="text/csv",
        )


else:
    st.markdown("## 📋 Prediction History")
    history = load_history()
    if history.empty:
        st.info("No saved single-patient predictions yet.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            choices = sorted(history["model"].dropna().unique())
            selected_models = st.multiselect(
                "Filter by model", choices, default=choices
            )
        with col_b:
            st.write("")
            if st.button("🗑️ Clear history", width="stretch"):
                try:
                    HISTORY_FILE.unlink(missing_ok=True)
                except OSError as error:
                    st.error(f"Unable to clear history: {error}")
                st.rerun()

        filtered = history[history["model"].isin(selected_models)]
        st.dataframe(
            filtered.sort_values("timestamp", ascending=False),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ Download history",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
