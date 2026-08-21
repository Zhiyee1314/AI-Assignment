import hashlib
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from CSS.styles import inject_global_css
from prediction_service import FEATURES, positive_probability, predict_all_models, preprocess_patient_data


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

MODEL_FILES = {
    "ANN": "models/ann_model.pkl",
    "SVM": "models/svm_model.pkl",
    "KNN": "models/knn_model.pkl",
}
IMPUTER_FILE = "models/imputer.pkl"
SCALER_FILE = "models/scaler.pkl"
HISTORY_FILE = "prediction_history.csv"
COMPARISON_FILE = "Data/model_comparison_results.csv"
ABLATION_FILE = "Data/feature_ablation_results.csv"

MODEL_TITLES = {
    "ANN": "ANN Diabetes Risk Predictor",
    "SVM": "SVM Diabetes Risk Predictor",
    "KNN": "KNN Diabetes Risk Predictor",
}
MODEL_ICONS = {"ANN": "🧠", "SVM": "📈", "KNN": "👥"}
MODEL_SUBTITLES = {
    "ANN": "A neural network that learns non-linear relationships between patient health features.",
    "SVM": "A support vector machine that separates higher-risk and lower-risk patient patterns.",
    "KNN": "A nearest-neighbor model that compares a patient with similar records in the dataset.",
}
MODEL_ACCENT = {"ANN": "#6C63FF", "SVM": "#FF6B6B", "KNN": "#22B07D"}


@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def get_available_models():
    return {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}


def load_all_models():
    return {name: load_pickle(path) for name, path in get_available_models().items()}


def load_history():
    columns = ["timestamp", "source", "patient_id", "model", "probability", "prediction"] + FEATURES
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        for column in columns:
            if column not in history.columns:
                history[column] = "" if column in {"source", "patient_id"} else pd.NA
        return history
    return pd.DataFrame(columns=columns)


def save_history_rows(rows):
    history = load_history()
    history = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


def render_header(model_choice):
    icon = MODEL_ICONS[model_choice]
    st.markdown(
        f"""
        <div class="app-header">
            <div class="icon">{icon}</div>
            <div>
                <h2>{MODEL_TITLES[model_choice]}</h2>
                <p>{MODEL_SUBTITLES[model_choice]}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(model_choice, prediction, probability):
    risk_pct = probability * 100
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
            <div class="disclaimer">Educational machine-learning prediction only; this is not a medical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def patient_form():
    left, right = st.columns(2)
    with left:
        pregnancies = st.number_input("Pregnancies", 0, 20, 3, 1)
        glucose = st.number_input("Glucose", 0, 300, 162, 1)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 80, 1)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20, 1)
    with right:
        insulin = st.number_input("Insulin", 0, 900, 70, 1)
        bmi = st.number_input("BMI", 0.0, 70.0, 30.01, 0.01, format="%.2f")
        dpf = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.51, 0.01, format="%.2f"
        )
        age = st.number_input("Age", 1, 120, 35, 1)

    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    return values, st.button("🩺 Predict", type="primary", width="stretch")


def make_batch_template():
    return pd.DataFrame(
        [
            {
                "PatientID": "P001",
                "Pregnancies": 3,
                "Glucose": 162,
                "BloodPressure": 80,
                "SkinThickness": 20,
                "Insulin": 70,
                "BMI": 30.01,
                "DiabetesPedigreeFunction": 0.51,
                "Age": 35,
            },
            {
                "PatientID": "P002",
                "Pregnancies": 1,
                "Glucose": 95,
                "BloodPressure": 66,
                "SkinThickness": 18,
                "Insulin": 85,
                "BMI": 25.4,
                "DiabetesPedigreeFunction": 0.25,
                "Age": 27,
            },
        ]
    )


def batch_history_rows(batch_data, batch_results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for index, patient in batch_data.reset_index(drop=True).iterrows():
        patient_id = patient.get("PatientID", index + 1)
        for model_name in ["ANN", "SVM", "KNN"]:
            row = {
                "timestamp": timestamp,
                "source": "Batch CSV",
                "patient_id": patient_id,
                "model": model_name,
                "probability": batch_results.loc[index, f"{model_name}_Probability_Percent"],
                "prediction": batch_results.loc[index, f"{model_name}_Prediction"],
            }
            row.update({feature: patient[feature] for feature in FEATURES})
            rows.append(row)
    return rows


available_models = get_available_models()
if not available_models:
    st.error("No model files were found in the models folder.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Single-patient model",
        list(available_models),
        format_func=lambda name: f"{MODEL_ICONS[name]} {name}",
    )
    st.caption(f"Detected models: {', '.join(available_models)}")
    st.divider()
    page = st.radio(
        "Page",
        ["Predict", "Batch CSV", "Compare Models", "Feature Ablation", "Prediction History"],
        format_func=lambda name: {
            "Predict": "🩺 Predict",
            "Batch CSV": "📤 Batch CSV",
            "Compare Models": "📊 Compare Models",
            "Feature Ablation": "🧪 Feature Ablation",
            "Prediction History": "📋 Prediction History",
        }[name],
    )

accent = MODEL_ACCENT[model_choice]
inject_global_css(accent)

if not os.path.exists(IMPUTER_FILE) or not os.path.exists(SCALER_FILE):
    st.error("The shared imputer.pkl and scaler.pkl files are required.")
    st.stop()

imputer = load_pickle(IMPUTER_FILE)
scaler = load_pickle(SCALER_FILE)


if page == "Predict":
    render_header(model_choice)
    st.caption("Enter normal medical values. The backend automatically converts all 8 features to 0.00–1.00.")
    values, run_clicked = patient_form()

    if run_clicked:
        raw_input = pd.DataFrame([values], columns=FEATURES)
        try:
            model_input = preprocess_patient_data(raw_input, imputer, scaler)
            model = load_pickle(available_models[model_choice])
            prediction = int(model.predict(model_input.to_numpy())[0])
            probability = float(positive_probability(model, model_input)[0])
            render_result_card(model_choice, prediction, probability)

            with st.expander("View backend values (0.00–1.00)"):
                st.dataframe(model_input.style.format("{:.4f}"), hide_index=True, width="stretch")

            save_history_rows(
                [
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Single Patient",
                        "patient_id": "",
                        "model": model_choice,
                        "probability": round(probability * 100, 2),
                        "prediction": "High Risk" if prediction == 1 else "Low Risk",
                        **values,
                    }
                ]
            )
        except ValueError as error:
            st.error(str(error))


elif page == "Batch CSV":
    st.markdown("## 📤 Predict Many Patients from CSV")
    st.write("Upload one CSV containing the 8 required medical values. Every patient will be predicted by ANN, SVM and KNN.")
    template = make_batch_template()
    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "patient_batch_template.csv",
        "text/csv",
    )

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            if st.session_state.get("batch_file_signature") != file_signature:
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_normalized", None)
                st.session_state["batch_file_signature"] = file_signature
            batch_data = pd.read_csv(uploaded_file)
            st.markdown("#### Uploaded data preview")
            st.dataframe(batch_data.head(20), hide_index=True, width="stretch")
            st.caption(f"{len(batch_data)} patient row(s) detected.")

            if st.button("🔎 Run ANN, SVM and KNN", type="primary", width="stretch"):
                models = load_all_models()
                if set(models) != {"ANN", "SVM", "KNN"}:
                    raise ValueError("Batch prediction requires ann_model.pkl, svm_model.pkl and knn_model.pkl.")
                results, normalized = predict_all_models(batch_data, models, imputer, scaler)
                st.session_state["batch_results"] = results
                st.session_state["batch_normalized"] = normalized
                save_history_rows(batch_history_rows(batch_data, results))

            if "batch_results" in st.session_state:
                results = st.session_state["batch_results"]
                st.success(f"Completed {len(results) * 3} predictions for {len(results)} patients.")
                st.dataframe(results, hide_index=True, width="stretch")
                st.download_button(
                    "⬇️ Download batch prediction results",
                    results.to_csv(index=False).encode("utf-8"),
                    "batch_prediction_results.csv",
                    "text/csv",
                )
                with st.expander("View normalized backend inputs (0.00–1.00)"):
                    st.dataframe(
                        st.session_state["batch_normalized"].style.format("{:.4f}"),
                        hide_index=True,
                        width="stretch",
                    )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))


elif page == "Compare Models":
    st.markdown("## 📊 Model Comparison")
    st.caption("All models use the same unique patients, 80/20 split, median imputer and 0–1 scaler.")
    if not os.path.exists(COMPARISON_FILE):
        st.error("Comparison report is missing. Run: python scripts/train_all_models.py")
    else:
        comparison = pd.read_csv(COMPARISON_FILE).set_index("Model")
        st.dataframe(comparison.style.format("{:.4f}"), width="stretch")
        st.bar_chart(comparison[["Accuracy", "F1 Score", "ROC AUC"]])
        best_model = comparison["Accuracy"].idxmax()
        st.success(f"{best_model} has the highest held-out accuracy: {comparison.loc[best_model, 'Accuracy']:.4f}.")
        st.download_button(
            "⬇️ Download comparison CSV",
            comparison.reset_index().to_csv(index=False).encode("utf-8"),
            "model_comparison_results.csv",
            "text/csv",
        )


elif page == "Feature Ablation":
    st.markdown("## 🧪 Leave-One-Feature-Out Accuracy")
    st.write(
        "Each row shows what happened after one feature was removed and the model was retrained. "
        "A negative Accuracy Change means removing that feature reduced accuracy."
    )
    if not os.path.exists(ABLATION_FILE):
        st.error("Feature-ablation report is missing. Run: python scripts/train_all_models.py")
    else:
        ablation = pd.read_csv(ABLATION_FILE)
        selected_model = st.selectbox("Model", ["ANN", "SVM", "KNN"], key="ablation_model")
        filtered = ablation[ablation["Model"] == selected_model].copy()
        st.dataframe(
            filtered.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Accuracy Change": "{:+.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC AUC": "{:.4f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        chart = filtered.set_index("Dropped Feature")[["Accuracy"]]
        st.bar_chart(chart)

        removed_only = ablation[ablation["Dropped Feature"] != "None (All 8 Features)"]
        most_useful = removed_only.loc[removed_only.groupby("Model")["Accuracy Change"].idxmin()]
        st.markdown("#### Feature whose removal hurt each model the most")
        st.dataframe(
            most_useful[["Model", "Dropped Feature", "Accuracy Change", "Effect of Removal"]]
            .style.format({"Accuracy Change": "{:+.4f}"}),
            hide_index=True,
            width="stretch",
        )
        st.info("A feature that increases accuracy when removed is not automatically useless; interactions and test-set variation should also be considered.")
        st.download_button(
            "⬇️ Download complete ablation CSV",
            ablation.to_csv(index=False).encode("utf-8"),
            "feature_ablation_results.csv",
            "text/csv",
        )


else:
    st.markdown("## 📋 Prediction History")
    history = load_history()
    if history.empty:
        st.info("No predictions yet.")
    else:
        left, right = st.columns([3, 1])
        with left:
            models = sorted(history["model"].dropna().unique())
            model_filter = st.multiselect("Filter by model", models, default=models)
        with right:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear history", width="stretch"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()

        filtered = history[history["model"].isin(model_filter)]
        st.dataframe(filtered.sort_values("timestamp", ascending=False), hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download history as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
        )
