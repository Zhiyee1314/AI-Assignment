"""Streamlit frontend for the trained ANN, KNN, and SVM models.

This file never retrains a model. It accepts raw medical values, validates
them, applies the saved imputer and 0-1 scaler, and loads existing .pkl files.
"""
import hashlib
import io
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from CSS.styles import inject_global_css
from scripts.preprocessing import (
    DATA_DIR,
    FEATURE_ORDER as FEATURES,
    MODELS_DIR,
    load_dataset,
    split_raw_dataset,
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

# ---------------------------------------------------------------
# MODIFIED: input widget config for the 4 selected features only.
# (label, min, max, default, step, help)
# ---------------------------------------------------------------
INPUT_FIELDS = [
    ("Pregnancies", "Pregnancies", 0, 20, 3, 1, "Number of times pregnant."),
    ("Glucose", "Glucose", 0, 300, 162, 1, "Plasma glucose concentration (mg/dL)."),
    ("BMI", "BMI", 0.0, 70.0, 30.0, 0.1, "Body Mass Index (weight in kg / height in m^2)."),
    ("Age", "Age", 1, 120, 35, 1, "Age in years."),
]


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


def compute_current_model_comparison(available_models, imputer, scaler):
    """Evaluate the currently loaded artifacts on one fair unseen test set."""
    data = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data)
    X_test = transform_features(X_test_raw, imputer, scaler)

    rows = []
    for label in ["ANN", "KNN", "SVM"]:
        if label not in available_models:
            continue

        model = load_artifact(available_models[label])
        try:
            prediction = np.asarray(model.predict(X_test), dtype=int)
        except NotFittedError as error:
            raise NotFittedError(
                f"{label} model is not fitted. Run scripts/{label.title()}_Model.py "
                "and replace the matching file in models/."
            ) from error

        probability = get_positive_probability(model, X_test)
        tn, fp, fn, tp = confusion_matrix(
            y_test, prediction, labels=[0, 1]
        ).ravel()
        rows.append({
            "Model": label,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test, prediction, zero_division=0
            ),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1 Score": f1_score(y_test, prediction, zero_division=0),
            "ROC-AUC": (
                np.nan
                if probability is None
                else roc_auc_score(y_test, probability)
            ),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Training Patients": len(y_train),
            "Test Patients": len(y_test),
        })

    if not rows:
        raise FileNotFoundError("No fitted model artifacts are available.")

    return pd.DataFrame(rows).set_index("Model")


def model_artifact_signature(available_models):
    """Return a stable signature so persisted batch results refresh after retraining."""
    paths = [IMPUTER_FILE, SCALER_FILE] + list(available_models.values())
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )


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
            "Prediction History",
        ],
        format_func=lambda value: {
            "Predict": "🩺  Predict",
            "Batch CSV": "📄 Upload Batch CSV",
            "Compare Models": "📊  Compare Models",
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

    # -----------------------------------------------------------
    # MODIFIED: single shared 4-input form (Pregnancies, Glucose,
    # BMI, Age) since every model now uses the same 4 features.
    # -----------------------------------------------------------
    col1, col2 = st.columns(2)
    input_columns = [col1, col2, col1, col2]
    input_values = {}
    for (key, label, min_val, max_val, default_val, step, help_text), col in zip(
        INPUT_FIELDS, input_columns
    ):
        with col:
            input_values[key] = st.number_input(
                label, min_val, max_val, default_val, step, help=help_text
            )

    if st.button(
        f"🩺 Predict with {model_choice}",
        type="primary",
        width="stretch",
    ):
        patient = pd.DataFrame([{
            "Pregnancies": input_values["Pregnancies"],
            "Glucose": input_values["Glucose"],
            "BMI": input_values["BMI"],
            "Age": input_values["Age"],
        }])

        try:
            validated = validate_patient_frame(patient)
            scaled = transform_features(validated, imputer, scaler)
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
                    scaled, columns=FEATURES, index=validated.index
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
        except NotFittedError:
            st.error(
                f"The saved {model_choice} model is not fitted. Run "
                f"scripts/{model_choice.title()}_Model.py, confirm it saves to "
                f"models/{model_choice.lower()}_model.pkl, then restart the app."
            )
        except ValueError as error:
            st.error(str(error))


elif page == "Batch CSV":
    st.markdown("## 📄 Batch Patient Prediction")
    st.write(
        "Upload a CSV containing the four raw medical-value columns "
        "(Pregnancies, Glucose, BMI, Age). Every patient will be "
        "evaluated by every available model."
    )
    template = pd.DataFrame(columns=FEATURES)
    st.download_button(
        "⬇️ Download empty CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_patient_template.csv",
        mime="text/csv",
    )

    if "batch_uploader_version" not in st.session_state:
        st.session_state.batch_uploader_version = 0

    uploaded_file = st.file_uploader(
        "Upload patient CSV",
        type=["csv"],
        key=f"batch_csv_upload_{st.session_state.batch_uploader_version}",
    )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        if uploaded_hash != st.session_state.get("batch_upload_hash"):
            st.session_state.batch_upload_bytes = uploaded_bytes
            st.session_state.batch_upload_name = uploaded_file.name
            st.session_state.batch_upload_hash = uploaded_hash
            st.session_state.pop("batch_results", None)
            st.session_state.pop("batch_result_signature", None)

    saved_upload = st.session_state.get("batch_upload_bytes")
    current_signature = model_artifact_signature(available_models)
    result_signature = (
        st.session_state.get("batch_upload_hash"),
        current_signature,
    )

    if saved_upload is not None and (
        "batch_results" not in st.session_state
        or st.session_state.get("batch_result_signature") != result_signature
    ):
        try:
            uploaded = pd.read_csv(io.BytesIO(saved_upload))
            extra = [column for column in uploaded.columns if column not in FEATURES]
            if extra:
                st.info("Extra columns ignored: " + ", ".join(extra))

            patients, predictions = predict_all_models(
                uploaded, available_models, imputer, scaler
            )
            duplicate_count = int(patients.duplicated().sum())

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

            st.session_state.batch_results = results
            st.session_state.batch_result_signature = result_signature
            st.session_state.batch_duplicate_count = duplicate_count
        except NotFittedError as error:
            st.error(
                "A saved model is not fitted. Run all three separate training "
                f"scripts and replace the model artifacts. Details: {error}"
            )
        except (
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error:
            st.error(f"Unable to process CSV: {error}")

    if "batch_results" in st.session_state:
        results = st.session_state.batch_results
        upload_name = st.session_state.get(
            "batch_upload_name", "uploaded patient CSV"
        )
        st.success(
            f"Processed {len(results)} patient(s) from {upload_name}. "
            "These results remain available when you visit another page."
        )
        duplicate_count = st.session_state.get("batch_duplicate_count", 0)
        if duplicate_count:
            st.warning(
                f"{duplicate_count} duplicate patient row(s) were detected "
                "and preserved to keep the uploaded row order."
            )
        st.dataframe(results, width="stretch", hide_index=True)

        download_col, clear_col = st.columns([3, 1])
        with download_col:
            st.download_button(
                "⬇️ Download batch prediction results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_batch_predictions.csv",
                mime="text/csv",
                width="stretch",
            )
        with clear_col:
            if st.button("🗑️ Clear batch", width="stretch"):
                for key in [
                    "batch_upload_bytes",
                    "batch_upload_name",
                    "batch_upload_hash",
                    "batch_results",
                    "batch_result_signature",
                    "batch_duplicate_count",
                ]:
                    st.session_state.pop(key, None)
                st.session_state.batch_uploader_version += 1
                st.rerun()


elif page == "Compare Models":
    st.markdown("## 📊 Model Comparison")
    st.caption(
        "The currently loaded ANN, KNN and SVM artifacts are evaluated "
        "on the same held-out test patients, so every section below is "
        "a fair, apples-to-apples comparison."
    )

    # ================================================================
    # SECTION 1: Metrics Table
    # ================================================================
    st.markdown("### 1️⃣ Metrics Table")
    try:
        comparison = compute_current_model_comparison(
            available_models, imputer, scaler,
        )
        metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

        st.dataframe(
            comparison[metric_columns].style.format("{:.4f}"),
            width="stretch",
        )

        best_model = comparison["Accuracy"].idxmax()
        st.success(
            f"**{best_model}** has the highest accuracy "
            f"({comparison.loc[best_model, 'Accuracy']:.4f})."
        )

        st.download_button(
            "⬇️ Download comparison table as CSV",
            data=comparison.round(4).to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
            key="download_metrics_table",
        )
    except (FileNotFoundError, NotFittedError) as error:
        st.error(str(error))
        comparison = None
    except (ValueError, KeyError) as error:
        st.error(f"Unable to calculate the model comparison: {error}")
        comparison = None

    st.divider()

    # ================================================================
    # SECTION 2: Grouped Metrics Comparison Chart (0-100%)
    # ================================================================
    st.markdown("### 2️⃣ Performance Metrics Comparison")
    if comparison is not None:
        import plotly.graph_objects as go

        # Multiply values by 100 for percentage scale (0-100)
        chart_data = comparison[metric_columns] * 100

        # Colors matching the target image (Blue, Green, Orange, Yellow, Purple)
        color_map = {
            "Accuracy": "#5B9BD5",
            "Precision": "#70AD47",
            "Recall": "#ED7D31",
            "F1 Score": "#FFC000",
            "ROC-AUC": "#B4A7D6",
        }

        fig = go.Figure()

        # Add a bar for each metric
        for metric in metric_columns:
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data[metric],
                    name=metric.replace("F1 Score", "F1-score"),
                    marker_color=color_map[metric],
                )
            )

        fig.update_layout(
            barmode="group",
            yaxis=dict(
                range=[0, 100],
                dtick=25,
                showgrid=True,
                gridcolor="#EBEBEB",
                gridwidth=1,
                zeroline=False,
            ),
            xaxis=dict(
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=13),
            ),
            margin=dict(l=20, r=20, t=20, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=450,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Metrics table above failed to load, so this chart cannot be shown.")


elif page == "Prediction History":
    st.markdown("## 📋 Prediction History")

    history = load_history()

    if history.empty:
        st.info(
            "No successful single-patient predictions have been saved yet. "
            "Go to Predict, select a model and complete a prediction first."
        )

    else:
        col_a, col_b = st.columns([3, 1])

        with col_a:
            model_choices = sorted(
                history["model"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_models = st.multiselect(
                "Filter by model",
                options=model_choices,
                default=model_choices,
            )

        with col_b:
            st.write("")

            if st.button(
                "🗑️ Clear history",
                width="stretch",
            ):
                # Create a new empty history table directly.
                st.session_state[
                    "prediction_history_df"
                ] = pd.DataFrame(
                    columns=[
                        "timestamp",
                        "model",
                        "probability",
                        "prediction",
                    ] + FEATURES
                )

                # Delete the locally saved history file.
                try:
                    HISTORY_FILE.unlink(
                        missing_ok=True
                    )

                except OSError as error:
                    st.warning(
                        "The session history was cleared, "
                        "but the saved history file could "
                        f"not be deleted: {error}"
                    )

                # Refresh the page immediately.
                st.rerun()

        filtered = history[
            history["model"].isin(
                selected_models
            )
        ]

        st.dataframe(
            filtered.sort_values(
                "timestamp",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download prediction history",
            data=filtered.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
