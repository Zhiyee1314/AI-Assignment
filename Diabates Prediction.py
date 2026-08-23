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
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
 
 
@st.cache_data
def compute_confusion_matrices(_available_models_tuple, _imputer, _scaler):
    """Confusion matrix (as a numpy array) for every currently available model,
    evaluated on the same held-out test set used everywhere else."""
    available_models = dict(_available_models_tuple)
 
    data = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data)
    X_test = transform_features(X_test_raw, _imputer, _scaler)
 
    matrices = {}
    for label, path in available_models.items():
        model = load_artifact(path)
        prediction = np.asarray(model.predict(X_test), dtype=int)
        matrices[label] = confusion_matrix(y_test, prediction, labels=[0, 1])
 
    return matrices
 
 
def plot_confusion_matrix_figure(model_label, cm, accent):
    """Green-yellow styled confusion matrix (rows=Predicted, cols=Actual)."""
    class_labels = ["No Diabetes", "Diabetes"]
    cm_t = cm.T
    col_sums = cm_t.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_percent = (cm_t / col_sums) * 100
 
    annot = np.empty_like(cm_t).astype(str)
    for i in range(cm_t.shape[0]):
        for j in range(cm_t.shape[1]):
            annot[i, j] = f"{cm_percent[i, j]:.1f}%\n{cm_t[i, j]}"
 
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
 
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sns.heatmap(
        cm_t, annot=annot, fmt='', cmap='YlGn_r', cbar=False,
        linewidths=1, linecolor='black',
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={"size": 11}, ax=ax
    )
    ax.set_title(f"{model_label} — Accuracy: {acc * 100:.2f}%", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Labels")
    ax.set_ylabel("Predicted Labels")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig
 
 
@st.cache_data
def compute_roc_curves(_available_models_tuple, _imputer, _scaler):
    """ROC curve points + AUC for every model that supports predict_proba."""
    available_models = dict(_available_models_tuple)
 
    data = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = split_raw_dataset(data)
    X_test = transform_features(X_test_raw, _imputer, _scaler)
 
    curves = {}
    for label, path in available_models.items():
        model = load_artifact(path)
        probability = get_positive_probability(model, X_test)
        if probability is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probability)
        roc_auc_value = auc(fpr, tpr)
        curves[label] = (fpr, tpr, roc_auc_value)
 
    return curves
 
 
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
    # SECTION 2: Metrics Bar Chart
    # ================================================================
    st.markdown("### 2️⃣ Metrics Bar Chart")
    if comparison is not None:
        st.markdown("**All metrics, grouped by model**")
        st.bar_chart(comparison[metric_columns])
 
        st.markdown("**Accuracy only**")
        st.bar_chart(comparison[["Accuracy"]])
    else:
        st.info("Metrics table above failed to load, so the bar chart cannot be shown.")
 
    st.divider()
 
    # ================================================================
    # SECTION 3: Confusion Matrix per model
    # ================================================================
    st.markdown("### 3️⃣ Confusion Matrix (per model)")
    try:
        matrices = compute_confusion_matrices(
            tuple(available_models.items()), imputer, scaler
        )
        cm_cols = st.columns(len(matrices))
        for col, (label, cm) in zip(cm_cols, matrices.items()):
            with col:
                fig = plot_confusion_matrix_figure(label, cm, MODEL_ACCENT[label])
                st.pyplot(fig, width="content")
    except (FileNotFoundError, NotFittedError) as error:
        st.error(str(error))
    except (ValueError, KeyError) as error:
        st.error(f"Unable to compute confusion matrices: {error}")
 
    st.divider()
 
    # ================================================================
    # SECTION 4: ROC Curve Comparison
    # ================================================================
    st.markdown("### 4️⃣ ROC Curve Comparison")
    try:
        curves = compute_roc_curves(
            tuple(available_models.items()), imputer, scaler
        )
        if not curves:
            st.info("None of the loaded models expose predict_proba, so no ROC curve can be drawn.")
        else:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            for label, (fpr, tpr, roc_auc_value) in curves.items():
                ax.plot(
                    fpr, tpr, linewidth=2,
                    color=MODEL_ACCENT.get(label, None),
                    label=f"{label} (AUC = {roc_auc_value:.4f})"
                )
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess (AUC = 0.5)")
            ax.set_title("ROC Curve Comparison", fontsize=13, fontweight='bold')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            fig.tight_layout()
            st.pyplot(fig, width="content")
    except (FileNotFoundError, NotFittedError) as error:
        st.error(str(error))
    except (ValueError, KeyError) as error:
        st.error(f"Unable to compute ROC curves: {error}")
 
    st.divider()
 
    # ================================================================
    # SECTION 5: Feature Ablation Comparison
    # ================================================================
    st.markdown("### 5️⃣ Feature Ablation Comparison")
    st.caption(
        "Each model was retrained after removing one feature at a time. "
        "A larger negative 'Change' means that feature mattered more to "
        "that model's accuracy."
    )
    ablation_df, missing_ablation = load_ablation_comparison()
 
    if missing_ablation:
        st.warning(
            "Ablation results not found for: " + ", ".join(missing_ablation) +
            ". Run the matching scripts/*_Model.py so it saves "
            "Data/<model>_ablation.csv."
        )
 
    if ablation_df is not None:
        st.dataframe(
            ablation_df.style.format(
                {c: "{:.4f}" for c in ablation_df.columns if c != "Removed Feature"}
            ),
            width="stretch",
            hide_index=True,
        )
 
        change_columns = [c for c in ablation_df.columns if c.endswith("Change")]
        if change_columns:
            chart_df = ablation_df.set_index("Removed Feature")[change_columns]
            st.bar_chart(chart_df)
 
        st.download_button(
            "⬇️ Download feature ablation comparison as CSV",
            data=ablation_df.round(4).to_csv(index=False).encode("utf-8"),
            file_name="feature_ablation_comparison.csv",
            mime="text/csv",
            key="download_ablation",
        )
    else:
        st.info("No ablation CSV files were found for any model yet.")
 
 
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
