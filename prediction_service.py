"""Reusable preprocessing, validation and prediction helpers for Streamlit."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

INPUT_BOUNDS = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120),
}


def validate_patient_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return the required feature columns as numeric values or raise a clear error."""
    if data.empty:
        raise ValueError("The uploaded CSV contains no patient rows.")

    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    numeric = data[FEATURES].copy()
    invalid_cells = []
    for column in FEATURES:
        converted = pd.to_numeric(numeric[column], errors="coerce")
        bad_rows = converted[converted.isna() & numeric[column].notna()].index.tolist()
        invalid_cells.extend(f"row {row + 2}, {column}" for row in bad_rows[:5])
        numeric[column] = converted

    if invalid_cells:
        raise ValueError("Non-numeric value found at " + "; ".join(invalid_cells))
    if numeric.isna().any().any():
        locations = np.argwhere(numeric.isna().to_numpy())
        examples = [f"row {row + 2}, {FEATURES[column]}" for row, column in locations[:5]]
        raise ValueError("Blank value found at " + "; ".join(examples))

    out_of_range = []
    for column, (minimum, maximum) in INPUT_BOUNDS.items():
        mask = (numeric[column] < minimum) | (numeric[column] > maximum)
        out_of_range.extend(
            f"row {row + 2}, {column}={numeric.loc[row, column]} (allowed {minimum}-{maximum})"
            for row in numeric.index[mask][:5]
        )
    if out_of_range:
        raise ValueError("Value outside the accepted range: " + "; ".join(out_of_range))
    return numeric


def preprocess_patient_data(data: pd.DataFrame, imputer, scaler) -> pd.DataFrame:
    """Convert normal medical values into model inputs restricted to 0.00-1.00."""
    numeric = validate_patient_data(data)
    clean = numeric.copy()
    for column in ZERO_AS_MISSING_COLS:
        clean[column] = clean[column].replace(0, np.nan)

    imputed = imputer.transform(clean)
    scaled = scaler.transform(imputed)
    # MinMaxScaler(clip=True) already clips. np.clip is a defensive guarantee
    # that backend model inputs stay inside the requested interval.
    scaled = np.clip(scaled, 0.0, 1.0)
    return pd.DataFrame(scaled, columns=FEATURES, index=data.index)


def positive_probability(model, model_input: pd.DataFrame) -> np.ndarray:
    model_values = model_input.to_numpy() if isinstance(model_input, pd.DataFrame) else model_input
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(model_values)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(model_values), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
    return np.asarray(model.predict(model_values), dtype=float)


def predict_all_models(
    original_data: pd.DataFrame,
    models: dict,
    imputer,
    scaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every patient with ANN, SVM and KNN and return results plus scaled inputs."""
    model_input = preprocess_patient_data(original_data, imputer, scaler)
    results = original_data.copy().reset_index(drop=True)
    model_values = model_input.to_numpy()

    for name in ["ANN", "SVM", "KNN"]:
        if name not in models:
            raise ValueError(f"The {name} model file is not available.")
        model = models[name]
        prediction = model.predict(model_values)
        probability = positive_probability(model, model_input)
        results[f"{name}_Prediction"] = np.where(prediction == 1, "High Risk", "Low Risk")
        results[f"{name}_Probability_Percent"] = np.round(probability * 100, 2)

    return results, model_input
