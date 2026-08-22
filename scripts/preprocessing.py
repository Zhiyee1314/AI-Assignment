from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Data" / "diabetes.csv"
DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
ORIGINAL_ROW_COUNT = 768
DUPLICATE_BLOCK_END = 1000

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COL = "Outcome"

ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120),
}


def validate_dataset(data):
    required = FEATURE_ORDER + [TARGET_COL]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    checked = data[required].copy()

    for column in required:
        converted = pd.to_numeric(checked[column], errors="coerce")
        invalid = checked[column].notna() & converted.isna()

        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values "
                f"at CSV rows: {rows[:10]}"
            )

        checked[column] = converted

    invalid_outcome = ~checked[TARGET_COL].isin([0, 1])
    if invalid_outcome.any():
        raise ValueError("Outcome must contain only 0 or 1.")

    return checked


def load_dataset():
    data = pd.read_csv(DATA_PATH)
    return validate_dataset(data)


def get_verified_original_data(data):
    """
    The current file layout is:
    rows 0-767: original Pima patients
    rows 768-999: exact duplicates of original patients
    rows 1000-1999: added synthetic patients

    Only the original 768 patients are used for the scientifically
    valid held-out evaluation.
    """
    if len(data) != 2000:
        raise ValueError(
            "Expected the audited 2,000-row dataset. "
            "Review the split logic if the dataset has changed."
        )

    original = data.iloc[:ORIGINAL_ROW_COUNT].copy()
    duplicate_block = data.iloc[
        ORIGINAL_ROW_COUNT:DUPLICATE_BLOCK_END
    ].copy()

    columns = FEATURE_ORDER + [TARGET_COL]
    original_rows = set(
        map(tuple, original[columns].to_numpy())
    )

    duplicate_is_verified = duplicate_block[columns].apply(
        lambda row: tuple(row) in original_rows,
        axis=1,
    )

    if not duplicate_is_verified.all():
        raise ValueError(
            "Rows 769-1000 no longer match the audited duplicate layout. "
            "Do not continue until provenance is reviewed."
        )

    if original.duplicated(subset=columns).any():
        raise ValueError(
            "The original 768-patient section unexpectedly contains "
            "exact duplicates."
        )

    return original


def split_raw_dataset(data):
    original = get_verified_original_data(data)

    X = original[FEATURE_ORDER].copy()
    y = original[TARGET_COL].copy()

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def replace_invalid_zeros(frame):
    clean = frame[FEATURE_ORDER].copy()
    clean[ZERO_AS_MISSING_COLS] = clean[
        ZERO_AS_MISSING_COLS
    ].replace(0, np.nan)

    return clean


def new_imputer():
    return SimpleImputer(strategy="median")


def new_scaler():
    return StandardScaler()


def fit_preprocessors(X_train):
    clean_train = replace_invalid_zeros(X_train)

    imputer = new_imputer()
    train_imputed = imputer.fit_transform(clean_train)

    scaler = new_scaler()
    scaler.fit(train_imputed)

    return imputer, scaler


def transform_features(frame, imputer, scaler):
    clean = replace_invalid_zeros(frame)
    imputed = imputer.transform(clean)
    return scaler.transform(imputed)


def validate_patient_frame(frame):
    missing = [
        column
        for column in FEATURE_ORDER
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    validated = frame[FEATURE_ORDER].copy()

    for column in FEATURE_ORDER:
        converted = pd.to_numeric(
            validated[column],
            errors="coerce",
        )

        non_numeric = validated[column].notna() & converted.isna()
        if non_numeric.any():
            rows = (non_numeric[non_numeric].index + 2).tolist()
            raise ValueError(
                f"'{column}' contains non-numeric values "
                f"at CSV rows: {rows[:10]}"
            )

        validated[column] = converted

        minimum, maximum = VALID_RANGES[column]
        outside = converted.notna() & (
            (converted < minimum) | (converted > maximum)
        )

        if outside.any():
            rows = (outside[outside].index + 2).tolist()
            raise ValueError(
                f"'{column}' must be between {minimum} and {maximum}. "
                f"Invalid CSV rows: {rows[:10]}"
            )

    for integer_column in ["Pregnancies", "Age"]:
        values = validated[integer_column].dropna()
        if ((values % 1) != 0).any():
            raise ValueError(
                f"'{integer_column}' must contain whole numbers."
            )

    return validated
