"""Shared data validation and preprocessing for all three model scripts.

This module does not train ANN, KNN, or SVM.  It only guarantees that the
three independently owned training scripts use the same patients, feature
order, missing-value rules, and 0-1 normalization.
"""

from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
 
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "diabetes.csv"
 
RANDOM_STATE = 42
TEST_SIZE = 0.20
 
# ---------------------------------------------------------------
# MODIFIED: reduced from 8 features to these 4.
# ---------------------------------------------------------------
FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "Age",
]
 
TARGET_COL = "Outcome"
 
# Only Glucose and BMI remain from the original 5 zero-as-missing columns
# (BloodPressure, SkinThickness, Insulin were dropped along with the
# feature reduction and are no longer part of FEATURE_ORDER).
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BMI",
]
 
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BMI": (0, 70),
    "Age": (1, 120),
}
 
# This requested configuration uses every row in the current 2,000-row CSV.
# The split is still performed before the imputer/scaler are fitted, so test
# statistics do not leak into preprocessing. Because the CSV contains generated
# and duplicate records, its score must be described as duplicate/generated-
# inclusive evaluation rather than independent real-patient validation.
EXPECTED_AUDITED_ROWS = 2000
 
 
def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the training CSV schema, numeric types, and target labels."""
    required = FEATURE_ORDER + [TARGET_COL]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )
 
    checked = data[required].copy()
    for column in required:
        converted = pd.to_numeric(checked[column], errors="coerce")
        invalid = checked[column].notna() & converted.isna()
        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values at CSV rows "
                f"{rows[:10]}."
            )
        checked[column] = converted
 
    if checked[FEATURE_ORDER].isna().all(axis=0).any():
        empty_columns = checked[FEATURE_ORDER].columns[
            checked[FEATURE_ORDER].isna().all(axis=0)
        ].tolist()
        raise ValueError(
            "Dataset contains completely empty features: "
            + ", ".join(empty_columns)
        )
 
    invalid_outcome = ~checked[TARGET_COL].isin([0, 1])
    if invalid_outcome.any():
        raise ValueError("Outcome must contain only 0 or 1.")
 
    return checked
 
 
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the project's diabetes dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return validate_dataset(pd.read_csv(path))
 
 
def get_verified_original_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return all 2,000 rows for the requested inclusive evaluation.
 
    The legacy function name is retained so existing app/model imports continue
    to work without modification.
    """
    if len(data) != EXPECTED_AUDITED_ROWS:
        raise ValueError(
            f"Expected the audited {EXPECTED_AUDITED_ROWS}-row dataset, but "
            f"found {len(data)} rows. Review the provenance before training."
        )
 
    return data[FEATURE_ORDER + [TARGET_COL]].copy()
 
 
def split_raw_dataset(data: pd.DataFrame):
    """Create the common stratified split using all 2,000 dataset rows."""
    evaluation_data = get_verified_original_data(data)
    X = evaluation_data[FEATURE_ORDER].copy()
    y = evaluation_data[TARGET_COL].astype(int).copy()
 
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
 
 
def replace_invalid_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert physiologically invalid zero measurements to missing values."""
    clean = frame[FEATURE_ORDER].copy()
    clean[ZERO_AS_MISSING_COLS] = clean[ZERO_AS_MISSING_COLS].replace(
        0, np.nan
    )
    return clean
 
 
def new_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")
 
 
def new_scaler() -> MinMaxScaler:
    # clip=True keeps future/unseen medical values inside the required 0-1
    # range even when they fall outside the minimum/maximum seen in training.
    return MinMaxScaler(feature_range=(0, 1), clip=True)
 
 
def fit_preprocessors(X_train: pd.DataFrame):
    """Fit preprocessing on training patients only."""
    clean_train = replace_invalid_zeros(X_train)
    imputer = new_imputer()
    train_imputed = imputer.fit_transform(clean_train)
    scaler = new_scaler()
    scaler.fit(train_imputed)
    return imputer, scaler
 
 
def transform_features(frame, imputer, scaler) -> np.ndarray:
    """Apply the saved training preprocessing in the required feature order."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame, columns=FEATURE_ORDER)
    clean = replace_invalid_zeros(frame)
    imputed = imputer.transform(clean)
    scaled = scaler.transform(imputed)
    return np.asarray(scaled, dtype=float)
 
 
def validate_patient_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate single-patient or batch raw medical values."""
    missing = [column for column in FEATURE_ORDER if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
 
    validated = frame[FEATURE_ORDER].copy()
    for column in FEATURE_ORDER:
        converted = pd.to_numeric(validated[column], errors="coerce")
        non_numeric = validated[column].notna() & converted.isna()
        if non_numeric.any():
            rows = (non_numeric[non_numeric].index + 2).tolist()
            raise ValueError(
                f"'{column}' contains non-numeric values at CSV rows "
                f"{rows[:10]}."
            )
 
        minimum, maximum = VALID_RANGES[column]
        outside = converted.notna() & (
            (converted < minimum) | (converted > maximum)
        )
        if outside.any():
            rows = (outside[outside].index + 2).tolist()
            raise ValueError(
                f"'{column}' must be between {minimum} and {maximum}. "
                f"Invalid CSV rows: {rows[:10]}."
            )
        validated[column] = converted
 
    for integer_column in ["Pregnancies", "Age"]:
        values = validated[integer_column].dropna()
        if ((values % 1) != 0).any():
            raise ValueError(f"'{integer_column}' must contain whole numbers.")
 
    return validated
