# Diabetes Risk Predictor

Streamlit application that compares ANN, SVM and KNN diabetes-risk classifiers.

## Features

- Single-patient prediction using normal medical values.
- Backend preprocessing converts all eight model inputs to the range 0.00-1.00.
- Batch CSV upload runs ANN, SVM and KNN for every patient.
- Downloadable batch results and prediction history.
- Fair model comparison using the same split and preprocessing.
- Leave-one-feature-out ablation page and downloadable CSV report.

## Eight required patient features

The column names and order used by every model are:

1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`

The training dataset additionally contains the target column `Outcome`.

## Preprocessing

1. Exact duplicate patient rows are removed before the train/test split. The
   current 1,000-row CSV contains 232 duplicates, so 768 unique rows are used.
2. Zero is treated as missing for Glucose, BloodPressure, SkinThickness,
   Insulin and BMI.
3. A median `SimpleImputer` is fitted only on the training split.
4. `MinMaxScaler(feature_range=(0, 1), clip=True)` is fitted only on the
   imputed training split.
5. ANN, SVM and KNN receive the same processed values and the same 80/20 split.

The frontend continues to accept understandable medical values. Conversion to
0.00-1.00 happens automatically before prediction. Scaling alone does not
guarantee higher accuracy; the saved reports contain the measured results.

## Train all models and generate reports

Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/train_all_models.py
```

This regenerates:

- `models/imputer.pkl`
- `models/scaler.pkl`
- `models/ann_model.pkl`
- `models/svm_model.pkl`
- `models/knn_model.pkl`
- `models/training_metadata.json`
- `Data/model_comparison_results.csv`
- `Data/feature_ablation_results.csv`

The older `Ann_Model.py`, `Svm_Model.py` and `Knn_Model.py` filenames remain as
compatibility entry points. Running any one of them retrains all three models
together so their preprocessing cannot become inconsistent.

## Run the Streamlit app

```bash
streamlit run "Diabates Prediction.py"
```

## Batch CSV format

Open **Batch CSV** in the app and download the template. `PatientID` is
optional, but all eight feature columns are required. Extra columns are kept in
the downloaded result file. The app validates missing, non-numeric and
out-of-range values before running all three models.

The result CSV adds:

- `ANN_Prediction` and `ANN_Probability_Percent`
- `SVM_Prediction` and `SVM_Probability_Percent`
- `KNN_Prediction` and `KNN_Probability_Percent`

## Feature ablation report

For each algorithm, the pipeline first records the baseline using all eight
features. It then removes one feature, refits that subset's imputer and scaler,
retrains the model, and evaluates it on the same held-out patient indices.

- Negative `Accuracy Change`: removal reduced accuracy.
- Positive `Accuracy Change`: removal increased accuracy on this test split.
- Zero `Accuracy Change`: accuracy was unchanged.

This is more valid than deleting a value only during prediction, because a
model must be retrained when its input feature set changes.

## Tests

```bash
python -m unittest discover -s tests -v
```
