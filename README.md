# Diabetes Risk Predictor — Team Setup Guide

## 1. Folder structure (everyone needs this exact layout)

```
diabetes_app/
├── app.py
├── train_ann.py          <- run this locally to GENERATE the .pkl files
├── requirements.txt
├── imputer.pkl           <- SHARED (same file for everyone)
├── scaler.pkl            <- SHARED (same file for everyone)
├── ann_model.pkl          <- yours
├── svm_model.pkl         <- teammate's (optional)
├── knn_model.pkl         <- teammate's (optional)
└── prediction_history.csv    <- auto-created after first prediction
```

All `.pkl` files are saved with **joblib** (`joblib.dump` / `joblib.load`),
matching your original app.py — not plain `pickle`.

## 2. Why imputer.pkl / scaler.pkl are shared but model.pkl is not

- `imputer.pkl` stores the median/strategy values used to fill missing data
  during training. If SVM/KNN refit their own imputer on a different split,
  the numbers going into their models won't mean the same thing as the
  numbers going into your ANN — the 3 models stop being comparable.
- `scaler.pkl` stores the mean/std (or min/max) used to standardize features.
  Same reasoning: everyone must scale with the exact same numbers.
- `*_model.pkl` is just the trained weights for one specific algorithm — this
  one is personal to whoever trained it. Each teammate exports their own.

**How your teammates get the shared files:** you send them your
`imputer.pkl` + `scaler.pkl` (or the cleaned training CSV so they fit their
own model on the same preprocessed data), and they train/export
`svm_model.pkl` / `knn_model.pkl` using those exact same imputer/scaler
before predicting.

## 3. Adding a new teammate's model — zero code changes needed

The app auto-detects which model files exist in the folder. To add KNN:

1. Teammate drops `knn_model.pkl` into the same folder as `app.py`.
2. Restart the app (or Streamlit will auto-reload).
3. "KNN" now appears automatically in the sidebar dropdown.

This works because `MODEL_FILES` in `app.py` already lists the three
expected filenames, and `get_available_models()` only shows the ones that
actually exist on disk:

```python
MODEL_FILES = {
    "ANN": "ann_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
}
```

If a teammate wants to use a different filename, just add a new line to
this dict, e.g. `"Random Forest": "rf_model.pkl"`.

## 4. Model selection page (good for demos)

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sidebar lets you pick which model runs the prediction — perfect for
switching live during a presentation ("now let's compare with SVM...").

## 5. Prediction history feature

- Every prediction (regardless of which model was used) is appended to
  `prediction_history.csv`, so it survives app restarts.
- Go to the **"Prediction History"** page in the sidebar to see:
  - a bar chart of High Risk vs Low Risk counts, grouped by model
  - a bar chart of the probability of each past prediction
  - the full history table
  - filter by model, clear history, or download as CSV

## 6. How to generate the .pkl files yourself (train_ann.py)

`train_ann.py` is a full working example of the script that CREATES the
3 files `app.py` loads. Run it locally:

```bash
python train_ann.py
```

It will:
1. Load your CSV (`diabetes.csv` — change the filename to match yours).
2. Replace 0s with `NaN` in Glucose/BloodPressure/SkinThickness/Insulin/BMI
   (same as your original app.py logic).
3. Split train/test, then `fit_transform` the imputer + scaler on train
   only (this avoids data leakage from the test set).
4. Train an `MLPClassifier` (swap this for your actual ANN architecture).
5. Save all 3 objects with `joblib.dump(...)`.

For teammates doing SVM/KNN: they should **not** re-fit their own
imputer/scaler. They load the ones you give them and call `.transform()`
(not `.fit_transform()`) before fitting their own model — this is spelled
out in the comment block at the bottom of `train_ann.py`.

## 7. Notes / assumptions I made

- Feature order: Pregnancies, Glucose, BloodPressure, SkinThickness,
  Insulin, BMI, DiabetesPedigreeFunction, Age (matches your original
  app.py exactly).
- `app.py` now uses `joblib.load(...)` and treats 0 as missing for
  Glucose/BloodPressure/SkinThickness/Insulin/BMI before imputing —
  copied directly from your uploaded script.
- Prediction calls `model.predict()` first (like your original), then
  `model.predict_proba()[:, 1]` if the model supports it, falling back to
  the raw prediction if it doesn't (e.g. an SVM without
  `probability=True`).
