import warnings
import tkinter as tk
from tkinter import ttk, messagebox
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
 
warnings.filterwarnings("ignore", category=UserWarning)
 
CSV_PATH = "diabetes.csv"
 
# (field_key, display_label, example/hint)
FIELDS = [
    ("Pregnancies", "Number of Pregnancies", "e.g. 2"),
    ("Glucose", "Glucose Level (mg/dL)", "e.g. 120"),
    ("BloodPressure", "Diastolic Blood Pressure (mmHg)", "e.g. 70"),
    ("SkinThickness", "Skin Thickness (mm)", "e.g. 20"),
    ("Insulin", "Insulin Level (mu U/ml)", "e.g. 80"),
    ("BMI", "Body Mass Index (BMI)", "e.g. 24.5"),
    ("DiabetesPedigreeFunction", "Diabetes Pedigree Function", "e.g. 0.5"),
    ("Age", "Age (years)", "e.g. 30"),
]
 
 
# ============================================
# Model training (runs once at startup)
# ============================================
def train_model(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
 
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
 
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    param_grid = {"n_neighbors": range(1, 31)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)
    best_k = grid.best_params_["n_neighbors"]
 
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
 
    y_pred = model.predict(X_test_scaled)
 
    results = {
        "best_k": best_k,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "dataset_shape": df.shape,
    }
 
    return model, scaler, feature_names, results
 
 
# ============================================
# GUI Application
# ============================================
class DiabetesApp(tk.Tk):
    def __init__(self):
        super().__init__()
 
        self.title("Diabetes Prediction System - KNN")
        self.geometry("720x640")
        self.configure(bg="#f4f6f8")
 
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[15, 8])
 
        # Header
        header = tk.Label(
            self, text="Diabetes Prediction System (KNN)",
            font=("Segoe UI", 16, "bold"), bg="#2c3e50", fg="white", pady=14
        )
        header.pack(fill="x")
 
        # Train the model before building the rest of the UI
        try:
            self.model, self.scaler, self.feature_names, self.results = train_model()
        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found",
                f"Could not find '{CSV_PATH}'.\nPlease place the dataset file in the same folder as this script."
            )
            self.destroy()
            return
 
        # Tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
 
        self.performance_tab = ttk.Frame(notebook)
        self.predict_tab = ttk.Frame(notebook)
        notebook.add(self.performance_tab, text="Model Performance")
        notebook.add(self.predict_tab, text="Predict")
 
        self.build_performance_tab()
        self.build_predict_tab()
 
    # ------------------------------------------------
    def build_performance_tab(self):
        frame = self.performance_tab
 
        info = tk.Label(
            frame,
            text=(f"Dataset: {self.results['dataset_shape'][0]} rows, "
                  f"{self.results['dataset_shape'][1]} columns   |   Best K = {self.results['best_k']}"),
            font=("Segoe UI", 10, "italic"), pady=8
        )
        info.pack()
 
        # ---- Metrics table ----
        tk.Label(frame, text="Overall Metrics", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        metrics_tree = ttk.Treeview(frame, columns=("Metric", "Score"), show="headings", height=4)
        metrics_tree.heading("Metric", text="Metric")
        metrics_tree.heading("Score", text="Score")
        metrics_tree.column("Metric", width=200, anchor="w")
        metrics_tree.column("Score", width=150, anchor="center")
        for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                            ("Recall", "recall"), ("F1 Score", "f1")]:
            metrics_tree.insert("", "end", values=(label, f"{self.results[key]:.4f}"))
        metrics_tree.pack(padx=15, fill="x")
 
        # ---- Confusion matrix table ----
        tk.Label(frame, text="Confusion Matrix", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        cm = self.results["confusion_matrix"]
        cm_tree = ttk.Treeview(frame, columns=("Label", "PredNo", "PredYes"), show="headings", height=2)
        cm_tree.heading("Label", text="")
        cm_tree.heading("PredNo", text="Predicted: No Diabetes")
        cm_tree.heading("PredYes", text="Predicted: Diabetes")
        cm_tree.column("Label", width=180, anchor="w")
        cm_tree.column("PredNo", width=180, anchor="center")
        cm_tree.column("PredYes", width=150, anchor="center")
        cm_tree.insert("", "end", values=("Actual: No Diabetes", cm[0][0], cm[0][1]))
        cm_tree.insert("", "end", values=("Actual: Diabetes", cm[1][0], cm[1][1]))
        cm_tree.pack(padx=15, fill="x")
 
        # ---- Classification report table ----
        tk.Label(frame, text="Classification Report", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        report_tree = ttk.Treeview(
            frame, columns=("Class", "Precision", "Recall", "F1", "Support"),
            show="headings", height=3
        )
        for col, width in [("Class", 160), ("Precision", 90), ("Recall", 90), ("F1", 90), ("Support", 90)]:
            report_tree.heading(col, text=col)
            report_tree.column(col, width=width, anchor="center")
        report_tree.column("Class", anchor="w")
 
        report = self.results["report"]
        for label, name in [("0", "No Diabetes (0)"), ("1", "Diabetes (1)")]:
            row = report[label]
            report_tree.insert("", "end", values=(
                name, f"{row['precision']:.2f}", f"{row['recall']:.2f}",
                f"{row['f1-score']:.2f}", int(row['support'])
            ))
        wavg = report["weighted avg"]
        report_tree.insert("", "end", values=(
            "Weighted Avg", f"{wavg['precision']:.2f}", f"{wavg['recall']:.2f}",
            f"{wavg['f1-score']:.2f}", int(wavg['support'])
        ))
        report_tree.pack(padx=15, fill="x")
 
    # ------------------------------------------------
    def build_predict_tab(self):
        frame = self.predict_tab
 
        tk.Label(
            frame, text="Enter Patient Details", font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=20, pady=(15, 10))
 
        form_frame = tk.Frame(frame)
        form_frame.pack(padx=20, fill="x")
 
        self.entries = {}
        for i, (key, label, hint) in enumerate(FIELDS):
            tk.Label(form_frame, text=label, font=("Segoe UI", 10), anchor="w").grid(
                row=i, column=0, sticky="w", pady=6
            )
            entry = tk.Entry(form_frame, font=("Segoe UI", 10), width=18)
            entry.grid(row=i, column=1, padx=10, pady=6)
            tk.Label(form_frame, text=hint, font=("Segoe UI", 9, "italic"), fg="gray").grid(
                row=i, column=2, sticky="w"
            )
            self.entries[key] = entry
 
        predict_btn = tk.Button(
            frame, text="Predict", font=("Segoe UI", 11, "bold"),
            bg="#2c7be5", fg="white", padx=20, pady=8,
            command=self.on_predict
        )
        predict_btn.pack(pady=20)
 
        # Result display area
        self.result_frame = tk.Frame(frame, bg="#f4f6f8")
        self.result_frame.pack(fill="x", padx=20)
 
        self.result_label = tk.Label(
            self.result_frame, text="", font=("Segoe UI", 13, "bold"), pady=8
        )
        self.result_label.pack()
 
        self.prob_label = tk.Label(
            self.result_frame, text="", font=("Segoe UI", 10), justify="center"
        )
        self.prob_label.pack()
 
        self.note_label = tk.Label(
            self.result_frame,
            text="Note: For academic purposes only. Not a medical diagnosis.",
            font=("Segoe UI", 8, "italic"), fg="gray"
        )
        self.note_label.pack(pady=(6, 0))
 
    # ------------------------------------------------
    def on_predict(self):
        values = []
        for key, label, hint in FIELDS:
            raw = self.entries[key].get().strip()
            try:
                values.append(float(raw))
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid number for:\n{label}")
                return
 
        user_df = pd.DataFrame([values], columns=self.feature_names)
        user_scaled = self.scaler.transform(user_df)
 
        prediction = self.model.predict(user_scaled)[0]
        probability = self.model.predict_proba(user_scaled)[0]
 
        if prediction == 1:
            self.result_label.config(text="HIGH likelihood of diabetes", fg="#c0392b")
        else:
            self.result_label.config(text="LOW likelihood of diabetes", fg="#1e8449")
 
        self.prob_label.config(
            text=(f"Probability of Diabetes: {probability[1] * 100:.2f}%    |    "
                  f"Probability of No Diabetes: {probability[0] * 100:.2f}%")
        )
 
 
if __name__ == "__main__":
    app = DiabetesApp()
    app.mainloop()
