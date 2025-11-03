import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

def load_dataset(path="entrega2/data/processed/clean_landmarks.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    X = df.drop(columns=["activity"])
    y = df["activity"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# === Paths ===
SAVE_PATH = "entrega2/experiments/models/"
RESULTS_PATH = "entrega2/experiments/results/"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# === Load data ===
X_train, X_test, y_train, y_test = load_dataset()

# === Model definitions ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
}

# === Training + Evaluation ===
results = []
cv_results = []

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    # --- Cross-validation (5 folds) ---
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    # --- Fit on train/test split ---
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    # --- Collect results ---
    results.append({
        "model": name,
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
    })

    cv_results.append({
        "model": name,
        "cv_mean_accuracy": mean_cv,
        "cv_std": std_cv
    })

    # --- Save model ---
    joblib.dump(pipe, os.path.join(SAVE_PATH, f"{name}_model.pkl"))
    print(f"{name} | Test Accuracy: {report['accuracy']:.3f} | Cross-val: {mean_cv:.3f} ± {std_cv:.3f}")

# === Save results ===
pd.DataFrame(results).to_csv(os.path.join(RESULTS_PATH, "model_performance.csv"), index=False)
pd.DataFrame(cv_results).to_csv(os.path.join(RESULTS_PATH, "cross_validation_results.csv"), index=False)

print("\nTraining complete. Models and results saved to:")
print(f"→ {RESULTS_PATH}")

