import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

MODEL_PATH = "entrega2/experiments/models/RandomForest_model.pkl"
RESULTS_PATH = "entrega2/experiments/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)


def load_dataset(path="entrega2/data/processed/clean_landmarks.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    X = df.drop(columns=["activity"])
    y = df["activity"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# === Load dataset and model ===
X_train, X_test, y_train, y_test = load_dataset()
model = joblib.load(MODEL_PATH)

# === Confusion Matrix ===
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Reds", xticks_rotation=45)
plt.title("Confusion Matrix - RandomForest")
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
plt.close()
print("Confusion matrix saved.")

# === ROC Curve (One-vs-Rest) ===
# Ensure model supports probability output
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)
else:
    print("Model does not support predict_proba(), skipping ROC curve.")
    exit()

# Binarize labels
classes = model.classes_
y_bin = label_binarize(y_test, classes=classes)

# Compute ROC curve and AUC for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])

# Plot ROC curves
plt.figure(figsize=(7, 6))
for cls in classes:
    plt.plot(fpr[cls], tpr[cls], label=f"{cls} (AUC = {roc_auc[cls]:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - RandomForest (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "roc_curve.png"))
plt.close()

# Save AUC values to CSV
auc_df = pd.DataFrame(list(roc_auc.items()), columns=["Activity", "AUC"])
auc_df.to_csv(os.path.join(RESULTS_PATH, "roc_auc_scores.csv"), index=False)

print("ROC curve and AUC values saved.")

