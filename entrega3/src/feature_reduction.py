# entrega3/src/generate_reduced_and_compare.py
"""
Full pipeline for feature reduction + training reduced model + comprehensive comparison
(original pipeline -> feature importances -> select top features -> train reduced model ->
evaluate both models with metrics, confusion matrices, ROC/PR, permutation importance,
timings, model sizes). Saves artifacts under entrega3/experiments/.

Usage: run from project root:
    python entrega3/src/generate_reduced_and_compare.py
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.inspection import permutation_importance

# -----------------------
# CONFIG
# -----------------------
ROOT = Path("entrega3/experiments")
DATA_CSV = Path("entrega2/data/processed/clean_landmarks.csv")
ORIG_PIPE_PKL = Path("entrega2/experiments/models/RandomForest_model.pkl")  # original pipeline (scaler + clf)
OUT_DIR = ROOT / "comparison_full"
PLOTS_DIR = OUT_DIR / "plots"
MODELS_DIR = ROOT / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# HELPERS
# -----------------------
def savefig(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def safe_get_orig_feature_names(pipe, fallback_cols):
    # Try common attributes to recover original feature order used in training
    try:
        # If pipeline contains scaler with get_feature_names_out
        scaler = pipe.named_steps.get("scaler", None)
        if scaler is not None and hasattr(scaler, "get_feature_names_out"):
            return list(scaler.get_feature_names_out())
    except Exception:
        pass
    # Try pipeline.feature_names_in_
    try:
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
    except Exception:
        pass
    # fallback: columns from dataset
    return list(fallback_cols)

# -----------------------
# LOAD DATA + ORIGINAL PIPELINE
# -----------------------
print("Loading data and original pipeline...")
df = pd.read_csv(DATA_CSV)
X = df.drop(columns=["activity"])
y = df["activity"].copy()

orig_pipe = joblib.load(ORIG_PIPE_PKL)  # expected to be a sklearn Pipeline (scaler + clf)
orig_clf = orig_pipe.named_steps["clf"]

# get original feature names (order) used by orig_pipe
orig_feature_names = safe_get_orig_feature_names(orig_pipe, X.columns.tolist())
print(f"Original pipeline feature count: {len(orig_feature_names)}")

# -----------------------
# TRAIN/TEST SPLIT (consistent for comparisons)
# -----------------------
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Ensure ordering for original model input
X_test_orig = X_test_full[orig_feature_names].copy()

# -----------------------
# FEATURE IMPORTANCE FROM ORIGINAL MODEL
# -----------------------
print("Computing feature importances from the original model...")
# If orig_clf is a RandomForest and pipeline used scaling, feature_importances_ aligns with scaler's feature order
importances = orig_clf.feature_importances_
# Map to orig_feature_names (lengths should match)
if len(importances) != len(orig_feature_names):
    # fallback: try using X columns order
    print("Warning: length mismatch between importances and recovered feature names. Using X.columns order.")
    orig_feature_names = X.columns.tolist()
    importances = orig_clf.feature_importances_

feat_imp = pd.DataFrame({"feature": orig_feature_names, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)
feat_imp["cum_importance"] = feat_imp["importance"].cumsum()
feat_imp.to_csv(OUT_DIR / "feature_importances_original_model.csv", index=False)

# Plot top 20 importances
fig = plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(20), x="importance", y="feature", palette="viridis")
plt.title("Top 20 Feature Importances (original model)")
savefig(fig, "top20_feature_importances_original.png")

# -----------------------
# SELECT TOP FEATURES (90% cumulative importance)
# -----------------------
threshold = 0.90
selected = feat_imp[feat_imp["cum_importance"] <= threshold]["feature"].tolist()
# If selection yields 0 (rare), pick top 20
if len(selected) == 0:
    selected = feat_imp.head(20)["feature"].tolist()

print(f"Selected {len(selected)} features (cumulative importance <= {threshold})")
pd.DataFrame({"selected_features": selected}).to_csv(OUT_DIR / "selected_features.csv", index=False)

# -----------------------
# BUILD REDUCED DATASET + SCALER
# -----------------------
X_reduced = X[selected].copy()
scaler_reduced = StandardScaler()
scaler_reduced.fit(X_reduced)  # fit on full data (alternatively fit on train only)
joblib.dump(scaler_reduced, MODELS_DIR / "scaler_reduced.pkl")
joblib.dump(selected, MODELS_DIR / "feature_names_reduced.pkl")

# Train/test split for reduced model (use same random_state to be comparable)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reduced, y, test_size=0.20, stratify=y, random_state=42
)

# -----------------------
# TRAIN REDUCED MODEL
# -----------------------
print("Training reduced RandomForest model...")
t0 = time.time()
reduced_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
reduced_clf.fit(Xr_train, yr_train)
train_time = time.time() - t0

# Save reduced model
joblib.dump(reduced_clf, MODELS_DIR / "randomforest_reduced.pkl")

# -----------------------
# EVALUATION: PREDICTIONS + TIMINGS
# -----------------------
print("Evaluating models on the shared test split...")

# Original pipeline predictions (pipeline will scale internally)
t0 = time.time()
orig_probs = orig_pipe.predict_proba(X_test_orig)
t1 = time.time()
orig_preds = orig_pipe.predict(X_test_orig)
t2 = time.time()

# Reduced model: prepare test set in selected order and scale
X_test_reduced = X_test_full[selected].copy()
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

t3 = time.time()
reduced_probs = reduced_clf.predict_proba(X_test_reduced_scaled)
t4 = time.time()
reduced_preds = reduced_clf.predict(X_test_reduced_scaled)
t5 = time.time()

timings = {
    "orig_predict_proba_s": t1 - t0,
    "orig_predict_s": t2 - t1,
    "orig_total_inference_s": t2 - t0,
    "reduced_predict_proba_s": t4 - t3,
    "reduced_predict_s": t5 - t4,
    "reduced_total_inference_s": t5 - t3,
    "orig_per_sample_ms": (t2 - t0) / len(X_test_orig) * 1000,
    "reduced_per_sample_ms": (t5 - t3) / len(X_test_reduced) * 1000
}

# -----------------------
# BASIC METRICS & REPORTS
# -----------------------
def compute_basic_metrics(y_true, y_pred):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred)
    }
    return out

orig_metrics = compute_basic_metrics(y_test_full, orig_preds)
reduced_metrics = compute_basic_metrics(y_test_full, reduced_preds)

pd.DataFrame([{"model":"original", **orig_metrics}, {"model":"reduced", **reduced_metrics}]) \
    .to_csv(OUT_DIR / "basic_metrics_comparison.csv", index=False)

# Save classification reports (per class)
pd.DataFrame(classification_report(y_test_full, orig_preds, output_dict=True)).T.to_csv(OUT_DIR / "orig_classification_report.csv")
pd.DataFrame(classification_report(y_test_full, reduced_preds, output_dict=True)).T.to_csv(OUT_DIR / "reduced_classification_report.csv")

# -----------------------
# CONFUSION MATRICES (raw + normalized)
# -----------------------
labels = sorted(y.unique())
cm_orig = confusion_matrix(y_test_full, orig_preds, labels=labels)
cm_red = confusion_matrix(y_test_full, reduced_preds, labels=labels)

pd.DataFrame(cm_orig, index=labels, columns=labels).to_csv(OUT_DIR / "confusion_matrix_original.csv")
pd.DataFrame(cm_red, index=labels, columns=labels).to_csv(OUT_DIR / "confusion_matrix_reduced.csv")

# Plot heatmaps
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_orig, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix - Original")
savefig(fig, "confusion_matrix_original.png")

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_red, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix - Reduced")
savefig(fig, "confusion_matrix_reduced.png")

# Normalized rows
cm_orig_norm = (cm_orig.astype(float) / cm_orig.sum(axis=1)[:, np.newaxis])
cm_red_norm = (cm_red.astype(float) / cm_red.sum(axis=1)[:, np.newaxis])
pd.DataFrame(cm_orig_norm, index=labels, columns=labels).to_csv(OUT_DIR / "confusion_matrix_original_normalized.csv")
pd.DataFrame(cm_red_norm, index=labels, columns=labels).to_csv(OUT_DIR / "confusion_matrix_reduced_normalized.csv")

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_orig_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title("Normalized Confusion Matrix - Original")
savefig(fig, "confusion_matrix_original_normalized.png")

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_red_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title("Normalized Confusion Matrix - Reduced")
savefig(fig, "confusion_matrix_reduced_normalized.png")

# -----------------------
# ROC & PR CURVES (multiclass)
# -----------------------
print("Computing ROC and Precision-Recall curves (multiclass)...")
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_full)
classes = lb.classes_

# Align probability columns to lb.classes_
def align_probs(probs, model_classes, target_classes):
    probs_df = pd.DataFrame(probs, columns=list(model_classes))
    probs_df = probs_df.reindex(columns=list(target_classes), fill_value=0)
    return probs_df.values

# original pipeline classes order
orig_model_classes = orig_pipe.named_steps["clf"].classes_
reduced_model_classes = reduced_clf.classes_

orig_probs_aligned = align_probs(orig_probs, orig_model_classes, classes)
red_probs_aligned = align_probs(reduced_probs, reduced_model_classes, classes)

# Per-class ROC & PR
roc_summary = []
pr_summary = []
for i, cls in enumerate(classes):
    fpr_o, tpr_o, _ = roc_curve(y_test_bin[:, i], orig_probs_aligned[:, i])
    auc_o = auc(fpr_o, tpr_o)
    prec_o, rec_o, _ = precision_recall_curve(y_test_bin[:, i], orig_probs_aligned[:, i])
    ap_o = average_precision_score(y_test_bin[:, i], orig_probs_aligned[:, i])

    fpr_r, tpr_r, _ = roc_curve(y_test_bin[:, i], red_probs_aligned[:, i])
    auc_r = auc(fpr_r, tpr_r)
    prec_r, rec_r, _ = precision_recall_curve(y_test_bin[:, i], red_probs_aligned[:, i])
    ap_r = average_precision_score(y_test_bin[:, i], red_probs_aligned[:, i])

    roc_summary.append({"class": cls, "orig_auc": auc_o, "reduced_auc": auc_r})
    pr_summary.append({"class": cls, "orig_ap": ap_o, "reduced_ap": ap_r})

    # plot per-class ROC (append to multi-line figure later)
    fig = plt.figure(figsize=(6,4))
    plt.plot(fpr_o, tpr_o, label=f"orig AUC={auc_o:.3f}")
    plt.plot(fpr_r, tpr_r, label=f"reduced AUC={auc_r:.3f}")
    plt.plot([0,1],[0,1],"k--", alpha=0.3)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - class {cls}")
    plt.legend(loc="lower right")
    savefig(fig, f"roc_class_{cls}.png")

    # per-class PR
    fig = plt.figure(figsize=(6,4))
    plt.plot(rec_o, prec_o, label=f"orig AP={ap_o:.3f}")
    plt.plot(rec_r, prec_r, label=f"reduced AP={ap_r:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - class {cls}")
    plt.legend(loc="lower left")
    savefig(fig, f"pr_class_{cls}.png")

pd.DataFrame(roc_summary).to_csv(OUT_DIR / "roc_summary_per_class.csv", index=False)
pd.DataFrame(pr_summary).to_csv(OUT_DIR / "pr_summary_per_class.csv", index=False)

# Macro/micro AUC summary
def multiclass_auc_average(probs_aligned, y_bin):
    aucs = []
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_aligned[:, i])
        aucs.append(auc(fpr, tpr))
    return np.mean(aucs), np.average(aucs, weights=y_bin.sum(axis=0))

orig_macro_auc, orig_weighted_auc = multiclass_auc_average(orig_probs_aligned, y_test_bin)
red_macro_auc, red_weighted_auc = multiclass_auc_average(red_probs_aligned, y_test_bin)

pd.DataFrame([{
    "model": "original", "macro_auc": orig_macro_auc, "weighted_auc": orig_weighted_auc
}, {
    "model": "reduced", "macro_auc": red_macro_auc, "weighted_auc": red_weighted_auc
}]).to_csv(OUT_DIR / "auc_macro_weighted.csv", index=False)

# -----------------------
# PERMUTATION IMPORTANCE (reduced model)
# -----------------------
print("Computing permutation importance for the reduced model...")
# Need X_reduced scaled for permutation_importance: use full X_reduced (not only train)
X_reduced_full = X_reduced.copy()
X_reduced_full_scaled = scaler_reduced.transform(X_reduced_full)

perm = permutation_importance(reduced_clf, X_reduced_full_scaled, y, n_repeats=10, random_state=42, n_jobs=2)
perm_df = pd.DataFrame({
    "feature": X_reduced_full.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(OUT_DIR / "permutation_importance_reduced.csv", index=False)

fig = plt.figure(figsize=(10,8))
sns.barplot(data=perm_df.head(20), x="importance_mean", y="feature", palette="viridis")
plt.title("Top 20 Permutation Importance (reduced model)")
savefig(fig, "permutation_importance_reduced_top20.png")

# -----------------------
# MODEL SIZES & TIME SUMMARY
# -----------------------
orig_size_kb = ORIG_PIPE_PKL.stat().st_size / 1024 if ORIG_PIPE_PKL.exists() else None
reduced_size_kb = (MODELS_DIR / "randomforest_reduced.pkl").stat().st_size / 1024

time_summary = timings.copy()
time_summary["reduced_train_time_s"] = train_time
time_summary["orig_model_size_kb"] = orig_size_kb
time_summary["reduced_model_size_kb"] = reduced_size_kb

pd.DataFrame([time_summary]).to_csv(OUT_DIR / "timings_and_sizes.csv", index=False)

# -----------------------
# SAVE SUMMARY JSON & CSVS
# -----------------------
full_summary = {
    "basic_metrics": {
        "original": orig_metrics,
        "reduced": reduced_metrics
    },
    "timings": time_summary,
    "auc_summary": {
        "orig_macro_auc": orig_macro_auc,
        "orig_weighted_auc": orig_weighted_auc,
        "reduced_macro_auc": red_macro_auc,
        "reduced_weighted_auc": red_weighted_auc
    },
    "selected_features": selected
}

json.dump(full_summary, open(OUT_DIR / "full_summary.json", "w"), indent=2)

print("All artifacts saved to:", OUT_DIR)
print("Finished.")
