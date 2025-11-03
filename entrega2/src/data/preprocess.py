# src/data/preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === 1. Load dataset ===
df = pd.read_csv("entrega1/data/processed/landmarks/merged_dataset.csv")
print(f"Loaded dataset: {df.shape}")

# === 2. Drop highly correlated or redundant columns ===
to_drop = [
    "left_shoulder_y", "right_shoulder_y", "right_shoulder_visibility",
    "left_hip_x", "right_hip_x", "right_hip_y", "right_hip_visibility",
    "left_knee_y", "right_knee_y", "left_ankle_x", "left_ankle_y",
    "right_ankle_x", "right_ankle_y", "left_wrist_x", "center_mass_y",
    "source_file", "frame", "timestamp"   # not useful for classification
]
df = df.drop(columns=[c for c in to_drop if c in df.columns])
print(f"After column filtering: {df.shape[1]} columns remain.")

# === 3. Encode the activity labels ===
label_encoder = LabelEncoder()
df["activity_encoded"] = label_encoder.fit_transform(df["activity"])
print("Label encoding:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# === 4. Split features / labels ===
X = df.drop(columns=["activity", "activity_encoded"])
y = df["activity_encoded"]

# === 5. Standardize numeric features ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# === 6. Merge scaled features with label ===
processed_df = X_scaled.copy()
processed_df["activity"] = y

# === 7. Save processed dataset ===
processed_df.to_csv("entrega2/data/processed/clean_landmarks.csv", index=False)
print("Preprocessed dataset saved to data/processed/clean_landmarks.csv")
