import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# CONFIG
# =====================
DATA_PATH = "data/processed/landmarks/merged_dataset.csv"  # unificado con columna 'activity'
TARGET = "activity"

# =====================
# LOAD DATA
# =====================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Drop non-numeric columns and missing data
if TARGET not in df.columns:
    raise ValueError(f"Missing '{TARGET}' column. Did you merge annotations?")

df = df.dropna().reset_index(drop=True)

# Select numeric features only
X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["frame", "timestamp"], errors="ignore")
y = df[TARGET]

print(f"Features: {X.shape[1]} | Samples: {X.shape[0]} | Classes: {y.nunique()}")

# =====================
# SPLIT DATA
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# =====================
# SCALE FEATURES
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# TRAIN MODEL
# =====================
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_scaled, y_train)

# =====================
# EVALUATE MODEL
# =====================
y_pred = clf.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# =====================
# SAVE MODEL (optional)
# =====================
import joblib
os.makedirs("data/processed/models", exist_ok=True)
joblib.dump(clf, "data/processed/models/random_forest_test.pkl")
print("\nModel saved to data/processed/models/random_forest_test.pkl")
