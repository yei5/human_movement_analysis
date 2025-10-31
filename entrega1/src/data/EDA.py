import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/landmarks/merged_dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="activity", palette="Reds")
plt.title("Distribution of Activity Classes")
plt.xlabel("Activity")
plt.ylabel("Frame Count")
plt.tight_layout()
plt.savefig("docs/EDA/activity_distribution.png")
plt.close()

activity_rate = df["activity"].value_counts(normalize=True) * 100
print("\nPercentage distribution of activities:")
print(activity_rate.round(2))

missing = df.isna().sum()
print("\nMissing values per column:")
print(missing[missing > 0])

key_features = [
    "left_knee_angle", "right_knee_angle",
    "trunk_lateral_inclination", "person_height"
]

for feature in key_features:
    if feature in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x="activity", y=feature, palette="coolwarm")
        plt.title(f"Distribution of {feature} by Activity")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"docs/EDA/{feature}_by_activity.png")
        plt.close()

        mean_vals = df.groupby("activity")[feature].mean().round(2)
        print(f"\nAverage {feature} by Activity:")
        print(mean_vals)

numeric_cols = df.select_dtypes(include=["number"]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="Reds", center=0, linewidths=0.3)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig("docs/EDA/correlation_matrix.png")
plt.close()

print("\nCorrelation matrix computed and saved.")


upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]

print("\nHighly correlated variables (r > 0.9):")
print(high_corr if high_corr else "None found")

subset_features = [
    "left_knee_angle",
    "right_knee_angle",
    "trunk_lateral_inclination",
    "person_height",
    "activity"
]

if all(f in df.columns for f in subset_features):
    sns.pairplot(df[subset_features].sample(min(500, len(df)), random_state=42),
                 hue="activity", diag_kind="kde", palette="husl")
    plt.suptitle("Pairwise Relationships Between Key Features", y=1.02)
    plt.tight_layout()
    plt.savefig("docs/EDA/pairwise_features.png")
    plt.close()

print("* Dataset size:", len(df))
print("* Activities:", list(df['activity'].unique()))
