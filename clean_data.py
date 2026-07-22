"""
Data Cleaning & Feature Engineering

Steps:
1. Load raw synthetic data
2. Check for missing values / duplicates / bad rows
3. Add useful engineered features for the demand-prediction model
4. Encode the "zone" column numerically (models need numbers, not text)
5. Split into train/test sets
6. Save cleaned + split data back into data/
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/ride_demand.csv"

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows.")

# Data quality checks
print("\n--- Data Quality Check ---")
print("Missing values per column:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

negative_demand = (df["ride_demand"] < 0).sum()
negative_vehicles = (df["available_vehicles"] < 0).sum()
print(f"Negative ride_demand values: {negative_demand}")
print(f"Negative available_vehicles values: {negative_vehicles}")

# Drop duplicates if any were found (safety net for real-world reuse of this script)
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Dropped {duplicate_count} duplicate rows.")

# Feature engineering
# Rush hour flag (useful signal for the model)
df["is_rush_hour"] = df["hour"].isin([8, 9, 17, 18, 19]).astype(int)

# Night hours flag (relevant for weekend nightlife demand)
df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 21 or h <= 1) else 0).astype(int)

# Supply-demand gap: the core business problem, made explicit as a feature
df["vehicle_gap"] = df["ride_demand"] - df["available_vehicles"]

# Encode zone (text -> numeric)
# LightGBM can handle categorical codes; we map zones to integers explicitly
zone_mapping = {zone: idx for idx, zone in enumerate(sorted(df["zone"].unique()))}
df["zone_id"] = df["zone"].map(zone_mapping)

print("\nZone mapping used:")
for zone, idx in zone_mapping.items():
    print(f"  {zone} -> {idx}")

# Train/test split 
feature_cols = ["day_of_week", "is_weekend", "hour", "zone_id", "is_rush_hour", "is_night"]
target_col = "ride_demand"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

# Save outputs
df.to_csv("data/ride_demand_cleaned.csv", index=False)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nSaved:")
print("  data/ride_demand_cleaned.csv  (full cleaned dataset with new features)")
print("  data/X_train.csv / X_test.csv (model input features)")
print("  data/y_train.csv / y_test.csv (model target: ride_demand)")

print("\nPreview of cleaned data:")
print(df.head(5))
