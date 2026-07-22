"""
Model Training:
Trains a LightGBM regression model to predict ride_demand per
zone/hour, using the train/test split. Output: models/demand_model.pkl
"""

import pandas as pd
import lightgbm as lgb
import joblib

#  Load train/test data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows.")
print(f"Features used: {list(X_train.columns)}")

# Define the model 
# LightGBM Regressor: predicts a continuous number (ride demand count)
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1, 
)

# Train 
print("\nTraining LightGBM model...")
model.fit(X_train, y_train)
print("Training complete.")

# Quick sanity check on training data 
train_predictions = model.predict(X_train)
train_score = model.score(X_train, y_train)  # R^2 score
print(f"\nTraining R^2 score: {train_score:.4f}")

# Feature importance (which features matter most) 
print("\nFeature importance:")
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.to_string(index=False))

# Save the trained model 
joblib.dump(model, "models/demand_model.pkl")
print("\nModel saved to models/demand_model.pkl")

print("\nNext step: run evaluate_model.py to test on unseen data.")
