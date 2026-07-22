"""
Evaluation
Tests the trained LightGBM model on UNSEEN test data (X_test/y_test)
to measure real-world performance, not just training accuracy.
"""

import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load test data + trained model 
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

model = joblib.load("models/demand_model.pkl")
print("Model loaded from models/demand_model.pkl")
print(f"Evaluating on {len(X_test)} unseen rows.\n")

# Predict on test data 
predictions = model.predict(X_test)

# Calculate metrics 
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("--- Evaluation Results (on unseen test data) ---")
print(f"R^2 Score : {r2:.4f}   (1.0 = perfect, closer to 1 is better)")
print(f"MAE       : {mae:.2f} rides   (average prediction error)")
print(f"RMSE      : {rmse:.2f} rides  (penalizes large errors more)")

# Compare training vs test performance (overfitting check)
print("\n--- Overfitting Check ---")
print("If test R^2 is much lower than training R^2 (0.9066), the model")
print("may be overfitting (memorizing training data instead of learning patterns).")
print(f"Test R^2: {r2:.4f}  vs  Training R^2: 0.9066")
gap = 0.9066 - r2
print(f"Gap: {gap:.4f}", "(small gap = healthy model)" if gap < 0.1 else "(large gap = check for overfitting)")

# Show sample predictions vs actual
comparison = pd.DataFrame({
    "actual_demand": y_test.values[:10],
    "predicted_demand": predictions[:10].round(1),
})
comparison["error"] = (comparison["actual_demand"] - comparison["predicted_demand"]).round(1)
print("\n--- Sample Predictions (first 10 test rows) ---")
print(comparison.to_string(index=False))

# Save evaluation report
with open("models/evaluation_report.txt", "w") as f:
    f.write("Voss Mobility - Demand Prediction Model Evaluation\n")
    f.write("=" * 50 + "\n")
    f.write(f"Test set size: {len(X_test)} rows\n")
    f.write(f"R^2 Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.2f} rides\n")
    f.write(f"RMSE: {rmse:.2f} rides\n")
    f.write(f"Training R^2 (reference): 0.9066\n")
    f.write(f"Train/Test R^2 gap: {gap:.4f}\n")

print("\nEvaluation report saved to models/evaluation_report.txt")
