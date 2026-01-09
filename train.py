import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Paths
DATA_PATH = "dataset/winequality-red.csv"
MODEL_DIR = "outputs/model"
RESULTS_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

# Feature selection (correlation-based)
corr = df.corr()["quality"].abs()
selected_features = corr[corr > 0.2].index.drop("quality")

X = df[selected_features]
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model pipeline (preprocessing + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (important for GitHub Actions)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
model_path = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(pipeline, model_path)

# Save metrics
metrics = {
    "mse": mse,
    "r2_score": r2
}

metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
