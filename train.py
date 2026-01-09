from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

import pandas as pd
import numpy as np

# Load dataset
data_path = "dataset/winequality-red.csv"  # change if needed
df = pd.read_csv(data_path)

# Assume last column is target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


experiments = []

configs = [
    {
        "name": "Baseline",
        "model": LinearRegression(),
        "scale": True,
        "test_size": 0.2,
        "features": None
    },
    {
        "name": "Ridge_alpha_1",
        "model": Ridge(alpha=1.0),
        "scale": True,
        "test_size": 0.2,
        "features": None
    },
    {
        "name": "Feature_Subset",
        "model": LinearRegression(),
        "scale": True,
        "test_size": 0.2,
        "features": [0, 1]  # example subset
    },
    {
        "name": "No_Scaling_70_30",
        "model": LinearRegression(),
        "scale": False,
        "test_size": 0.3,
        "features": None
    }
]

for cfg in configs:
    if cfg["features"] is None or len(cfg["features"]) == 0:
        X_exp = X
    elif max(cfg["features"]) >= X.shape[1]:
        # Skip invalid feature subset
        print(f"Skipping {cfg['name']} due to invalid feature indices")
        continue
    else:
        X_exp = X[:, cfg["features"]]


    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y, test_size=cfg["test_size"], random_state=42
    )

    if cfg["scale"]:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", cfg["model"])
        ])
    else:
        pipeline = Pipeline([
            ("model", cfg["model"])
        ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    experiments.append({
        "experiment": cfg["name"],
        "mse": mse,
        "r2": r2
    })

# Save all experiment results
os.makedirs("outputs/results", exist_ok=True)
with open("outputs/results/experiments.json", "w") as f:
    json.dump(experiments, f, indent=4)



summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

if summary_path:
    with open(summary_path, "a") as f:
        f.write("## Model Evaluation Metrics\n")
        f.write(f"- **Mean Squared Error (MSE):** {mse:.4f}\n")
        f.write(f"- **R² Score:** {r2:.4f}\n")
if summary_path:
    with open(summary_path, "a") as f:
        f.write("\n## Experiment Results\n")
        for exp in experiments:
            f.write(
                f"- **{exp['experiment']}** → "
                f"MSE: {exp['mse']:.4f}, R²: {exp['r2']:.4f}\n"
            )
