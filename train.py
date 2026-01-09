from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import json
import os

# Load dataset (IMPORTANT FIX)
data_path = "dataset/winequality-red.csv"

# Wine dataset is semicolon-separated
df = pd.read_csv(data_path, sep=";")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

assert X.shape[1] > 0, "Dataset loaded incorrectly — no features found"

# Experiment configurations
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
        "features": [0]  # SAFE subset
    },
    {
        "name": "No_Scaling_70_30",
        "model": LinearRegression(),
        "scale": False,
        "test_size": 0.3,
        "features": None
    }
]

experiments = []

# Run experiments
for cfg in configs:

    if cfg["features"] is None:
        X_exp = X
    else:
        X_exp = X[:, cfg["features"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y, test_size=cfg["test_size"], random_state=42
    )

    steps = []
    if cfg["scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", cfg["model"]))

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    experiments.append({
        "experiment": cfg["name"],
        "mse": mse,
        "r2": r2
    })

# Save results
os.makedirs("outputs/results", exist_ok=True)

with open("outputs/results/experiments.json", "w") as f:
    json.dump(experiments, f, indent=4)

# GitHub Actions summary
summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

if summary_path:
    with open(summary_path, "a") as f:
        f.write("## Experiment Results\n")
        for exp in experiments:
            f.write(
                f"- **{exp['experiment']}** → "
                f"MSE: {exp['mse']:.4f}, R²: {exp['r2']:.4f}\n"
            )
