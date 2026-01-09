# MLOps Lab 2 – Automated Model Training with CI/CD

## 📌 Overview
This project demonstrates a complete MLOps workflow using Python, Git, and GitHub Actions.
The objective is to build a reproducible machine learning pipeline that performs multiple
experiments, evaluates model performance, and runs automatically using CI/CD.

---

## 📂 Dataset
- **Dataset Name:** Wine Quality (Red)
- **Source:** UCI Machine Learning Repository
- **File:** `winequality-red.csv`
- **Format:** CSV (semicolon-separated)
- **Target Variable:** `quality`
- **Features:** Physicochemical properties of red wine

---

## ⚙️ Project Structure
      mlops/
      │── dataset/
      │ └── winequality-red.csv
      │── outputs/
      │ └── results/
      │ └── experiments.json
      │── .github/
      │ └── workflows/
      │ └── train.yml
      │── train.py
      │── requirements.txt
      │── README.md

---

## Setup Instructions (Local Execution)

### 
```bash
python -m venv env
source env/bin/activate      # Linux / Mac
env\Scripts\activate         # Windows
pip install -r requirements.txt
python train.py
