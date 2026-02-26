# AI House Price ML Project

Beginner-friendly, deployment-ready project for **GitHub + Streamlit Cloud**.

## Project Structure

```text
House-Price-ML/
|
|- data/
|  |- housing.csv
|- app.py
|- train.py
|- requirements.txt
|- README.md
```

After training, `model.pkl` will be created in the project root.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Train Model

```bash
python train.py --data-path data/housing.csv --model-path model.pkl
```

With external test data (example path from Downloads):

```bash
python train.py --data-path data/housing.csv --test-data-path "C:\Users\chara\Downloads\archive (1)\Housing.csv" --model-path model.pkl
```

What this does:
- Loads your CSV (`price` is target)
- Handles missing values
- One-hot encodes categorical columns
- Trains a `RandomForestRegressor`
- Optionally evaluates on external test CSV
- Saves `model.pkl` with metadata for the Streamlit app

## 3. Run App Locally

```bash
streamlit run app.py
```

## 4. Push to GitHub

From inside `House-Price-ML`:

```bash
git init
git add .
git commit -m "Initial ML app"
git branch -M main
git remote add origin <YOUR_REPO_LINK>
git push -u origin main
```

## 5. Deploy on Streamlit Cloud

1. Open `https://streamlit.io/cloud`
2. Sign in with GitHub
3. Click **New app**
4. Select your repository
5. Branch: `main`
6. Main file: `app.py`
7. Click **Deploy**

## App Features

- Price prediction
- What-if simulator for numeric features
- Scenario-based trend projection
- Model feature importance chart

## Common Fixes

- `ModuleNotFoundError`: add missing package to `requirements.txt`
- `model.pkl not found`: run `python train.py ...` first
- Prediction errors after dataset changes: retrain model and replace `model.pkl`
