# 📊 Loan Defaulter Prediction

## 1. 📖 Introduction

This repository contains a complete end-to-end project for predicting loan default risk using machine learning. The dataset includes both current application data and historical application records which are used for feature engineering, exploratory data analysis, model training, and inference.

**Files included in the dataset:**

* `application_data.csv` — current client-level features and the `TARGET` label (0 = repaid, 1 = default).
* `previous_application.csv` — historical loan applications for the same clients (use to create aggregated features per `SK_ID_CURR`).
* `columns_description.csv` — a dictionary describing column meanings and units.

---

## 2. 💻 Contributors

| Name            | GitHub                                                                     |
| --------------- | -------------------------------------------------------------------------- |
| Ahmed Ashraf    | [https://github.com/ahmedfashraf-1](https://github.com/ahmedfashraf-1)     |
| Malak Ahmed     | [https://github.com/Malak-A7med](https://github.com/Malak-A7med)           |
| Tasneem Hussein | [https://github.com/tasneemhussein12](https://github.com/tasneemhussein12) |
| Mohamed Sheta   | [https://github.com/Mohamed-Sheta](https://github.com/Mohamed-Sheta)       |
| Ossama Ayman    | [https://github.com/Ossama-Ayman](https://github.com/Ossama-Ayman)         |

---

## 3. 🎯 Project Objective

To build robust ML models that estimate the probability of loan default by combining current application features with aggregated historical behaviour from previous applications. The resulting models and UI are intended to help data scientists and risk analysts inspect predictions and understand the most important risk drivers.

---

## 4. 📂 Dataset Overview

Use the three CSVs together during EDA and feature engineering. The `columns_description.csv` file is the authority for field meanings.

> Column description (stored in repo): `data/columns_description.csv`

---

## 5. 🏗️ Project Structure

```
Loan-Defaulter-Prediction/
│
├── data/
│   ├── application_data.csv
│   ├── previous_application.csv
│   └── columns_description.csv
│
├── notebooks/
│   └── eda_and_modeling.ipynb
│
├── models/
│   └── final_model.pkl
│
├── fast_api_app/
│   └── main.py
│
├── streamlit/
│   ├── app.py
│   └── assets/
│
├── requirements.txt
└── README.md
```

**What each folder contains:**

* `data/` — raw CSVs (do **not** push sensitive data to public repos).
* `notebooks/` — notebooks for EDA, preprocessing, feature engineering and training experiments.
* `models/` — serialized model(s) and preprocessor objects used by the API/UI.
* `fast_api_app/` — FastAPI backend exposing prediction endpoints.
* `streamlit/` — Streamlit frontend for interactive model exploration.
* `requirements.txt` — pinned Python dependencies.

---

## 6. ⚙️ Installation

Recommended: create and activate a virtual environment first.

```bash
# create & activate venv (example for Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# or on macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

> Tip: `requirements.txt` should contain packages such as `pandas`, `numpy`, `scikit-learn`, `joblib`/`pickle` (for model I/O), `fastapi`, `uvicorn`, and `streamlit`. Adjust pins to match your environment.

---

## 7. 🚀 Running the Applications

You can run the backend API and the Streamlit frontend locally. They should be launched in separate terminals (or as background processes).

### A) Run the Streamlit App

```bash
cd streamlit
streamlit run app.py
```

Streamlit will automatically open a browser window (or show a local URL like `http://localhost:8501`).

If the Streamlit UI expects the locally-running FastAPI backend, make sure the API is running before interacting with the UI.

### B) Run the FastAPI Backend

```bash
cd fast_api_app
uvicorn fast_api_app.main:app --reload
```

This binds by default to `127.0.0.1:8000`. Open the interactive API docs at `http://127.0.0.1:8000/docs`.

**Notes:**

* `--reload` enables auto-reload for development. Remove it in production.
* If your `main.py` exposes the app with a different variable name or module path, replace `fast_api_app.main:app` accordingly.

### C) Running both in parallel

Open two terminals/tabs and run the Streamlit command in one and the Uvicorn command in the other. Alternatively use a process manager (tmux, GNU screen) or Docker (see optional section below).

---

## 8. 🧭 Usage & Workflow

1. Put the CSVs in the `data/` folder.
2. Open `notebooks/eda_and_modeling.ipynb` and run the cells to reproduce preprocessing, feature engineering, and model training steps.
3. After training, save the model and any preprocessing pipeline into `models/` (e.g., `final_model.pkl`).
4. Update `fast_api_app/main.py` to load the model from `models/` and expose an inference endpoint (e.g., `/predict`).
5. Start the FastAPI server and the Streamlit app to serve predictions.

---

## 9. 📚 References & Resources

* **Data Source:** Kaggle — Loan Defaulter Dataset
* **UI/UX Design:** [https://shaper-dark-muse.lovable.app/](https://shaper-dark-muse.lovable.app/)

---
