# ❤️ Heart Failure Prediction

A machine learning web application that predicts heart failure risk from patient health data, using a Logistic Regression model served through a Flask backend.

## Overview

This project trains a classification model on the [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction) and deploys it as an interactive web app. Users enter patient details through a form, and the app returns a predicted risk percentage along with a visual risk gauge.

**Model performance:**

| Metric | Score |
|---|---|
| Accuracy | 88.59% |
| Precision | 0.8716 |
| Recall | 0.9314 |
| F1 Score | 0.9005 |
| F2 Score | 0.9188 |

## Project Structure

```
├── app.py                                   # Flask backend serving the model
├── Model_Training.py                        # Script to train and save the model
├── index.html                                # Frontend UI
├── heart.csv                                 # Dataset (download separately, see below)
├── heart_failure_model.joblib                # Trained model (generated)
├── heart_failure_scaler.joblib                # Fitted scaler (generated)
├── heart_failure_original_features.joblib    # Original feature names (generated)
├── heart_failure_encoded_features.joblib      # One-hot encoded feature names (generated)
└── requirements.txt                          # Python dependencies
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Download `heart.csv` from the [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction) and place it in the project root folder.

### 3. Train the model

```bash
python Model_Training.py
```

This loads `heart.csv`, preprocesses the data (one-hot encoding for categorical features, scaling for numeric features), trains a Logistic Regression model, evaluates it, and saves four files: the trained model, the fitted scaler, and two feature-name lists used by the app to correctly format incoming requests.

### 4. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser. The app should open automatically.

## How It Works

**Training (`Model_Training.py`)**
1. Loads and splits the dataset into features and target (`HeartDisease`).
2. One-hot encodes categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope).
3. Scales numeric features (Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak) using a `ColumnTransformer` + `StandardScaler`.
4. Splits data into train/test sets and trains a Logistic Regression classifier.
5. Evaluates the model and saves the model, scaler, and feature lists with `joblib`.

**Serving (`app.py`)**
1. Loads the trained model, scaler, and feature lists on startup.
2. Serves the web form at `/`.
3. On form submission (`/predict`), builds a DataFrame from the input, applies the same encoding and scaling used during training, and returns the predicted risk percentage as JSON.

**Frontend (`index.html`)**
A form with 11 inputs (mix of dropdowns for categorical fields and number inputs for numeric fields), with a popup showing the predicted risk as a percentage and visual gauge.

## Tech Stack

- **Python** — Flask, scikit-learn, pandas, numpy, joblib
- **Model** — Logistic Regression
- **Frontend** — HTML, CSS, JavaScript

## Disclaimer

This is a student/educational project trained on a public dataset for learning machine learning and web deployment concepts. It is **not a validated clinical or diagnostic tool** and should not be used to assess real medical risk. Consult a qualified healthcare professional for any actual health concerns.
