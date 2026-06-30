import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Imported additional metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from sklearn.compose import ColumnTransformer
import joblib
import os

DATASET_PATH = 'heart.csv'

# --- Data Loading ---
print("Step 1: Loading Data...")
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset file not found at {DATASET_PATH}")
    print("Please download 'heart.csv' from Kaggle and place it in this folder.")
    exit()
    
data = pd.read_csv(DATASET_PATH)

# --- Feature and Target Separation ---
print("Step 2: Separating Features (X) and Target (y)...")
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease'] # Target: 1 for heart failure, 0 for normal

# --- Preprocessing (One-Hot Encoding & Scaling) ---
print("Step 3: Preprocessing Data...")

# Identify which columns are numeric and which are categorical
# 'FastingBS' is binary (0/1) but we'll treat it as numeric here.
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create the full list of original features in order
# This will be used by the app.py to create the DataFrame
original_features_list = numeric_features + categorical_features

# Perform One-Hot Encoding on categorical features
# This converts 'M'/'F' to 'Sex_M': 1, 'Sex_F': 0, etc.
X_processed = pd.get_dummies(X, columns=categorical_features)

# Get the list of ALL features *after* one-hot encoding
# This order is critical for the model
encoded_feature_names = X_processed.columns.tolist()

# --- Data Splitting ---
print("Step 4: Splitting Data into Train and Test sets...")
# We split the *processed* data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# --- Data Scaling ---
print("Step 5: Scaling Numeric Data...")
# We only want to scale the numeric features, not the new 0/1 dummy features.
# We create a scaler that will *only* transform the numeric columns
# and leave the one-hot encoded columns alone.
# `remainder='passthrough'` means any columns not listed will be untouched.
scaler = ColumnTransformer(
    [("scaler", StandardScaler(), numeric_features)],
    remainder="passthrough"
)

# We fit the scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
print("Step 6: Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
print("Step 7: Evaluating Model...")
y_pred = model.predict(X_test_scaled)

# Calculate all requested metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2) # F2 weighs recall higher than precision
conf_matrix = confusion_matrix(y_test, y_pred)

print("-" * 30)
print("Model Performance Metrics:")
print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"F2 Score:       {f2:.4f}")
print("-" * 30)
print("Confusion Matrix (Text):")
print(conf_matrix)
print("-" * 30)

# --- Generating Confusion Matrix Graph ---
print("Step 7b: Generating Confusion Matrix Heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Heart Failure'],
            yticklabels=['Normal', 'Heart Failure'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)

# Show the plot interactively
# NOTE: The script will pause here until you close the popup window.
plt.show()

# --- Model Saving ---
print("Step 8: Saving Model, Scaler, and Feature Lists...")
# Save the trained model
joblib.dump(model, 'heart_failure_model.joblib')

# CRITICAL: Save the ColumnTransformer scaler
joblib.dump(scaler, 'heart_failure_scaler.joblib')

# CRITICAL: Save the list of original features (for app.py to build the DataFrame)
# and the list of encoded features (for app.py to re-order the DataFrame)
joblib.dump(original_features_list, 'heart_failure_original_features.joblib')
joblib.dump(encoded_feature_names, 'heart_failure_encoded_features.joblib')

print("model_training.py script finished successfully.")