from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import webbrowser  # Added for opening browser
from threading import Timer # Added for delaying the browser open

# Initialize the Flask application
app = Flask(__name__, template_folder='.')

# --- Load the Trained Model and Scaler ---
MODEL_PATH = 'heart_failure_model.joblib'
SCALER_PATH = 'heart_failure_scaler.joblib'
ORIG_FEATURES_PATH = 'heart_failure_original_features.joblib'
ENCODED_FEATURES_PATH = 'heart_failure_encoded_features.joblib'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    original_features_list = joblib.load(ORIG_FEATURES_PATH)
    encoded_feature_names = joblib.load(ENCODED_FEATURES_PATH)
    print("Model, scaler, and feature lists loaded successfully.")
except Exception as e:
    print(f"An error occurred loading files: {e}")
    print("Please run model_training.py first.")
    model = None
    scaler = None
    original_features_list = None
    encoded_feature_names = None

# --- Define the Routes ---

@app.route('/')
def home():
    """ Renders the main homepage (index.html). """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives data from the HTML form, makes a prediction, and returns it as JSON.
    """
    if not all([model, scaler, original_features_list, encoded_feature_names]):
        return jsonify({'error': 'Model or supporting files not loaded. Please check server logs.'}), 500

    try:
        # 1. Get the data from the form
        form_data = request.form.to_dict()
        
        # 2. Create a single-row DataFrame from the form data
        input_data = {}
        for feature in original_features_list:
            input_data[feature] = [form_data.get(feature)]
            
        input_df = pd.DataFrame.from_dict(input_data)
        
        # 3. Convert numeric columns to float/int
        input_df['Age'] = input_df['Age'].astype(int)
        input_df['RestingBP'] = input_df['RestingBP'].astype(int)
        input_df['Cholesterol'] = input_df['Cholesterol'].astype(int)
        input_df['FastingBS'] = input_df['FastingBS'].astype(int)
        input_df['MaxHR'] = input_df['MaxHR'].astype(int)
        input_df['Oldpeak'] = input_df['Oldpeak'].astype(float)

        # 4. Perform One-Hot Encoding on the single row
        input_df_encoded = pd.get_dummies(input_df)

        # 5. Reindex to match the model's training columns
        input_df_reindexed = input_df_encoded.reindex(columns=encoded_feature_names, fill_value=0)

        # 6. Scale the data using the saved scaler
        features_scaled = scaler.transform(input_df_reindexed)

        # 7. Make the prediction
        prediction_value = model.predict(features_scaled)[0] # Get 0 or 1
        probabilities = model.predict_proba(features_scaled)

        # 8. --- NEW LOGIC ---
        # Get the probability of class 1 (heart failure) REGARDLESS of the final prediction.
        # This gives us the "risk percentage" the user wanted.
        probability_of_failure = probabilities[0][1]
        
        # Send the risk percentage and the final prediction
        return jsonify({
            'prediction_value': int(prediction_value),
            'probability_failure_percent': f"{probability_of_failure * 100:.2f}"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An server error occurred: {str(e)}'}), 500

# --- Run the App ---

def open_browser():
    """
    Opens the default web browser to the app's URL.
    """
    webbrowser.open_new_tab("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # This 'if' statement prevents the browser from
    # opening a new tab every time the server reloads
    # in debug mode (which is caused by os.environ.get).
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # Start a timer to open the browser 1 second
        # after the server starts.
        Timer(1, open_browser).start()
        
    app.run(debug=True, host='0.0.0.0', port=5000)