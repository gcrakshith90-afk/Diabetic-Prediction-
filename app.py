import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model.pkl"

def load_or_train_model():
    """Load model artifacts or train a new model if model.pkl is missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"'{MODEL_PATH}' not found. Training model automatically...")
        from train import train_and_save_model
        train_and_save_model(data_path="diabetes.csv", output_path=MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    print("Model artifacts loaded successfully.")
    return artifacts

artifacts = load_or_train_model()
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
features_with_zeros = artifacts['features_with_zeros']
median_values = artifacts['median_values']

# Map frontend input names to dataset feature columns
INPUT_MAPPING = {
    'pregnancies': 'Pregnancies',
    'glucose': 'Glucose',
    'bloodPressure': 'BloodPressure',
    'skinThickness': 'SkinThickness',
    'insulin': 'Insulin',
    'bmi': 'BMI',
    'diabetesPedigreeFunction': 'DiabetesPedigreeFunction',
    'age': 'Age'
}

@app.route('/')
def home():
    """Render the patient diabetes prediction dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive patient health metrics and return diabetes prediction."""
    try:
        # Support both JSON payload and Form submission
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()

        patient_record = {}
        for key, feature_col in INPUT_MAPPING.items():
            # Support lowercase and original column casing in requests
            val = data.get(key)
            if val is None:
                val = data.get(feature_col)

            if val is None or str(val).strip() == "":
                return jsonify({"error": f"Missing value for '{feature_col}'."}), 400

            try:
                num_val = float(val)
            except ValueError:
                return jsonify({"error": f"Invalid numeric value for '{feature_col}': {val}"}), 400

            # Biological 0 imputation using trained medians
            if feature_col in features_with_zeros and num_val == 0:
                num_val = median_values[feature_col]

            patient_record[feature_col] = num_val

        # Create DataFrame in exact feature order expected by scaler and model
        df = pd.DataFrame([patient_record], columns=feature_names)
        

        # Scale features
        df_scaled = scaler.transform(df)

        # Predict
        prediction = int(model.predict(df_scaled)[0])
        # Predict probability for class 1 (diabetes)
        probability = float(model.predict_proba(df_scaled)[0][1])
        probability_pct = round(probability * 100, 2)

        risk_level = "High" if probability_pct >= 70 else ("Moderate" if probability_pct >= 40 else "Low")

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probability": probability_pct,
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == '__main__':
    # Run the application
    app.run(host='127.0.0.1', port=5000, debug=True)
