from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and scaler
MODEL_PATH = "air_quality_model_v5.h5"
SCALER_PATH = "air_scaler.pkl"

model = load_model(MODEL_PATH, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler = joblib.load(SCALER_PATH)

# Features and sequence length
FEATURES = ['CO(ppm)', 'SO2(ppm)', 'NO2(ppm)', 'O3(ppm)', 'PM2.5', 'PM10', 'H2S', 'CO2(ppm)', 'TVOC(ppb)']
SEQ_LENGTH = 7  # Expecting 7 time steps

app = Flask(__name__)

def preprocess_input(data):
    """Scale and reshape input data for LSTM prediction."""
    try:
        df = pd.DataFrame(data, columns=FEATURES)
        scaled_data = scaler.transform(df)  # Scale input features
        scaled_data = np.array(scaled_data).reshape(1, SEQ_LENGTH, len(FEATURES))  # Reshape for LSTM
        return scaled_data
    except Exception as e:
        return str(e)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]  # Should be a list of lists (7 time steps of 9 features)
        if len(data) != SEQ_LENGTH or len(data[0]) != len(FEATURES):
            return jsonify({"error": "Invalid input shape. Expected (7, 9)"}), 400

        processed_data = preprocess_input(data)
        if isinstance(processed_data, str):
            return jsonify({"error": processed_data}), 400  # Return error if preprocessing fails

        prediction = model.predict(processed_data)
        return jsonify({"prediction": prediction.tolist()})  # Convert to JSON format

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
