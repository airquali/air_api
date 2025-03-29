import numpy as np
import tensorflow as tf
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained LSTM model
model = tf.keras.models.load_model("air_quality_model_v5.h5")

# Load the saved scaler
with open("air_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Define expected input format
class InputData(BaseModel):
    sequence: list  # List of 7 days, each with 9 features

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input into numpy array
        X_input = np.array(data.sequence).reshape(1, 7, 9)

        # Reshape to 2D and scale (since scaler was fitted on 2D data)
        X_reshaped = X_input.reshape(-1, 9)  # Convert (7,9) → (7*9,)
        X_scaled = scaler.transform(X_reshaped).reshape(1, 7, 9)  # Reshape back

        # Make prediction
        y_pred_scaled = model.predict(X_scaled)

        # Inverse transform the predicted values
        y_pred_original = scaler.inverse_transform(y_pred_scaled)

        return {"prediction": y_pred_original.tolist()}

    except Exception as e:
        return {"error": str(e)}
