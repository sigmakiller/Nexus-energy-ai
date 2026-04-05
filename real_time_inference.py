import os
import time
import numpy as np
import pandas as pd
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

class RealTimePredictor:
    def __init__(self, model_path, scaler_path, sequence_length=8):
        print(f"Loading Model from {model_path} ...")
        self.model = load_model(model_path)
        
        print(f"Loading Scaler from {scaler_path} ...")
        self.scaler = joblib.load(scaler_path)
        
        self.sequence_length = sequence_length
        self.data_buffer = []

    def process_incoming_data(self, hvac, lighting, mels):
        """
        Receives new real-time data point, scales it, adds to buffer, and predicts if buffer is full.
        Returns the raw predicted values if prediction is made, otherwise None.
        """
        # 1. Prepare raw input array
        raw_data = np.array([[hvac, lighting, mels]])
        
        # 2. Scale the data point
        scaled_data = self.scaler.transform(raw_data)
        
        # 3. Add to buffer
        self.data_buffer.append(scaled_data[0])
        
        # 4. Limit buffer to recent `sequence_length` steps
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)
            
        # 5. Predict using the model when we have enough data (8 timesteps = 40 mins)
        if len(self.data_buffer) == self.sequence_length:
            # Reshape input to (batch_size, time_steps, features)
            model_input = np.array([self.data_buffer]) 
            
            # Predict the next time step (normalized)
            scaled_prediction = self.model.predict(model_input, verbose=0)
            
            # Inverse transform to get true values (kWH, etc.)
            raw_prediction = self.scaler.inverse_transform(scaled_prediction)
            
            return {
                "hvac_pred": raw_prediction[0][0],
                "lighting_pred": raw_prediction[0][1],
                "mels_pred": raw_prediction[0][2]
            }
        
        # Not enough data yet
        return None

def simulate_real_time_stream(csv_file, predictor, delay=1.0):
    """
    Simulates a real-time data feed by iterating over a CSV file with raw unscaled data.
    """
    print(f"\n--- Starting Real-Time Simulation Stream ---\n")
    print(f"Reading from local API/Stream equivalent: {csv_file}")
    print(f"Update frequency configured to every {delay} seconds (Simulating 5-minute intervals).")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Simulation data not found at {csv_file}.")
        print("Please run `train_model.py` to generate the required model and data artifacts first.")
        return

    print("Buffering... waiting for first 8 timesteps to gather enough context data.\n")
    start_idx = 0
    
    for i in range(start_idx, min(start_idx + 100, len(df))): # Run for 100 iterations as a demo
        row = df.iloc[i]
        
        hvac = row['total_hvac']
        lighting = row['total_lighting']
        mels = row['total_mels']
        timestamp = row['date']
        
        print(f"[{timestamp}] LIVE IN: HVAC: {hvac:.2f} | Lighting: {lighting:.2f} | MELS: {mels:.2f}")
        
        # Process continuous data
        prediction = predictor.process_incoming_data(hvac, lighting, mels)
        
        if prediction:
            print(f"    -> AI MODEL PREDICTED NEXT STEP:")
            print(f"       -> HVAC: {prediction['hvac_pred']:.2f}")
            print(f"       -> Lighting: {prediction['lighting_pred']:.2f}")
            print(f"       -> MELS: {prediction['mels_pred']:.2f}\n")
            
        time.sleep(delay)

    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "hourly_energy_prediction.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_y.pkl")
    SIMULATION_DATA = os.path.join(BASE_DIR, "data", "simulated_real_time_stream.csv")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found locally. Please run `train_model.py` first to download data, train, and save the models.")
        exit(1)
        
    predictor = RealTimePredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH, sequence_length=8)
    
    simulate_real_time_stream(SIMULATION_DATA, predictor, delay=0.5)

