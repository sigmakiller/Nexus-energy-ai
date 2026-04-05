import os
import asyncio
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from real_time_inference import RealTimePredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hourly_energy_prediction.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_y.pkl")
SIMULATION_DATA = os.path.join(BASE_DIR, "data", "simulated_real_time_stream.csv")

predictor = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        predictor = RealTimePredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH, sequence_length=8)
    else:
        print("Model or Scaler not found.")
except Exception as e:
    print(f"Error loading predictor: {e}")

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ...

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/style.css")
def get_style():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "style.css"))

@app.get("/app.js")
def get_app():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "app.js"))

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not predictor:
        await websocket.send_json({"error": "Predictor not initialized. Models missing."})
        await websocket.close()
        return

    try:
        df = pd.read_csv(SIMULATION_DATA)
    except FileNotFoundError:
        await websocket.send_json({"error": "Simulation CSV not found."})
        await websocket.close()
        return

    start_idx = 0
    try:
        # State variables for dynamic metrics
        mae_accumulated = {"hvac": 0.0, "lighting": 0.0, "mels": 0.0}
        prediction_count = 0

        while True:
            for i in range(start_idx, len(df)):
                row = df.iloc[i]
                hvac = float(row['total_hvac'])
                lighting = float(row['total_lighting'])
                mels = float(row['total_mels'])
                timestamp = str(row['date'])

                prediction = predictor.process_incoming_data(hvac, lighting, mels)
                
                payload = {
                    "timestamp": timestamp,
                    "actual": {
                        "hvac": hvac,
                        "lighting": lighting,
                        "mels": mels
                    },
                    "predicted": None,
                    "metrics": None
                }
                
                if prediction:
                    pred_hvac = float(prediction['hvac_pred'])
                    pred_lighting = float(prediction['lighting_pred'])
                    pred_mels = float(prediction['mels_pred'])

                    payload["predicted"] = {
                        "hvac": pred_hvac,
                        "lighting": pred_lighting,
                        "mels": pred_mels
                    }

                    # Calculate Absolute Error for current step (proxy for Reconstruction Error)
                    err_hvac = abs(hvac - pred_hvac)
                    err_lighting = abs(lighting - pred_lighting)
                    err_mels = abs(mels - pred_mels)
                    reconstruction_error = err_hvac + err_lighting + err_mels
                    
                    # Update Running MAE
                    prediction_count += 1
                    mae_accumulated["hvac"] += err_hvac
                    mae_accumulated["lighting"] += err_lighting
                    mae_accumulated["mels"] += err_mels
                    
                    payload["metrics"] = {
                        "reconstruction_error": reconstruction_error,
                        "mae": {
                            "hvac": mae_accumulated["hvac"] / prediction_count,
                            "lighting": mae_accumulated["lighting"] / prediction_count,
                            "mels": mae_accumulated["mels"] / prediction_count
                        }
                    }

                await websocket.send_json(payload)
                await asyncio.sleep(0.5)  # half-second delay for premium, fast feel
            start_idx = 0 # loop back for demo
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    # When run directly with `python main.py`, start the server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
