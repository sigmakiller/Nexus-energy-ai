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

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")

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
                    "predicted": None
                }
                
                if prediction:
                    payload["predicted"] = {
                        "hvac": float(prediction['hvac_pred']),
                        "lighting": float(prediction['lighting_pred']),
                        "mels": float(prediction['mels_pred'])
                    }

                await websocket.send_json(payload)
                await asyncio.sleep(0.5)  # half-second delay for premium, fast feel
            start_idx = 0 # loop back for demo
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
