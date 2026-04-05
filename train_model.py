import os
import glob
import numpy as np
import pandas as pd
import joblib

# Google Drive downloader
import gdown

# Scikit-learn (Data Processing & ML)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# TensorFlow & Keras (Deep Learning)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def main():
    # Define Local Folder Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER = os.path.join(BASE_DIR, "data")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")

    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # Define Google Drive Folder ID
    FOLDER_ID = "1VM2BEzEf6DWlQJjtFA5V4z5oq1DteCGd"
    local_data_folder = os.path.join(DATA_FOLDER, "Building_59_clean_data")
    
    print("\n--- 1. Downloading Data ---")
    if not os.path.exists(local_data_folder):  
        print("Downloading dataset from Google Drive...")
        os.makedirs(local_data_folder, exist_ok=True)
        gdown.download_folder(f"https://drive.google.com/drive/folders/{FOLDER_ID}", output=local_data_folder, quiet=False, use_cookies=False)
    else:
        print("Dataset already exists locally, skipping download.")

    csv_files = glob.glob(os.path.join(local_data_folder, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files in dataset!")

    essential_sensor_files = [
        "ele.csv", "rtu_sa_t.csv", "zone_temp_interior.csv", 
        "zone_temp_exterior.csv", "rtu_fan_spd.csv", "ashp_meter.csv", 
        "ashp_cw.csv", "ashp_hw.csv"
    ]

    filtered_files = [f for f in csv_files if os.path.basename(f) in essential_sensor_files]
    print(f"Found {len(filtered_files)} essential sensor CSV files!")

    print("\n--- 2. Processing Data ---")
    full_time_index = pd.date_range(start="2020-01-01 00:00:00", end="2020-12-31 23:55:00", freq="5min")
    full_time_df = pd.DataFrame({"date": full_time_index})

    dfs = []
    for file in filtered_files:
        print(f"Loading {os.path.basename(file)} ...")
        df = pd.read_csv(file)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = full_time_df.merge(df, on="date", how="left")
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer", suffixes=("", "_dup"))

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    print("\nInterpolating missing data...")
    # Fix interpolation Warning
    merged_df = merged_df.copy()
    for col in merged_df.select_dtypes(include=[np.number]).columns:
        merged_df[col] = merged_df[col].interpolate(method="linear")
    merged_df.bfill(inplace=True)
    merged_df.ffill(inplace=True)

    print("\n--- 3. Normalizing Data ---")
    merged_df["total_hvac"] = (merged_df["hvac_N"] + merged_df["hvac_S"]) / 2
    merged_df["total_lighting"] = merged_df["lig_S"]
    merged_df["total_mels"] = (merged_df["mels_N"] + merged_df["mels_S"]) / 2

    # Fill NaN for created columns if they are fully nan before bfill/ffill
    for col in ["total_hvac", "total_lighting", "total_mels"]:
        merged_df[col] = merged_df[col].fillna(0)

    numeric_cols = merged_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    for col in ["total_hvac", "total_lighting", "total_mels"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    target_cols = ["total_hvac", "total_lighting", "total_mels"]
    
    # Save a copy of raw target data for real-time simulation testing before scaling
    raw_stream_df = merged_df[["date"] + target_cols].copy()
    raw_stream_df.to_csv(os.path.join(DATA_FOLDER, "simulated_real_time_stream.csv"), index=False)

    scaler_y = MinMaxScaler()
    scaler_features = MinMaxScaler()

    merged_df[numeric_cols] = scaler_features.fit_transform(merged_df[numeric_cols])
    merged_df[target_cols] = scaler_y.fit_transform(merged_df[target_cols])

    print("Saving processed files and scalers...")
    merged_df.to_csv(os.path.join(DATA_FOLDER, "processed_data.csv"), index=False)
    
    joblib.dump(scaler_features, os.path.join(MODEL_FOLDER, "scaler_features.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_FOLDER, "scaler_y.pkl"))

    print("\n--- 4. Training Model ---")
    df_3m = merged_df[(merged_df["date"] >= "2020-01-01") & (merged_df["date"] < "2020-04-01")].copy()
    
    # We only use past values of targets to predict future targets
    data = df_3m[["total_hvac", "total_lighting", "total_mels"]].values

    time_steps = 8
    X = np.array([data[i:i+time_steps] for i in range(0, len(data) - time_steps)])
    y = np.array([data[i+time_steps] for i in range(0, len(data) - time_steps)])

    print(f"Training Data Shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(32, activation="relu", return_sequences=True, input_shape=(time_steps, 3)),
        LSTM(16, activation="relu"),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mae")

    print("\nStarting Training...")
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=1)

    model_path = os.path.join(MODEL_FOLDER, "hourly_energy_prediction.keras")
    model.save(model_path)
    print(f"\nModel securely saved at {model_path}")

if __name__ == "__main__":
    main()
