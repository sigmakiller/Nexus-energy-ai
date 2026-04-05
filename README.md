# Nexus Energy AI

AI-driven real-time energy prediction dashboard for commercial buildings. This project features machine learning models developed for hourly energy forecasting, temperature forecasting, and anomaly detection to optimize building energy usage and detect faulty systems.

## AI Model Persistence and Accuracy

The repository trains multiple AI models to predict energy usage and identify anomalous patterns. The most prominent metrics used to evaluate the AI trained in this repository are **Mean Absolute Error (MAE)** and **Reconstruction Error**.

### 1. Mean Absolute Error (MAE)
MAE is the primary performance indicator across our forecasting models, measuring the average magnitude of errors without considering their direction.

**Hourly Energy Forecasting:**
*   **Normalized MAE:** Used during model compilation (`loss="mae"`) to evaluate error on scaled data.
*   **Unnormalized MAE (kWh):** Crucial for real-world assessment.
    *   **HVAC:** ~0.68 kWh
    *   **Lighting:** ~0.15 kWh
    *   **MELS (Miscellaneous Electric Loads):** ~0.18 kWh

**Hourly Temperature Forecasting:**
*   Evaluates accuracy in degrees Celsius.
*   **Temp Zone 016:** Achieved an unnormalized MAE of **0.13°C**.

### 2. Reconstruction Error (Loss)
For the **Anomaly Detection** model (an LSTM Autoencoder), the most prominent metric is the reconstruction error, calculated as MAE.
*   The model is trained to compress and then reconstruct "normal" energy patterns.
*   Anomalies are identified when the reconstruction error exceeds a certain threshold, indicating a significant deviation from normal patterns.

### 3. Business & Efficiency Metrics
High-level outcome metrics are tracked to justify the utility of the AI:
*   **Percent Savings:** Calculated by simulating load-shifting strategies (e.g., shifting HVAC usage from peak to off-peak hours).
*   **Energy Waste Reduction:** The project aims for a real-world benefit of saving up to **30% in wasted energy costs** by detecting faulty HVAC systems early through anomaly detection.
