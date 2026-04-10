# 🏎️ DRIVe-SYNC: AI-Powered Vehicle Health & Driver Safety Platform

DRIVe-SYNC is a predictive machine learning platform designed to forecast vehicle component damage before it becomes expensive and classify real-time driver behavior. Built as a Project-Based Learning (PBL) submission, it relies on real-world driving data rather than simulations.

## ✨ Core Features
* **Predictive Maintenance:** Forecasts wear and tear on brakes, engine, transmission, tires, and suspension.
* **Driver Safety Scoring:** Classifies driving style into safe, aggressive, or highway categories.
* **Hybrid AI Architecture:** Utilizes an ensemble ML model (Random Forest + XGBoost) achieving **97.4% accuracy**, shielded by a physics-based rule guard to catch extreme unsafe patterns.
* **Hardware Integration:** Supports OBD-II integration for live telemetry, with a statistically equivalent synthetic data pipeline for offline testing.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Frontend:** Streamlit, Plotly
* **Machine Learning:** scikit-learn, XGBoost, pandas, NumPy, joblib
* **Hardware Interfacing:** ELM327-emulator, python-obd

## 📊 The Data
The models were trained on three comprehensive datasets:
1. **GPS Telematics:** 475,751 real-world records.
2. **Smartphone Sensor Data:** 216,000 lines of accelerometer/gyroscope logs.
3. **Automobile Specifications:** Baseline data for 205 vehicle models.
*(Note: 17 custom features were engineered from the raw GPS data to capture non-linear interactions between speed, braking force, and event frequency).*

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/DRIVe-SYNC.git](https://github.com/yourusername/DRIVe-SYNC.git)
   cd DRIVe-SYNC
