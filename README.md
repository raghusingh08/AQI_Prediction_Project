# AQI (PM2.5) Forecasting – Delhi Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![Machine Learning](https://img.shields.io/badge/ML-Project-green.svg)]()
[![Random Forest](https://img.shields.io/badge/Model-RandomForest-yellow.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

# Air Quality Index (AQI) Forecasting – Delhi

### Next-Day PM2.5 Prediction Using Machine Learning

This project focuses on forecasting next-day PM2.5 levels in Delhi using historical air quality and weather data. Delhi faces extremely high pollution, especially in winters. Predicting PM2.5 is important for public safety, planning, and government policy‑making.

This project uses Machine Learning (Random Forest Regressor) along with time‑series feature engineering to build a reliable AQI forecasting model.

---

## Project Objectives

* Predict next-day PM2.5 concentration.
* Analyze pollution trends using historical AQI & weather data.
* Build a complete ML pipeline including:

  * Preprocessing
  * Feature Engineering
  * Model Training
  * Evaluation
* Provide a practical forecasting solution for early warnings and air quality planning.

---

## Dataset Overview

The dataset contains daily AQI & weather measurements, designed similar to real Delhi CPCB data.

**Features Used:**

* PM2.5 (Target variable)
* PM10
* NO2
* SO2
* CO
* O3
* Temperature
* Humidity
* Wind Speed
* Date

**Time Duration:** ~1.5 years of daily observations.

---

## Preprocessing Steps

* Date parsing and chronological sorting
* Missing value handling (median imputation)
* Feature scaling using StandardScaler
* Dropping invalid rows due to lag features
* Time-based split (80% training, 20% validation)

---

## Feature Engineering

To improve prediction accuracy, the following features were added:

### 🔹 Lag Features

* PM2.5_lag1 to PM2.5_lag14

### 🔹 Rolling Means

* 7-day rolling mean
* 14-day rolling mean

### 🔹 Date-Time Features

* Day of week
* Month
* Day of year

These features help capture seasonal variations and weekly pollution cycles.

---

## Model Used

### Random Forest Regressor

Chosen because:

* Handles non-linear patterns
* High accuracy on tabular data
* Robust to noise
* Minimal tuning required
* Works well for time-series‑like tabular data

---

## Training & Validation

* Used Time‑Series Split to avoid data leakage
* Trained on earlier data
* Validated on last 20% recent days
* Forecasting horizon: **1 day ahead**

---

## Evaluation Metrics

Model performance was evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**
* **R² Score**

Also generated an **Actual vs Predicted PM2.5** plot for visualization.

---

## Project Structure

```
AQI_Prediction_Project/
│
├── dataset/
│     └── sample_delhi_aqi.csv
│
├── src/
│     ├── train.py
│     └── utils.py
│
├── outputs/
│     ├── model.pkl
│     ├── metrics.json
│     ├── actual_vs_pred.png
│     └── feature_importance.png
│
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Train the Model

```
python src/train.py --data dataset/sample_delhi_aqi.csv --target pm25 --horizon 1
```

### Outputs will be saved in `/outputs` folder:

* model.pkl
* metrics.json
* actual_vs_pred.png
* feature_importance.png

---

## Applications

* Government pollution monitoring
* Public alert systems
* Hospital health advisories
* Traffic planning
* Air purifier companies
* Weather & AQI forecasting mobile apps

---

## Conclusion

This project successfully predicts next-day PM2.5 concentration in Delhi using a Random Forest model. The pipeline captures pollution trends, seasonal patterns, and sudden spikes effectively. It demonstrates a practical ML solution for real‑world air quality forecasting.

---

## Future Enhancements

* Use LSTM / GRU deep learning models
* Add rainfall, wind direction & traffic data
* Deploy model as an API or web dashboard
* Multi-step forecasting (3–7 days ahead)

---

## Author

**Raghubar Kushwaha**
B.Tech CSE – Data Science and Artificial Intelligence
