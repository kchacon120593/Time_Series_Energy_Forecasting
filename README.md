# ⏱️ Time Series Forecasting – Energy Consumption

This project explores time series forecasting methods to predict future household energy consumption. Using historical electricity usage data, it applies both statistical (ARIMA) and machine learning (XGBoost & Facebook Prophet) models to detect trends, seasonality, and make short-term hourly forecasts.

---

## 📊 Objective

To develop, compare, and visualize time series models that can support decision-making in energy management, resource allocation, and sustainability initiatives. The focus is on using reproducible, modular code and applying models that can scale to real-world use cases.

---

## 🧠 Methods Used
- **Data Preprocessing:** Data loading and Indexing
- **Time Series Models:**
  - ARIMA (AutoRegressive Integrated Moving Average)
  - XGBoost
  - Facebook Prophet
- **Forecast Evaluation & Visualization**

---

## 📂 Dataset

- **Source:** Kaggle Datasets
  [Electricity consumption UK 2009-2024](https://www.kaggle.com/datasets/albertovidalrod/electricity-consumption-uk-20092022)

- **Description:** National Grid ESO is the electricity system operator for Great Britain. They have gathered information of the electricity demand in Great Britain from 2009. The is updated twice an hour, which means 48 entries per day. This makes this dataset ideal for time series forecasting

---

## 🧪 How to Run

- **Clone this repository:**
  - git clone https://github.com/kchacon120593/time-series-energy-forecasting.git
  - cd time-series-energy-forecasting

- **Open the Jupyter notebook:**
  - time_series_forecasting.ipynb

- **Run the notebook cells to:**
  - Load and preprocess the dataset
  - Use functions from the .py scripts inside src/
  -  Train ARIMA, XGBoost, and Prophet models
  - Visualize forecast results

---

## 🧠 Future Work
  - Extend to deep learning models (LSTM, Transformer)
  - Automate pipeline and deploy as API (MLOps)

