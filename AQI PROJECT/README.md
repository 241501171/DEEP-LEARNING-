# AQI Prediction: ML + Deep Learning Mini Project

## 1. Problem
Predict the Air Quality Index (AQI) from pollutant readings and classify it into
the official categories (Good ... Severe).

## 2. Dataset
Kaggle "Air Quality Data in India (2015-2020)", `city_day.csv`.
- Rows with missing AQI dropped; missing pollutant values imputed with the median
- Features: PM2.5, PM10, NO, NO2, NH3, CO, SO2, O3
- Engineered feature: `pm_ratio` = PM2.5 / PM10
- Features standardised; 80/20 train-test split

## 3. Models
Linear Regression, Random Forest, Gradient Boosting (RandomizedSearchCV, 10 iterations, 3-fold CV), MLP (PyTorch, 64-32-1).

## 4. Results
(Paste the table from `model/results.csv` here)

| Model | MAE | RMSE | R2 | Accuracy | F1 |
|---|---|---|---|---|---|

## 5. Architecture
User -> `static/index.html` (form) -> `POST /predict` (FastAPI, `app.py`) -> saved model -> AQI + category -> page

## 6. Run
```
pip install -r requirements.txt
python train.py
uvicorn app:app --reload          # http://localhost:8000
docker build -t aqi-app .
docker run -p 8000:8000 aqi-app
```

## 7. Challenges
- (e.g. many missing values in pollutant columns, handled with median imputation)
- (e.g. AQI is heavily right-skewed, so a few extreme values inflate RMSE)

## 8. Testing
(Add screenshots of 3 inputs: one Good, one Poor, one Severe)

## 9. Future work
City and seasonal features, time-series models (LSTM), live data from a pollution API.
