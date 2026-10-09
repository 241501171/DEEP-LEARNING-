"""FastAPI backend: serves the prediction API and the frontend page."""
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import add_features, aqi_bucket

bundle = joblib.load("model/aqi_model.joblib")
prep, model = bundle["prep"], bundle["model"]

app = FastAPI(title="AQI Predictor")


class Pollutants(BaseModel):
    pm25: float = Field(..., ge=0, description="PM2.5 (µg/m³)")
    pm10: float = Field(..., ge=0, description="PM10 (µg/m³)")
    no: float = Field(..., ge=0)
    no2: float = Field(..., ge=0)
    nh3: float = Field(..., ge=0)
    co: float = Field(..., ge=0)
    so2: float = Field(..., ge=0)
    o3: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["name"]}


@app.post("/predict")
def predict(p: Pollutants):
    row = pd.DataFrame([{
        "PM2.5": p.pm25, "PM10": p.pm10, "NO": p.no, "NO2": p.no2,
        "NH3": p.nh3, "CO": p.co, "SO2": p.so2, "O3": p.o3,
    }])
    x = prep.transform(add_features(row))
    aqi = max(0.0, float(model.predict(x)[0]))
    return {"aqi": round(aqi, 1), "category": aqi_bucket([aqi])[0]}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
