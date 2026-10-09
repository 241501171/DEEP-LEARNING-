"""Shared feature definitions, used by both train.py and app.py."""
import pandas as pd

BASE_FEATURES = ["PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3"]
ALL_FEATURES = BASE_FEATURES + ["pm_ratio"]

# Official Indian AQI categories
BINS = [-1, 50, 100, 200, 300, 400, float("inf")]
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: ratio of fine to coarse particles."""
    df = df.copy()
    df["pm_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)
    return df[ALL_FEATURES]


def aqi_bucket(values):
    """Convert numeric AQI values into category labels."""
    return pd.cut(pd.Series(values), bins=BINS, labels=LABELS).astype(str).tolist()
