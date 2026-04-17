"""
MODULE 2: Data Preprocessing
=============================
- Handles missing values (forward-fill + interpolation).
- Converts timestamp strings to proper datetime objects.
- Engineers features (lag, rolling mean, day-of-week) for the ML model.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data(data_dir: str = DATA_DIR):
    """
    Load weather and usage CSVs.

    Returns
    -------
    weather_df, usage_df : tuple[pd.DataFrame, pd.DataFrame]
    """
    weather_df = pd.read_csv(os.path.join(data_dir, "weather_data.csv"))
    usage_df = pd.read_csv(os.path.join(data_dir, "usage_data.csv"))
    return weather_df, usage_df


def clean_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    General cleaning:
    1. Convert 'timestamp' to datetime.
    2. Sort by timestamp.
    3. Forward-fill then back-fill missing values.
    4. Interpolate remaining numeric NaNs.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    # Interpolate any remaining NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    return df


def engineer_features(usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features useful for demand prediction:
    - day_of_week (0=Mon … 6=Sun)
    - is_weekend (bool → int)
    - usage_lag_1  (previous day usage)
    - usage_lag_7  (same day last week)
    - usage_rolling_7 (7-day rolling mean)
    """
    df = usage_df.copy()

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features
    df["usage_lag_1"] = df["usage_litres"].shift(1)
    df["usage_lag_7"] = df["usage_litres"].shift(7)

    # Rolling statistics
    df["usage_rolling_7"] = df["usage_litres"].rolling(window=7).mean()

    # Drop rows with NaN introduced by shifting / rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[Preprocessing] Feature-engineered dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def preprocess(data_dir: str = DATA_DIR):
    """
    Full preprocessing pipeline.

    Returns
    -------
    weather_clean : pd.DataFrame
    usage_features : pd.DataFrame  (cleaned + feature-engineered)
    """
    weather_df, usage_df = load_data(data_dir)

    weather_clean = clean_and_convert(weather_df)
    usage_clean = clean_and_convert(usage_df)
    usage_features = engineer_features(usage_clean)

    print("[Preprocessing] Done.")
    return weather_clean, usage_features


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    weather, usage = preprocess()
    print("\n--- Weather (cleaned, first 5) ---")
    print(weather.head())
    print("\n--- Usage (features, first 5) ---")
    print(usage.head())
