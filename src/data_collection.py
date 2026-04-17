"""
MODULE 1: Data Collection
=========================
- Fetches weather data from OpenWeatherMap API (temperature, rainfall, timestamps).
- Generates a synthetic daily water-usage dataset influenced by temperature & noise.
- Synthetic data varies by city (different climate profiles).
- Saves both datasets to CSV for downstream consumption.
"""

import os
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CITY = "Mumbai"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ─── City Climate Profiles ──────────────────────────────────────────────────────
# Realistic base temperature and rainfall profiles for Indian cities & a few global ones.
CITY_PROFILES = {
    "mumbai":    {"base_temp": 28, "temp_amplitude": 4,  "rain_scale": 4.0, "base_usage": 520},
    "kolkata":   {"base_temp": 27, "temp_amplitude": 7,  "rain_scale": 3.5, "base_usage": 490},
    "delhi":     {"base_temp": 25, "temp_amplitude": 14, "rain_scale": 2.0, "base_usage": 550},
    "chennai":   {"base_temp": 29, "temp_amplitude": 4,  "rain_scale": 3.0, "base_usage": 500},
    "bangalore": {"base_temp": 24, "temp_amplitude": 3,  "rain_scale": 2.5, "base_usage": 460},
    "hyderabad": {"base_temp": 27, "temp_amplitude": 6,  "rain_scale": 2.2, "base_usage": 480},
    "pune":      {"base_temp": 26, "temp_amplitude": 5,  "rain_scale": 3.0, "base_usage": 470},
    "jaipur":    {"base_temp": 26, "temp_amplitude": 12, "rain_scale": 1.5, "base_usage": 530},
    "london":    {"base_temp": 11, "temp_amplitude": 8,  "rain_scale": 2.0, "base_usage": 380},
    "new york":  {"base_temp": 13, "temp_amplitude": 14, "rain_scale": 2.5, "base_usage": 420},
    "tokyo":     {"base_temp": 16, "temp_amplitude": 12, "rain_scale": 3.0, "base_usage": 400},
    "sydney":    {"base_temp": 18, "temp_amplitude": 6,  "rain_scale": 2.0, "base_usage": 410},
}


def _get_city_profile(city: str) -> dict:
    """Return climate profile for a city. Falls back to a hash-derived profile for unknown cities."""
    key = city.strip().lower()
    if key in CITY_PROFILES:
        return CITY_PROFILES[key]

    # Deterministic fallback: derive profile from city name hash
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return {
        "base_temp": 15 + (h % 20),              # 15–34 °C
        "temp_amplitude": 3 + (h % 12),           # 3–14 °C
        "rain_scale": 1.0 + (h % 40) / 10.0,     # 1.0–5.0
        "base_usage": 400 + (h % 200),            # 400–600 L
    }


def _city_seed(city: str) -> int:
    """Derive a reproducible but city-specific random seed."""
    return int(hashlib.md5(city.strip().lower().encode()).hexdigest(), 16) % (2**31)


def fetch_weather_data(city: str = DEFAULT_CITY, api_key: str = None, days: int = 90) -> pd.DataFrame:
    """
    Fetch historical and forecast weather data from Open-Meteo (No API key required!).
    1. Geocode the city to get lat/lon.
    2. Fetch daily data for the requested past days.
    Falls back to synthetic weather if the network call fails.
    """
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if "results" not in geo_data or not geo_data["results"]:
            raise ValueError(f"City '{city}' not found in geocoding.")
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
        # 2. Weather mapping (max past_days supported by forecast API is 92)
        past = min(days, 92)
        meteo_url = (f"https://api.open-meteo.com/v1/forecast"
                     f"?latitude={lat}&longitude={lon}"
                     f"&past_days={past}&forecast_days=1"
                     f"&daily=temperature_2m_mean,precipitation_sum"
                     f"&timezone=auto")
                     
        w_resp = requests.get(meteo_url, timeout=15)
        w_resp.raise_for_status()
        w_data = w_resp.json()["daily"]
        
        # Extract matching the expected schema
        times = pd.to_datetime(w_data["time"])
        temps = w_data["temperature_2m_mean"]
        rains = w_data["precipitation_sum"]
        
        df = pd.DataFrame({
            "timestamp": times,
            "temperature_C": temps,
            "rainfall_mm": rains,
        })
        # Forward fill any missing temps/rains from Open-mEteo
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Ensure we return exactly the requested number of days (tail of the dataframe)
        df = df.tail(days).reset_index(drop=True)
        df["city"] = city
        
        print(f"[DataCollection] Fetched {len(df)} real weather records for '{city}' via Open-Meteo.")
        return df

    except Exception as e:
        print(f"[DataCollection] Open-Meteo API call failed ({e}). Generating synthetic weather for '{city}'.")
        return _generate_synthetic_weather(city=city, days=days)


def _generate_synthetic_weather(city: str = DEFAULT_CITY, days: int = 90) -> pd.DataFrame:
    """Create city-specific synthetic weather data."""
    profile = _get_city_profile(city)
    seed = _city_seed(city)
    rng = np.random.RandomState(seed)

    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Temperature: city-specific base + amplitude + noise
    base_t = profile["base_temp"]
    amp = profile["temp_amplitude"]
    temps = base_t + amp * np.sin(np.linspace(0, 2 * np.pi, days)) + rng.normal(0, 2, days)

    # Rainfall: city-specific intensity
    rainfall = np.maximum(0, rng.exponential(scale=profile["rain_scale"], size=days) - 1.5)
    rainfall = np.round(rainfall, 2)

    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_C": np.round(temps, 2),
        "rainfall_mm": rainfall,
        "city": city,
    })
    print(f"[DataCollection] Generated {len(df)} synthetic weather records for '{city}' "
          f"(base_temp={base_t}°C, rain_scale={profile['rain_scale']}).")
    return df


def generate_synthetic_usage(city: str = DEFAULT_CITY, days: int = 90) -> pd.DataFrame:
    """
    Generate a city-specific synthetic daily water-usage dataset.
    Usage (litres) = city_base + temperature_effect + day_of_week_effect + noise
    """
    profile = _get_city_profile(city)
    seed = _city_seed(city)
    rng = np.random.RandomState(seed)

    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    base_t = profile["base_temp"]
    amp = profile["temp_amplitude"]
    temps = base_t + amp * np.sin(np.linspace(0, 2 * np.pi, days)) + rng.normal(0, 2, days)

    base_usage = profile["base_usage"]
    temp_effect = (temps - base_t) * 12
    dow_effect = np.array([30 if d.weekday() >= 5 else 0 for d in dates])
    noise = rng.normal(0, 25, days)

    usage = base_usage + temp_effect + dow_effect + noise
    usage = np.maximum(usage, 100)

    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_C": np.round(temps, 2),
        "usage_litres": np.round(usage, 2),
        "city": city,
    })
    print(f"[DataCollection] Generated {len(df)} synthetic usage records for '{city}' "
          f"(base_usage={base_usage} L).")
    return df


def save_data(weather_df: pd.DataFrame, usage_df: pd.DataFrame, out_dir: str = DATA_DIR):
    """Persist both DataFrames to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    weather_path = os.path.join(out_dir, "weather_data.csv")
    usage_path = os.path.join(out_dir, "usage_data.csv")
    weather_df.to_csv(weather_path, index=False)
    usage_df.to_csv(usage_path, index=False)
    print(f"[DataCollection] Saved weather data → {weather_path}")
    print(f"[DataCollection] Saved usage data  → {usage_path}")


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for test_city in ["Mumbai", "Kolkata", "Delhi"]:
        print(f"\n{'='*50}")
        print(f"  City: {test_city}")
        print(f"{'='*50}")
        weather = fetch_weather_data(city=test_city)
        usage = generate_synthetic_usage(city=test_city)
        print(f"  Avg Temp: {weather['temperature_C'].mean():.1f}°C")
        print(f"  Avg Rain: {weather['rainfall_mm'].mean():.2f} mm")
        print(f"  Avg Usage: {usage['usage_litres'].mean():.1f} L")
