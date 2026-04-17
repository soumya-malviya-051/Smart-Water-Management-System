"""
main.py — Orchestrator
======================
Runs the full pipeline end-to-end (CLI mode, no dashboard).

Execution flow:
  1. Fetch / generate data (city-aware)
  2. Preprocess (in-memory)
  3. Train prediction model (Gradient Boosting default)
  4. Estimate rainwater
  5. Compute water balance
  6. Generate recommendations
  7. Print summary report
"""

import os
import sys
import argparse
import numpy as np

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_collection import fetch_weather_data, generate_synthetic_usage, save_data
from preprocessing import clean_and_convert, engineer_features
from demand_model import DemandPredictor, MODEL_REGISTRY, DEFAULT_MODEL
from rainwater import estimate_rainwater_series
from water_balance import simulate_balance_over_period, TANK_CAPACITY
from decision_engine import generate_recommendations, format_recommendations


def main(city: str = "Mumbai", model_name: str = DEFAULT_MODEL, days: int = 90):
    print("=" * 60)
    print("  Water Resource Intelligence Platform")
    print("  Enterprise Edition")
    print(f"  City: {city}  |  Model: {model_name}")
    print("=" * 60)

    # ─── 1. Data Collection ──────────────────────────────────────────────────
    print(f"\n[1/7] Data Collection for {city}")
    weather_df = fetch_weather_data(city=city, days=days)
    usage_df = generate_synthetic_usage(city=city, days=days)
    save_data(weather_df, usage_df)

    # ─── 2. Preprocessing (in-memory) ────────────────────────────────────────
    print("\n[2/7] Preprocessing")
    weather_clean = clean_and_convert(weather_df)
    usage_clean = clean_and_convert(usage_df)
    usage_features = engineer_features(usage_clean)

    # ─── 3. Demand Prediction ────────────────────────────────────────────────
    print(f"\n[3/7] Training {model_name} Demand Model")
    predictor = DemandPredictor(model_name=model_name)
    metrics = predictor.train(usage_features)
    predictions = predictor.predict(usage_features)

    # ─── 4. Rainwater Estimation ─────────────────────────────────────────────
    print("\n[4/7] Rainwater Estimation")
    weather_rain = estimate_rainwater_series(weather_clean)

    # ─── 5. Water Balance ────────────────────────────────────────────────────
    print("\n[5/7] Water Balance Simulation")
    n = min(len(usage_features), len(weather_rain))
    demands = predictions[:n].tolist()
    rainfalls = weather_rain["harvested_litres"].values[:n].tolist()
    usages = usage_features["usage_litres"].values[:n].tolist()

    balance_results = simulate_balance_over_period(
        demands, rainfalls, usages,
        initial_storage=TANK_CAPACITY * 0.5,
    )

    # ─── 6. Recommendations ─────────────────────────────────────────────────
    print("\n[6/7] Generating Recommendations")
    latest = balance_results[-1]
    last_rainfall = weather_rain["rainfall_mm"].values[min(n - 1, len(weather_rain) - 1)]
    recs = generate_recommendations(
        rainfall_mm=last_rainfall,
        balance=latest.balance,
        predicted_demand=latest.predicted_demand,
        total_available=latest.total_available,
        status=latest.status,
    )

    # ─── 7. Summary Report ───────────────────────────────────────────────────
    print("\n[7/7] Summary Report")
    print("=" * 60)
    print(f"  City: {city}")
    print(f"  Model: {model_name}")
    print(f"  Performance:  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  R²={metrics['R2']}")
    print(f"  Avg Predicted Demand : {np.mean(demands):,.1f} L/day")
    print(f"  Total Rainwater Harvested : {sum(rainfalls):,.1f} L")
    surplus_days = sum(1 for r in balance_results if r.status == "SURPLUS")
    deficit_days = sum(1 for r in balance_results if r.status == "DEFICIT")
    print(f"  Surplus days: {surplus_days}  |  Deficit days: {deficit_days}  |  Total: {n}")
    print(f"\n  Latest Day Water Balance:")
    for k, v in vars(latest).items():
        print(f"    {k:20s}: {v}")

    print(f"\n{format_recommendations(recs)}")
    print("\n✅ Pipeline complete. Run the dashboard with:")
    print("   streamlit run src/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Water Management CLI")
    parser.add_argument("--city", default="Mumbai", help="City name (default: Mumbai)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"ML model (default: {DEFAULT_MODEL})")
    parser.add_argument("--days", type=int, default=90, help="Simulation days (default: 90)")
    args = parser.parse_args()
    main(city=args.city, model_name=args.model, days=args.days)
