"""
MODULE 4: Rainwater Harvesting Estimation
==========================================
Formula:  Harvested (litres) = Rainfall_mm × Catchment_Area_m2 × Efficiency

Default assumptions:
  - Catchment area : 200 m² (typical small building rooftop)
  - Collection efficiency : 0.80 (80 %)
"""

import pandas as pd
import numpy as np

# ─── Default Parameters ─────────────────────────────────────────────────────────
DEFAULT_CATCHMENT_AREA_M2 = 200    # m²
DEFAULT_EFFICIENCY = 0.80          # 80 %


def estimate_rainwater(
    rainfall_mm: float,
    catchment_area: float = DEFAULT_CATCHMENT_AREA_M2,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> float:
    """
    Estimate harvested rainwater for a single rainfall value.

    Parameters
    ----------
    rainfall_mm : float
        Rainfall in millimetres.
    catchment_area : float
        Roof / catchment surface area in m².
    efficiency : float
        Collection efficiency (0–1).

    Returns
    -------
    float
        Harvested water in litres.
        (1 mm of rain on 1 m² = 1 litre)
    """
    return round(rainfall_mm * catchment_area * efficiency, 2)


def estimate_rainwater_series(
    weather_df: pd.DataFrame,
    catchment_area: float = DEFAULT_CATCHMENT_AREA_M2,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> pd.DataFrame:
    """
    Add a 'harvested_litres' column to the weather DataFrame.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Must contain a 'rainfall_mm' column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with an extra 'harvested_litres' column.
    """
    df = weather_df.copy()
    df["harvested_litres"] = np.round(
        df["rainfall_mm"] * catchment_area * efficiency, 2
    )
    total = df["harvested_litres"].sum()
    print(f"[Rainwater] Total estimated harvest over period: {total:,.1f} litres "
          f"(area={catchment_area} m², eff={efficiency:.0%})")
    return df


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import preprocess

    weather, _ = preprocess()
    weather = estimate_rainwater_series(weather)
    print("\n--- Rainwater Harvest (first 10 rows) ---")
    print(weather[["timestamp", "rainfall_mm", "harvested_litres"]].head(10).to_string(index=False))
