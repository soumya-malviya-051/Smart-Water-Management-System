"""
MODULE 5: Water Balance Model
==============================
Computes total available water and determines surplus / deficit.

Available Water  =  Storage  +  Rainwater  +  External Supply  +  Recycled Water
Balance          =  Available Water  −  Predicted Demand

Assumptions:
  - Storage tank capacity : 5,000 litres (initial fill = 50 %)
  - External daily supply : 400 litres
  - 75 % of usage becomes wastewater
  - 65 % of wastewater is recyclable
"""

from dataclasses import dataclass, field

# ─── Default Parameters ─────────────────────────────────────────────────────────
TANK_CAPACITY = 5000       # litres
INITIAL_STORAGE_FRAC = 0.5
DAILY_EXTERNAL_SUPPLY = 400  # litres / day
WASTEWATER_FRACTION = 0.75   # 75 % of usage → wastewater
RECYCLE_EFFICIENCY = 0.65     # 65 % of wastewater is reusable


@dataclass
class WaterBalanceResult:
    """Container for a single day's water-balance computation."""
    storage: float
    rainwater: float
    external_supply: float
    recycled_water: float
    total_available: float
    predicted_demand: float
    balance: float          # positive = surplus, negative = deficit
    status: str             # "SURPLUS" | "BALANCED" | "DEFICIT"


def compute_recycled_water(usage: float,
                           wastewater_frac: float = WASTEWATER_FRACTION,
                           recycle_eff: float = RECYCLE_EFFICIENCY) -> float:
    """Estimate recyclable water from the previous day's usage."""
    return round(usage * wastewater_frac * recycle_eff, 2)


def compute_balance(
    predicted_demand: float,
    harvested_rainwater: float,
    previous_usage: float,
    current_storage: float = TANK_CAPACITY * INITIAL_STORAGE_FRAC,
    external_supply: float = DAILY_EXTERNAL_SUPPLY,
    wastewater_frac: float = WASTEWATER_FRACTION,
    recycle_eff: float = RECYCLE_EFFICIENCY,
) -> WaterBalanceResult:
    """
    Compute the water balance for a single day.

    Parameters
    ----------
    predicted_demand : float
        ML-predicted demand in litres.
    harvested_rainwater : float
        Estimated rainwater harvest for the day (litres).
    previous_usage : float
        Previous day's actual or estimated usage (litres) for recycling calc.
    current_storage : float
        Current water in storage tank (litres).
    external_supply : float
        Externally supplied water (litres).

    Returns
    -------
    WaterBalanceResult
    """
    recycled = compute_recycled_water(previous_usage, wastewater_frac, recycle_eff)
    total = current_storage + harvested_rainwater + external_supply + recycled
    balance = total - predicted_demand

    if balance > 50:
        status = "SURPLUS"
    elif balance < -50:
        status = "DEFICIT"
    else:
        status = "BALANCED"

    return WaterBalanceResult(
        storage=round(current_storage, 2),
        rainwater=round(harvested_rainwater, 2),
        external_supply=round(external_supply, 2),
        recycled_water=round(recycled, 2),
        total_available=round(total, 2),
        predicted_demand=round(predicted_demand, 2),
        balance=round(balance, 2),
        status=status,
    )


def simulate_balance_over_period(demands, rainfalls, usages,
                                 initial_storage=TANK_CAPACITY * INITIAL_STORAGE_FRAC):
    """
    Run the water-balance model day-by-day over a period.

    Parameters
    ----------
    demands : list[float]   – predicted daily demands (litres)
    rainfalls : list[float] – harvested rainwater per day (litres)
    usages : list[float]    – actual daily usage per day (litres) – for recycling calc

    Returns
    -------
    list[WaterBalanceResult]
    """
    results = []
    storage = initial_storage
    for i in range(len(demands)):
        prev_usage = usages[i - 1] if i > 0 else usages[0]
        result = compute_balance(
            predicted_demand=demands[i],
            harvested_rainwater=rainfalls[i],
            previous_usage=prev_usage,
            current_storage=storage,
        )
        results.append(result)
        # Update storage: carry over surplus (capped by tank), drain on deficit
        storage = min(TANK_CAPACITY, max(0, storage + result.balance))
    return results


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick demo with dummy numbers
    res = compute_balance(
        predicted_demand=600,
        harvested_rainwater=120,
        previous_usage=550,
    )
    print("--- Water Balance Demo ---")
    for k, v in vars(res).items():
        print(f"  {k:20s}: {v}")
