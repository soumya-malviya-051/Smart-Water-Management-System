"""
MODULE 6: Decision Engine
=========================
Rule-based logic that produces actionable recommendations based on:
  - rainfall forecast
  - water-balance status (surplus / deficit)
  - predicted demand
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Recommendation:
    """A single recommendation item."""
    priority: str       # "HIGH" | "MEDIUM" | "LOW"
    category: str       # e.g. "STORAGE", "RECYCLING", "CONSERVATION"
    message: str


def generate_recommendations(
    rainfall_mm: float,
    balance: float,
    predicted_demand: float,
    total_available: float,
    status: str,
) -> List[Recommendation]:
    """
    Produce a list of recommendations given current conditions.

    Parameters
    ----------
    rainfall_mm : float       – today's rainfall in mm
    balance : float           – surplus (+) / deficit (−) litres
    predicted_demand : float  – ML-predicted demand (litres)
    total_available : float   – total available water (litres)
    status : str              – "SURPLUS" | "DEFICIT" | "BALANCED"

    Returns
    -------
    list[Recommendation]
    """
    recs: List[Recommendation] = []

    # ── Rainfall-based rules ────────────────────────────────────────────────
    if rainfall_mm > 10:
        recs.append(Recommendation(
            priority="HIGH",
            category="STORAGE",
            message=(
                f"Heavy rainfall detected ({rainfall_mm:.1f} mm). "
                "Maximise rainwater harvesting — ensure all collection systems are active "
                "and divert overflow to secondary storage."
            ),
        ))
    elif rainfall_mm > 3:
        recs.append(Recommendation(
            priority="MEDIUM",
            category="STORAGE",
            message=(
                f"Moderate rainfall expected ({rainfall_mm:.1f} mm). "
                "Activate rainwater collection and check gutter filters."
            ),
        ))
    else:
        recs.append(Recommendation(
            priority="LOW",
            category="SUPPLY",
            message=(
                f"Little to no rainfall expected ({rainfall_mm:.1f} mm). "
                "Rely on stored water and external supply."
            ),
        ))

    # ── Balance-based rules ─────────────────────────────────────────────────
    if status == "DEFICIT":
        deficit_pct = abs(balance) / max(predicted_demand, 1) * 100
        if deficit_pct > 30:
            recs.append(Recommendation(
                priority="HIGH",
                category="CONSERVATION",
                message=(
                    f"⚠️ Severe water deficit ({balance:+.0f} L, {deficit_pct:.0f}% of demand). "
                    "Immediately reduce non-essential usage (irrigation, washing). "
                    "Consider activating emergency supply."
                ),
            ))
        else:
            recs.append(Recommendation(
                priority="MEDIUM",
                category="RECYCLING",
                message=(
                    f"Water deficit detected ({balance:+.0f} L). "
                    "Increase use of recycled grey-water for non-potable needs."
                ),
            ))

    elif status == "SURPLUS":
        recs.append(Recommendation(
            priority="LOW",
            category="STORAGE",
            message=(
                f"Water surplus detected ({balance:+.0f} L). "
                "Store excess water; consider topping up reserves or irrigating green spaces."
            ),
        ))

    else:  # BALANCED
        recs.append(Recommendation(
            priority="LOW",
            category="MAINTENANCE",
            message="Water supply and demand are balanced. Continue normal operations.",
        ))

    # ── Demand-based rules ──────────────────────────────────────────────────
    if predicted_demand > 700:
        recs.append(Recommendation(
            priority="MEDIUM",
            category="CONSERVATION",
            message=(
                f"High demand forecast ({predicted_demand:.0f} L). "
                "Schedule water-intensive tasks (laundry, car wash) during off-peak hours."
            ),
        ))

    return recs


def format_recommendations(recs: List[Recommendation]) -> str:
    """Return a human-readable string of all recommendations."""
    if not recs:
        return "No recommendations at this time."

    lines = ["=" * 60, "  SMART WATER MANAGEMENT RECOMMENDATIONS", "=" * 60]
    for i, r in enumerate(recs, 1):
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.priority, "⚪")
        lines.append(f"\n{icon} [{r.priority}] {r.category}")
        lines.append(f"   {r.message}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test scenario: moderate rain + deficit
    recs = generate_recommendations(
        rainfall_mm=5.2,
        balance=-120,
        predicted_demand=650,
        total_available=530,
        status="DEFICIT",
    )
    print(format_recommendations(recs))

    print("\n\n--- Scenario 2: Heavy rain + surplus ---")
    recs2 = generate_recommendations(
        rainfall_mm=18.0,
        balance=320,
        predicted_demand=480,
        total_available=800,
        status="SURPLUS",
    )
    print(format_recommendations(recs2))
