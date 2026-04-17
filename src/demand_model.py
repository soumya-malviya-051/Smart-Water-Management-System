"""
MODULE 3: Demand Prediction (ML)
================================
Supports multiple ML models for water-demand prediction:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor (default — most robust)

Input : temperature, lag usage, rolling mean, day-of-week indicators.
Output: predicted daily water demand (litres).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature columns used by all models
FEATURE_COLS = ["temperature_C", "day_of_week", "is_weekend",
                "usage_lag_1", "usage_lag_7", "usage_rolling_7"]
TARGET_COL = "usage_litres"

# Available model registry
MODEL_REGISTRY = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, min_samples_split=5, random_state=42
    ),
}

DEFAULT_MODEL = "Gradient Boosting"


class DemandPredictor:
    """Multi-model wrapper for water-demand prediction."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Parameters
        ----------
        model_name : str
            One of: 'Linear Regression', 'Random Forest', 'Gradient Boosting'
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
        self.model_name = model_name
        self.model = MODEL_REGISTRY[model_name]()
        self.is_trained = False

    # ── Training ────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Train the model and return evaluation metrics on the test split.

        Returns
        -------
        metrics : dict  {'MAE', 'RMSE', 'R2', 'model_name'}
        """
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "R2": round(r2_score(y_test, y_pred), 4),
            "model_name": self.model_name,
        }
        print(f"[DemandModel] {self.model_name} — MAE={metrics['MAE']}, "
              f"RMSE={metrics['RMSE']}, R²={metrics['R2']}")
        return metrics

    # ── Feature importances (for tree-based models) ─────────────────────────
    def get_feature_importances(self) -> dict:
        """Return feature importances if available (tree-based models only)."""
        if not self.is_trained:
            return {}
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(FEATURE_COLS, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            return dict(zip(FEATURE_COLS, self.model.coef_))
        return {}

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict demand for each row in *df*."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        X = df[FEATURE_COLS].values
        return self.model.predict(X)

    def predict_single(self, temperature: float, day_of_week: int, is_weekend: int,
                       lag_1: float, lag_7: float, rolling_7: float) -> float:
        """Convenience method: predict for a single set of feature values."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        X = np.array([[temperature, day_of_week, is_weekend, lag_1, lag_7, rolling_7]])
        return float(self.model.predict(X)[0])


# ─── Standalone Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import preprocess

    _, usage_features = preprocess()

    # Compare all models
    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    for name in MODEL_REGISTRY:
        predictor = DemandPredictor(model_name=name)
        m = predictor.train(usage_features)
        importances = predictor.get_feature_importances()
        if importances:
            top = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"  Top features: {[(f, round(v, 4)) for f, v in top]}")
        print()
