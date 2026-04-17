"""
MODULE 7: Streamlit Dashboard
==============================
Interactive dashboard displaying:
  • Predicted water demand (with model selection)
  • Rainfall & rainwater harvest data
  • Water availability (surplus / deficit)
  • Smart recommendations
  • Feature importances for ML model
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─── Ensure src/ is importable ──────────────────────────────────────────────────
SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_collection import fetch_weather_data, generate_synthetic_usage, save_data, CITY_PROFILES
from preprocessing import clean_and_convert, engineer_features
from demand_model import DemandPredictor, MODEL_REGISTRY, DEFAULT_MODEL, FEATURE_COLS
from rainwater import estimate_rainwater_series
from water_balance import simulate_balance_over_period, TANK_CAPACITY
from decision_engine import generate_recommendations

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Resource Intelligence",
    page_icon="🌍",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stApp { background-color: #0b0f19; }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label { color: #94a3b8; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;}
    div[data-testid="stExpander"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Controls ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/water.png", width=80)
st.sidebar.title("⚙️ Settings")

city_opts = [c.title() for c in CITY_PROFILES.keys()]
city = st.sidebar.selectbox("City for weather data", options=city_opts, index=0)
model_choice = st.sidebar.selectbox(
    "ML Model",
    options=list(MODEL_REGISTRY.keys()),
    index=list(MODEL_REGISTRY.keys()).index(DEFAULT_MODEL),
    help="Gradient Boosting is the most robust; Linear Regression is the simplest baseline."
)
catchment_area = st.sidebar.slider("Catchment Area (m²)", 50, 500, 200)
efficiency = st.sidebar.slider("Collection Efficiency (%)", 50, 95, 80) / 100
tank_cap = st.sidebar.slider("Tank Capacity (L)", 1000, 20000, 5000, step=500)
days = st.sidebar.slider("Simulation Days", 30, 180, 90)

run_btn = st.sidebar.button("🚀 Run Simulation", use_container_width=True)


# ─── Main Header ────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Water Resource Intelligence Platform")
st.markdown("*Enterprise Edition — Automated Prediction & Resource Allocation*")
st.divider()

if run_btn:
    # ── Step 1: Data Collection (city-aware) ─────────────────────────────────
    with st.spinner(f"📡 Fetching weather data for **{city}** & generating usage dataset…"):
        weather_df = fetch_weather_data(city=city, days=days)
        usage_df = generate_synthetic_usage(city=city, days=days)
        save_data(weather_df, usage_df)

    # ── Step 2: Preprocessing (in-memory, no stale CSV reload) ───────────────
    with st.spinner("🔧 Preprocessing data…"):
        weather_clean = clean_and_convert(weather_df)
        usage_clean = clean_and_convert(usage_df)
        usage_features = engineer_features(usage_clean)

    # ── Step 3: Train demand model ───────────────────────────────────────────
    with st.spinner(f"🤖 Training **{model_choice}** model…"):
        predictor = DemandPredictor(model_name=model_choice)
        metrics = predictor.train(usage_features)
        predictions = predictor.predict(usage_features)
        usage_features = usage_features.copy()
        usage_features["predicted_demand"] = predictions

    # ── Step 4: Rainwater estimation ─────────────────────────────────────────
    with st.spinner("🌧️ Estimating rainwater harvest…"):
        weather_rain = estimate_rainwater_series(weather_clean, catchment_area, efficiency)

    # ── Step 5 & 6: Water balance & recommendations ──────────────────────────
    with st.spinner("⚖️ Computing water balance & generating recommendations…"):
        n = min(len(usage_features), len(weather_rain))
        demands = predictions[:n].tolist()
        rainfalls = weather_rain["harvested_litres"].values[:n].tolist()
        usages = usage_features["usage_litres"].values[:n].tolist()

        balance_results = simulate_balance_over_period(demands, rainfalls, usages,
                                                       initial_storage=tank_cap * 0.5)

    # ════════════════════════════════════════════════════════════════════════
    #  DASHBOARD DISPLAY
    # ════════════════════════════════════════════════════════════════════════

    st.success(f"✅ Simulation complete for **{city}** using **{model_choice}**!")

    # ── KPI Cards ────────────────────────────────────────────────────────────
    latest = balance_results[-1]
    avg_demand = np.mean(demands)
    total_rain = sum(rainfalls)
    surplus_days = sum(1 for r in balance_results if r.status == "SURPLUS")
    deficit_days = sum(1 for r in balance_results if r.status == "DEFICIT")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Predicted Demand", f"{avg_demand:,.0f} L/day")
    k2.metric("Total Rainwater Harvested", f"{total_rain:,.0f} L")
    k3.metric("Surplus Days", f"{surplus_days}", delta=f"{surplus_days}/{n} days")
    k4.metric("Deficit Days", f"{deficit_days}", delta=f"-{deficit_days}/{n} days", delta_color="inverse")

    st.divider()

    tab_overview, tab_analytics, tab_data = st.tabs(["📊 Overview & Decisions", "📈 Predictive Analytics", "📋 Source Data"])

    with tab_overview:
        # ── Recommendations ──────────────────────────────────────────────────────
        st.subheader("💡 Smart Recommendations")
        last_rainfall = weather_rain["rainfall_mm"].values[min(n - 1, len(weather_rain) - 1)]
        recs = generate_recommendations(
            rainfall_mm=last_rainfall,
            balance=latest.balance,
            predicted_demand=latest.predicted_demand,
            total_available=latest.total_available,
            status=latest.status,
        )
        for r in recs:
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.priority, "⚪")
            st.markdown(f"**{icon} [{r.priority}] {r.category}**")
            st.info(r.message)
            
        # ── Water Availability Breakdown (latest day) ────────────────────────────
        st.subheader("🥧 Water Availability Breakdown (Latest Day)")
        fig_pie = go.Figure(go.Pie(
            labels=["Storage", "Rainwater", "External Supply", "Recycled Water"],
            values=[latest.storage, latest.rainwater, latest.external_supply, latest.recycled_water],
            hole=0.45,
            marker=dict(colors=["#42a5f5", "#66bb6a", "#ffca28", "#ab47bc"]),
        ))
        fig_pie.update_layout(template="plotly_dark", height=380, margin=dict(t=40, b=40))
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab_analytics:
        # ── Model Performance ────────────────────────────────────────────────────
        with st.expander(f"📊 {model_choice} — Model Performance", expanded=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['MAE']:.2f} L")
            m2.metric("RMSE", f"{metrics['RMSE']:.2f} L")
            m3.metric("R² Score", f"{metrics['R2']:.4f}")
    
            # Feature importances chart (tree models)
            importances = predictor.get_feature_importances()
            if importances:
                imp_df = pd.DataFrame({
                    "Feature": list(importances.keys()),
                    "Importance": list(importances.values()),
                }).sort_values("Importance", ascending=True)
    
                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Importance"], y=imp_df["Feature"],
                    orientation="h",
                    marker_color="#3a7bd5",
                ))
                fig_imp.update_layout(
                    template="plotly_dark", title="Feature Importances",
                    height=280, margin=dict(l=120, r=20, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        # ── Demand Chart ─────────────────────────────────────────────────────────
        st.subheader(f"📈 Water Demand: Actual vs Predicted — {city}")
        fig_demand = go.Figure()
        timestamps = usage_features["timestamp"].values[:n]
        fig_demand.add_trace(go.Scatter(
            x=timestamps, y=usages,
            name="Actual Usage", mode="lines",
            line=dict(color="#00d2ff", width=2),
        ))
        fig_demand.add_trace(go.Scatter(
            x=timestamps, y=demands,
            name="Predicted Demand", mode="lines",
            line=dict(color="#ff6f61", width=2, dash="dash"),
        ))
        fig_demand.update_layout(
            template="plotly_dark",
            xaxis_title="Date", yaxis_title="Litres",
            height=400, margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_demand, use_container_width=True)

        # ── Rainfall & Harvest Chart ─────────────────────────────────────────────
        st.subheader(f"🌧️ Rainfall & Rainwater Harvest — {city}")
        rain_ts = weather_rain["timestamp"].values[:n]
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=rain_ts, y=weather_rain["rainfall_mm"].values[:n],
            name="Rainfall (mm)", marker_color="#4fc3f7", opacity=0.7,
        ))
        fig_rain.add_trace(go.Scatter(
            x=rain_ts, y=rainfalls,
            name="Harvested (L)", mode="lines+markers",
            line=dict(color="#66bb6a", width=2), yaxis="y2",
        ))
        fig_rain.update_layout(
            template="plotly_dark",
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Harvested (L)", overlaying="y", side="right"),
            height=400, margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_rain, use_container_width=True)

        # ── Water Balance Chart ──────────────────────────────────────────────────
        st.subheader("⚖️ Water Balance Over Time")
        balances = [r.balance for r in balance_results]
        colors = ["#00e676" if b > 0 else "#ff5252" for b in balances]
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Bar(
            x=list(range(n)), y=balances,
            marker_color=colors, name="Balance (L)",
        ))
        fig_balance.update_layout(
            template="plotly_dark",
            xaxis_title="Day", yaxis_title="Balance (L)",
            height=400, margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_balance, use_container_width=True)

    with tab_data:
        # ── Raw Data Tables ──────────────────────────────────────────────────────
        st.markdown("### Raw datasets")
        tab1, tab2, tab3 = st.tabs(["Usage Data", "Weather Data", "Balance Results"])
        with tab1:
            st.dataframe(usage_features.head(20), use_container_width=True)
        with tab2:
            st.dataframe(weather_rain.head(20), use_container_width=True)
        with tab3:
            balance_df = pd.DataFrame([vars(r) for r in balance_results[:20]])
            st.dataframe(balance_df, use_container_width=True)

else:
    st.info("👈 Configure settings in the sidebar and press **Run Simulation** to start.")
    st.markdown("""
    ### System Modules
    | # | Module | Description |
    |---|--------|-------------|
    | 1 | **Data Collection** | City-specific weather data & usage generation |
    | 2 | **Preprocessing** | Clean data, engineer features |
    | 3 | **Demand Prediction** | Gradient Boosting / Random Forest / Linear Regression |
    | 4 | **Rainwater Estimation** | Harvest calculation from rainfall |
    | 5 | **Water Balance** | Supply vs. demand analysis |
    | 6 | **Decision Engine** | Rule-based smart recommendations |
    | 7 | **Dashboard** | Interactive Streamlit visualisation |
    """)
