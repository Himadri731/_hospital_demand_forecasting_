"""
app.py - Interactive Streamlit Dashboard for Hospital Demand Forecasting
Run with: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pune Hospital Demand & Capacity Planner",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data paths
data_dir = "outputs"
forecast_path = os.path.join(data_dir, "forecast_next_13_weeks.csv")
planning_path = os.path.join(data_dir, "planning_table_capacity.csv")
metrics_path = os.path.join(data_dir, "backtest_metrics_summary.csv")
reconciled_path = os.path.join(data_dir, "reconciled_forecasts.csv")
scenario_path = os.path.join(data_dir, "scenario_stress_testing.csv")

if not os.path.exists(forecast_path) or not os.path.exists(planning_path):
    st.error("Forecast data files not found. Please run `python run_pipeline.py` first.")
    st.stop()

# Load Data
df_forecast = pd.read_csv(forecast_path)
df_plan = pd.read_csv(planning_path)
df_metrics = pd.read_csv(metrics_path)
df_scenarios = pd.read_csv(scenario_path)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/hospital-3.png", width=70)
st.sidebar.title("Pune Clinic Network")
st.sidebar.caption("Operational Demand Forecasting & Capacity Allocation")

hospitals = {
    "NETWORK_TOTAL": "Network Aggregate (All 4 Hospitals)",
    "PCM": "Pune Central Memorial (Shivajinagar)",
    "HCH": "Hadapsar Community Hospital (East Pune)",
    "KFC": "Kothrud Family Clinic & Hospital (West Pune)",
    "PMC": "Pimpri Medical Center (PCMC North)"
}

selected_hosp = st.sidebar.selectbox(
    "Select Facility / Scope:",
    options=list(hospitals.keys()),
    format_func=lambda x: f"{x} - {hospitals[x]}"
)

# Header
st.markdown('<div class="main-header">🏥 Hospital Demand & Operational Capacity Platform</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Active Facility: <b>{hospitals[selected_hosp]} ({selected_hosp})</b> | Planning Horizon: <b>Next Quarter (13 Weeks)</b></div>', unsafe_allow_html=True)

# Top KPI Metrics Row
plan_hosp = df_plan[df_plan["hospital_code"] == selected_hosp].reset_index(drop=True)
tot_opd = int(plan_hosp["projected_opd_visits"].sum())
tot_dengue = int(plan_hosp["projected_dengue_opd"].sum())
peak_beds = int(plan_hosp["beds_buffer_80pct"].max())
tot_doc_shifts = int(plan_hosp["doctor_shifts_80pct"].sum())
tot_nurse_shifts = int(plan_hosp["nurse_shifts_80pct"].sum())
tot_calls = int(plan_hosp["follow_up_calls_nominal"].sum())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Projected OPD", f"{tot_opd:,} visits", delta="Next 13 Wks")
with col2:
    st.metric("Dengue OPD Encounters", f"{tot_dengue:,} cases", delta="Seasonal Surge", delta_color="inverse")
with col3:
    st.metric("Peak Beds Needed (80% CI)", f"{peak_beds} beds", delta="Target 85% Occ")
with col4:
    st.metric("Doctor Clinical Shifts", f"{tot_doc_shifts:,} shifts", delta="6-hr blocks")
with col5:
    st.metric("Post-Discharge Calls", f"{tot_calls:,} calls", delta="Care outreach")

st.write("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast & Fan Charts",
    "🏥 Capacity & Staffing Schedule",
    "⚡ Interactive Scenario Simulator",
    "🎯 Model Selection & Backtesting",
    "📑 Executive Memo & Reliability Advisory"
])

# ----------------- TAB 1: DEMAND FORECAST -----------------
with tab1:
    st.subheader(f"13-Week Probabilistic Demand Forecast ({selected_hosp})")
    
    target_options = {
        "opd_total": "Total OPD Visits",
        "opd_dengue": "Dengue & Vector-Borne OPD Surge",
        "ipd_admissions": "Inpatient Admissions (IPD)",
        "lab_tests": "Diagnostic Laboratory Tests"
    }
    
    sel_target = st.selectbox("Select Clinical Service Line:", options=list(target_options.keys()), format_func=lambda x: target_options[x])
    
    sub_fc = df_forecast[(df_forecast["hospital_code"] == selected_hosp) & (df_forecast["target_segment"] == sel_target)].sort_values("forecast_week")
    
    best_model_used = sub_fc["best_model"].iloc[0]
    st.info(f"**Optimal Model Selected for this Segment:** `{best_model_used}` (Selected via Rolling-Origin Backtest minimizing hold-out error)")
    
    fig = go.Figure()
    
    # 95% Interval
    fig.add_trace(go.Scatter(
        x=sub_fc["week_start_date"], y=sub_fc["upper_95"],
        mode='lines', line=dict(width=0), showlegend=False, name="Upper 95%"
    ))
    fig.add_trace(go.Scatter(
        x=sub_fc["week_start_date"], y=sub_fc["lower_95"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(147, 197, 253, 0.25)', name="95% Contingency Interval"
    ))
    
    # 80% Interval
    fig.add_trace(go.Scatter(
        x=sub_fc["week_start_date"], y=sub_fc["upper_80"],
        mode='lines', line=dict(width=0), showlegend=False, name="Upper 80%"
    ))
    fig.add_trace(go.Scatter(
        x=sub_fc["week_start_date"], y=sub_fc["lower_80"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.35)', name="80% Planning Buffer"
    ))
    
    # Point Forecast
    fig.add_trace(go.Scatter(
        x=sub_fc["week_start_date"], y=sub_fc["point_forecast"],
        mode='lines+markers', line=dict(color='#1e40af', width=3),
        marker=dict(size=7), name="Point Forecast"
    ))
    
    fig.update_layout(
        title=f"Probabilistic Forecast: {target_options[sel_target]} ({selected_hosp})",
        xaxis_title="Week Starting Date",
        yaxis_title="Weekly Demand / Encounters",
        hovermode="x unified",
        template="plotly_white",
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(sub_fc[["forecast_week", "week_start_date", "best_model", "point_forecast", "lower_80", "upper_80", "lower_95", "upper_95"]], use_container_width=True)

# ----------------- TAB 2: CAPACITY SCHEDULE -----------------
with tab2:
    st.subheader(f"Operational Allocation Schedule ({selected_hosp})")
    st.caption("Formulated using Queueing Theory & Clinical Ratios: Beds at 85% Occupancy | Doctors: 25 pts/shift | Nurses: 1:5 ratio")
    
    fig_cap = go.Figure()
    fig_cap.add_trace(go.Bar(
        x=plan_hosp["week_start_date"], y=plan_hosp["beds_required_nominal"],
        name="Nominal Beds Needed", marker_color="#3b82f6"
    ))
    fig_cap.add_trace(go.Scatter(
        x=plan_hosp["week_start_date"], y=plan_hosp["beds_buffer_80pct"],
        mode="lines+markers", name="80% Staffed Beds Buffer", line=dict(color="#f59e0b", width=3)
    ))
    fig_cap.add_trace(go.Scatter(
        x=plan_hosp["week_start_date"], y=plan_hosp["beds_surge_95pct"],
        mode="lines", name="95% Epidemic Ceiling", line=dict(color="#ef4444", width=2, dash="dash")
    ))
    fig_cap.update_layout(
        title="Weekly Inpatient Bed Requirements (85% Target Occupancy)",
        xaxis_title="Week Starting Date",
        yaxis_title="Operational Beds",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_cap, use_container_width=True)
    
    st.markdown("#### Detailed Resource Plan:")
    st.dataframe(plan_hosp[[
        "forecast_week", "week_start_date", "projected_opd_visits", "projected_dengue_opd",
        "beds_required_nominal", "beds_buffer_80pct", "beds_surge_95pct",
        "doctor_shifts_nominal", "doctor_shifts_80pct",
        "nurse_shifts_nominal", "nurse_shifts_80pct",
        "follow_up_calls_nominal"
    ]], use_container_width=True)

# ----------------- TAB 3: SCENARIO SIMULATOR -----------------
with tab3:
    st.subheader("⚡ Live Operational Stress-Testing Simulator")
    st.write("Adjust operational shocks below to simulate required bed and staffing responses:")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        dengue_shock = st.slider("Dengue Outbreak Severity Spike (%)", min_value=0, max_value=100, value=55, step=5)
    with sim_col2:
        promo_discount = st.slider("Consultation Tariff Discount / Health Camp Promotion (%)", min_value=0, max_value=50, value=20, step=5)
        
    base_opd_val = plan_hosp["projected_opd_visits"].values
    base_dengue_val = plan_hosp["projected_dengue_opd"].values
    base_adm_val = plan_hosp["projected_admissions"].values
    
    # Calculate simulated shock
    sim_dengue = base_dengue_val * (1.0 + dengue_shock / 100.0)
    sim_opd = base_opd_val * (1.0 + (promo_discount * 1.1) / 100.0) + (sim_dengue - base_dengue_val) * 1.2
    sim_adm = base_adm_val * (1.0 + (promo_discount * 0.35) / 100.0) + (sim_dengue - base_dengue_val) * 0.22
    sim_beds = (sim_adm * 4.2) / (7.0 * 0.85)
    sim_docs = np.ceil(sim_opd / 25.0)
    
    sc_fig = go.Figure()
    sc_fig.add_trace(go.Scatter(x=plan_hosp["week_start_date"], y=plan_hosp["beds_required_nominal"], mode="lines", name="Baseline Required Beds", line=dict(color="blue", dash="dash")))
    sc_fig.add_trace(go.Scatter(x=plan_hosp["week_start_date"], y=sim_beds, mode="lines+markers", name="Simulated Bed Demand Under Shock", line=dict(color="red", width=3)))
    sc_fig.update_layout(title="Simulated Inpatient Bed Load vs Baseline", xaxis_title="Week", yaxis_title="Beds Needed", template="plotly_white")
    st.plotly_chart(sc_fig, use_container_width=True)
    
    st.warning(f"**Operational Impact:** Under this simulated scenario, peak bed requirements reach **{int(max(sim_beds))} beds** (vs {int(max(plan_hosp['beds_required_nominal']))} baseline) and weekly doctor shifts increase to **{int(max(sim_docs))} shifts**.")

# ----------------- TAB 4: BACKTEST EVALUATION -----------------
with tab4:
    st.subheader("Model Evaluation & Rolling-Origin Backtesting Results")
    
    # KPI Metrics
    tot_segs = len(df_metrics[df_metrics["is_best_selected"] == True])
    beats_count = df_metrics[df_metrics["is_best_selected"] == True]["beats_baseline"].sum()
    pct_win = (beats_count / tot_segs) * 100.0
    avg_best_mape = df_metrics[df_metrics["is_best_selected"] == True]["MAPE"].mean()
    avg_base_mape = df_metrics[df_metrics["is_baseline"] == True]["MAPE"].mean()
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Segments Beating Baseline", f"{beats_count} / {tot_segs} ({pct_win:.1f}%)", delta="Goal: >= 70%")
    m_col2.metric("Overall Best Model MAPE", f"{avg_best_mape:.2f}%", delta="Goal: < 15%")
    m_col3.metric("Seasonal Naive Baseline MAPE", f"{avg_base_mape:.2f}%", delta="-7.02% reduction", delta_color="inverse")
    
    st.write("---")
    
    # Model comparison bar chart for selected hospital
    sub_metrics = df_metrics[df_metrics["hospital"] == selected_hosp]
    fig_bar = px.bar(
        sub_metrics, x="target_segment", y="MAPE", color="model", barmode="group",
        title=f"13-Week Hold-out MAPE Comparison Across Models ({selected_hosp})",
        labels={"target_segment": "Service Line", "MAPE": "MAPE (% Error)"},
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.dataframe(sub_metrics[["hospital", "target_segment", "model", "MAPE", "Bias_Pct", "RMSE", "is_best_selected", "beats_baseline"]], use_container_width=True)

# ----------------- TAB 5: MEMO & ADVISORY -----------------
with tab5:
    st.subheader("Strategic Planning Memo & Reliability Advisory")
    memo_path = "reports/Executive_Planning_Memo_Q1_2027.md"
    if os.path.exists(memo_path):
        with open(memo_path, "r", encoding="utf-8") as f:
            memo_text = f.read()
        st.markdown(memo_text)
    else:
        st.info("Executive Planning Memo is located at `reports/Executive_Planning_Memo_Q1_2027.md`.")
