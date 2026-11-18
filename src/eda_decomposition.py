"""
eda_decomposition.py - Time-series decomposition, festival impact analysis,
and exploratory visualizations for Pune community clinic network.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

def run_eda_and_decomposition(data_path="data/processed/weekly_aggregated_segments.csv", output_fig_dir="outputs/figures"):
    os.makedirs(output_fig_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    
    # Filter for Network Total
    df_net = df[df["hospital_code"] == "NETWORK_TOTAL"].sort_values("week_start").reset_index(drop=True)
    
    print("Generating EDA Visualizations and Time Series Decomposition...")
    
    # 1. Multi-Hospital Overview
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    hospitals = ["PCM", "HCH", "KFC", "PMC"]
    colors = {"PCM": "#1f77b4", "HCH": "#ff7f0e", "KFC": "#2ca02c", "PMC": "#d62728"}
    
    # (a) OPD Total
    for hosp in hospitals:
        sub = df[df["hospital_code"] == hosp]
        axes[0].plot(sub["week_start"], sub["opd_total"], label=f"{hosp}", color=colors[hosp], linewidth=1.8)
    axes[0].set_title("Weekly OPD Visits Across 4 Pune Hospitals (2023 - 2026)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Weekly OPD Visits", fontsize=11)
    axes[0].legend(loc="upper left", frameon=True)
    
    # (b) Inpatient Admissions
    for hosp in hospitals:
        sub = df[df["hospital_code"] == hosp]
        axes[1].plot(sub["week_start"], sub["ipd_admissions"], label=f"{hosp}", color=colors[hosp], linewidth=1.8)
    axes[1].set_title("Weekly Inpatient Admissions (IPD)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Weekly Admissions", fontsize=11)
    
    # (c) Dengue / Vector-Borne OPD Cases (Surge segment)
    for hosp in hospitals:
        sub = df[df["hospital_code"] == hosp]
        axes[2].plot(sub["week_start"], sub["opd_dengue"], label=f"{hosp}", color=colors[hosp], linewidth=1.8)
    axes[2].set_title("Dengue & Vector-Borne OPD Surge (Monsoon Episodes)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Dengue OPD Visits", fontsize=11)
    axes[2].set_xlabel("Week Start Date", fontsize=11)
    
    # Add monsoon shaded bands (July - Oct each year)
    for yr in [2024, 2025, 2026]:
        axes[2].axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-10-25"), alpha=0.15, color="orange", label="Monsoon Surge Window" if yr == 2024 else None)
    axes[2].legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    fig_path1 = os.path.join(output_fig_dir, "eda_time_series_overview.png")
    fig.savefig(fig_path1)
    plt.close()
    print(f"Saved: {fig_path1}")
    
    # 2. Time Series Decomposition (Classical Additive / Multiplicative with 52-week period)
    # Using Network Total OPD
    ts_opd = df_net.set_index("week_start")["opd_total"]
    decomp = seasonal_decompose(ts_opd, model="additive", period=52)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="#2b5c8f", lw=2)
    axes[0].set_title("Network OPD Visits: Observed Data", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Observed")
    
    decomp.trend.plot(ax=axes[1], color="#e6550d", lw=2)
    axes[1].set_title("Underlying Trend (Steady ~8% Annual Expansion)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Trend")
    
    decomp.seasonal.plot(ax=axes[2], color="#31a354", lw=2)
    axes[2].set_title("52-Week Annual Seasonality (Monsoon Surges & Festival Lulls)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Seasonality")
    
    decomp.resid.plot(ax=axes[3], color="#756bb1", marker="o", markersize=3, linestyle="none")
    axes[3].axhline(0, color="gray", linestyle="--")
    axes[3].set_title("Irregular Residual / Noise Component", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date", fontsize=11)
    
    plt.tight_layout()
    fig_path2 = os.path.join(output_fig_dir, "eda_decomposition_stl.png")
    fig.savefig(fig_path2)
    plt.close()
    print(f"Saved: {fig_path2}")
    
    # 3. Dengue Surge vs Monsoon Intensity & Lab Platelet Demand
    fig, ax1 = plt.subplots(figsize=(13, 6))
    color1 = "#d95f02"
    color2 = "#7570b3"
    
    ax1.plot(df_net["week_start"], df_net["opd_dengue"], color=color1, lw=2.2, label="Dengue OPD Cases")
    ax1.plot(df_net["week_start"], df_net["lab_dengue_platelet"], color="#e7298a", lw=2.0, linestyle="--", label="Platelet & Dengue Lab Orders")
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Dengue Clinical Encounters & Lab Orders", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    ax2.plot(df_net["week_start"], df_net["monsoon_rain_index"], color=color2, lw=1.8, linestyle=":", label="Monsoon Rain & Vector Index")
    ax2.set_ylabel("Monsoon Rain & Vector Index (0 - 1)", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    plt.title("Pune Dengue Season Dynamics: Vector Index Leading Clinical & Platelet Test Demand", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    fig_path3 = os.path.join(output_fig_dir, "eda_dengue_surge_vs_monsoon.png")
    fig.savefig(fig_path3)
    plt.close()
    print(f"Saved: {fig_path3}")
    
    # 4. Festival Impact Analysis
    # Classify each week: Normal, Diwali Week, Post-Diwali Surge, Ganesh Chaturthi, Holi
    week_types = []
    for _, row in df_net.iterrows():
        if row["fest_diwali"] == 1:
            week_types.append("Diwali Week")
        elif row["fest_diwali_post_surge"] == 1:
            week_types.append("Post-Diwali Surge")
        elif row["fest_ganesh_chaturthi"] == 1:
            week_types.append("Ganesh Chaturthi")
        elif row["fest_holi"] == 1:
            week_types.append("Holi Week")
        elif row["monsoon_rain_index"] > 0.4:
            week_types.append("Peak Monsoon")
        else:
            week_types.append("Normal Non-Peak")
            
    df_net["event_cohort"] = week_types
    
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Normal Non-Peak", "Ganesh Chaturthi", "Diwali Week", "Post-Diwali Surge", "Holi Week", "Peak Monsoon"]
    sns.boxplot(data=df_net, x="event_cohort", y="opd_total", order=order, hue="event_cohort", palette="Set2", legend=False, ax=ax)
    sns.stripplot(data=df_net, x="event_cohort", y="opd_total", order=order, color="black", alpha=0.6, jitter=0.2, ax=ax)
    ax.set_title("Impact of Pune Festivals & Monsoon on Network OPD Volumes", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Weekly OPD Visits", fontsize=11)
    ax.set_xlabel("Calendar Event Cohort", fontsize=11)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    fig_path4 = os.path.join(output_fig_dir, "eda_festival_impact_boxplots.png")
    fig.savefig(fig_path4)
    plt.close()
    print(f"Saved: {fig_path4}")
    
    print("EDA and decomposition complete!")
    return decomp

if __name__ == "__main__":
    run_eda_and_decomposition()
