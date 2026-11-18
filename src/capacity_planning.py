"""
capacity_planning.py - Converts point forecasts and 80%/95% prediction intervals
into operational hospital planning tables for:
1. Inpatient Bed Requirements (at 85% target occupancy)
2. Outpatient Doctor Shifts (25 patients / 6-hour shift)
3. Inpatient Nursing Shifts (1:5 general ward ratio, 3 shifts/day)
4. Outpatient Chronic & Post-Discharge Follow-Up Care Coordination Calls
5. Laboratory Phlebotomy & Technician Staffing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_forecast_and_planning_tables(
    data_path="data/processed/weekly_aggregated_segments.csv",
    calendar_path="data/processed/pune_calendar_features.csv",
    metrics_path="outputs/backtest_metrics_summary.csv",
    output_dir="outputs",
    horizon=13
):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    cal_df = pd.read_csv(calendar_path)
    cal_df["week_start"] = pd.to_datetime(cal_df["week_start"])
    
    metrics_df = pd.read_csv(metrics_path)
    
    # Identify future 13 weeks (after the last historical week in df)
    last_hist_date = df["week_start"].max()
    future_cal = cal_df[cal_df["week_start"] > last_hist_date].sort_values("week_start").head(horizon).reset_index(drop=True)
    
    if len(future_cal) < horizon:
        # Create future weeks if not enough in calendar
        future_dates = [last_hist_date + pd.Timedelta(weeks=w) for w in range(1, horizon + 1)]
        future_cal = pd.DataFrame({"week_start": future_dates})
        future_cal["week_of_year"] = future_cal["week_start"].dt.isocalendar().week.astype(int)
        future_cal["monsoon_rain_index"] = 0.05
        future_cal["fest_diwali"] = 0
        future_cal["fest_diwali_post_surge"] = 0
        future_cal["fest_ganesh_chaturthi"] = 0
        future_cal["fest_ganesh_post_surge"] = 0
        future_cal["fest_holi"] = 0
        
    hospitals = ["PCM", "HCH", "KFC", "PMC", "NETWORK_TOTAL"]
    targets = ["opd_total", "opd_dengue", "ipd_admissions", "lab_tests"]
    
    forecast_rows = []
    
    from src.models import SeasonalNaiveModel, ETSModel, ProphetModel, LightGBMTimeSeriesModel
    
    print("\nGenerating Production Forecasts (Point, 80% & 95% Intervals) for Next Quarter (13 Weeks)...")
    
    for hosp in hospitals:
        sub_df = df[df["hospital_code"] == hosp].sort_values("week_start").reset_index(drop=True)
        
        for tgt in targets:
            # Lookup best model from backtest metrics
            seg_metrics = metrics_df[(metrics_df["hospital"] == hosp) & (metrics_df["target_segment"] == tgt)]
            if len(seg_metrics) > 0 and (seg_metrics["is_best_selected"] == True).any():
                best_model_name = seg_metrics[seg_metrics["is_best_selected"] == True]["model"].values[0]
            else:
                best_model_name = "LightGBM"
                
            # Fit best model on 100% of available historical data
            if best_model_name == "Seasonal_Naive":
                model = SeasonalNaiveModel(season_length=52).fit(sub_df[tgt].values)
                preds = model.predict(horizon=horizon)
            elif best_model_name == "ETS":
                model = ETSModel(seasonal_periods=52).fit(sub_df[tgt].values)
                preds = model.predict(horizon=horizon)
            elif best_model_name == "Prophet":
                model = ProphetModel().fit(sub_df, target_col=tgt)
                preds = model.predict(future_cal)
            else: # LightGBM
                model = LightGBMTimeSeriesModel(lags=[1, 2, 3, 4, 13, 52]).fit(sub_df, target_col=tgt)
                preds = model.predict(future_cal, horizon=horizon)
                
            for h in range(horizon):
                wk_date = future_cal.iloc[h]["week_start"].strftime("%Y-%m-%d")
                forecast_rows.append({
                    "hospital_code": hosp,
                    "target_segment": tgt,
                    "forecast_week": h + 1,
                    "week_start_date": wk_date,
                    "best_model": best_model_name,
                    "point_forecast": round(max(0, float(preds["point"][h])), 1),
                    "lower_80": round(max(0, float(preds["lower_80"][h])), 1),
                    "upper_80": round(max(0, float(preds["upper_80"][h])), 1),
                    "lower_95": round(max(0, float(preds["lower_95"][h])), 1),
                    "upper_95": round(max(0, float(preds["upper_95"][h])), 1)
                })
                
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_csv = os.path.join(output_dir, "forecast_next_13_weeks.csv")
    forecast_df.to_csv(forecast_csv, index=False)
    print(f"Saved 13-week segment forecasts to: {forecast_csv}")
    
    # 2. CONVERT TO OPERATIONAL PLANNING TABLE
    # Parameters for clinical conversion:
    # Average Length of Stay (LoS) in days:
    los_lookup = {"PCM": 4.2, "HCH": 3.6, "KFC": 3.4, "PMC": 3.8, "NETWORK_TOTAL": 3.8}
    target_occupancy = 0.85
    pts_per_doctor_shift = 25.0 # 6-hour outpatient clinic shift
    bed_nurse_ratio = 5.0 # 1 nurse per 5 beds
    tests_per_lab_shift = 45.0 # tests per technician shift
    
    planning_rows = []
    
    # Pivot forecasts by hospital and week
    for hosp in hospitals:
        hosp_los = los_lookup.get(hosp, 3.8)
        
        for w in range(1, horizon + 1):
            sub_w = forecast_df[(forecast_df["hospital_code"] == hosp) & (forecast_df["forecast_week"] == w)]
            wk_date = sub_w["week_start_date"].values[0]
            
            # OPD total
            opd_pt = sub_w[sub_w["target_segment"] == "opd_total"]["point_forecast"].values[0]
            opd_u80 = sub_w[sub_w["target_segment"] == "opd_total"]["upper_80"].values[0]
            opd_u95 = sub_w[sub_w["target_segment"] == "opd_total"]["upper_95"].values[0]
            
            # Dengue OPD
            dengue_pt = sub_w[sub_w["target_segment"] == "opd_dengue"]["point_forecast"].values[0]
            
            # Admissions
            adm_pt = sub_w[sub_w["target_segment"] == "ipd_admissions"]["point_forecast"].values[0]
            adm_u80 = sub_w[sub_w["target_segment"] == "ipd_admissions"]["upper_80"].values[0]
            adm_u95 = sub_w[sub_w["target_segment"] == "ipd_admissions"]["upper_95"].values[0]
            
            # Lab tests
            lab_pt = sub_w[sub_w["target_segment"] == "lab_tests"]["point_forecast"].values[0]
            lab_u80 = sub_w[sub_w["target_segment"] == "lab_tests"]["upper_80"].values[0]
            lab_u95 = sub_w[sub_w["target_segment"] == "lab_tests"]["upper_95"].values[0]
            
            # --- Operational Conversions ---
            # 1. Beds Required (Formula: Weekly Adm * LoS / (7 * Occupancy))
            beds_nominal = round((adm_pt * hosp_los) / (7.0 * target_occupancy), 1)
            beds_buffer_80 = round((adm_u80 * hosp_los) / (7.0 * target_occupancy), 1)
            beds_surge_95 = round((adm_u95 * hosp_los) / (7.0 * target_occupancy), 1)
            
            # 2. Doctor Shifts (6-hr shifts / week)
            doc_shifts_nominal = int(np.ceil(opd_pt / pts_per_doctor_shift))
            doc_shifts_buffer_80 = int(np.ceil(opd_u80 / pts_per_doctor_shift))
            doc_shifts_surge_95 = int(np.ceil(opd_u95 / pts_per_doctor_shift))
            
            # 3. Nurse Shifts (Bed-days * 3 shifts/day / 5)
            # Bed days in week = Admissions * LoS
            nurse_shifts_nominal = int(np.ceil((adm_pt * hosp_los * 3.0) / bed_nurse_ratio))
            nurse_shifts_buffer_80 = int(np.ceil((adm_u80 * hosp_los * 3.0) / bed_nurse_ratio))
            nurse_shifts_surge_95 = int(np.ceil((adm_u95 * hosp_los * 3.0) / bed_nurse_ratio))
            
            # 4. Lab Tech Shifts
            lab_shifts_nominal = int(np.ceil(lab_pt / tests_per_lab_shift))
            lab_shifts_buffer_80 = int(np.ceil(lab_u80 / tests_per_lab_shift))
            lab_shifts_surge_95 = int(np.ceil(lab_u95 / tests_per_lab_shift))
            
            # 5. Follow-Up Calls
            # 80% of inpatient discharges within 72h + 20% of chronic outpatient cohort
            # (Chronic represents ~25% of OPD)
            calls_nominal = int(np.ceil(0.80 * adm_pt + 0.20 * (opd_pt * 0.25)))
            calls_buffer_80 = int(np.ceil(0.80 * adm_u80 + 0.20 * (opd_u80 * 0.25)))
            calls_surge_95 = int(np.ceil(0.80 * adm_u95 + 0.20 * (opd_u95 * 0.25)))
            
            planning_rows.append({
                "hospital_code": hosp,
                "forecast_week": w,
                "week_start_date": wk_date,
                "projected_opd_visits": opd_pt,
                "projected_dengue_opd": dengue_pt,
                "projected_admissions": adm_pt,
                "projected_lab_tests": lab_pt,
                "beds_required_nominal": beds_nominal,
                "beds_buffer_80pct": beds_buffer_80,
                "beds_surge_95pct": beds_surge_95,
                "doctor_shifts_nominal": doc_shifts_nominal,
                "doctor_shifts_80pct": doc_shifts_buffer_80,
                "doctor_shifts_95pct": doc_shifts_surge_95,
                "nurse_shifts_nominal": nurse_shifts_nominal,
                "nurse_shifts_80pct": nurse_shifts_buffer_80,
                "nurse_shifts_95pct": nurse_shifts_surge_95,
                "lab_tech_shifts_nominal": lab_shifts_nominal,
                "lab_tech_shifts_80pct": lab_shifts_buffer_80,
                "follow_up_calls_nominal": calls_nominal,
                "follow_up_calls_80pct": calls_buffer_80,
                "follow_up_calls_95pct": calls_surge_95
            })
            
    planning_df = pd.DataFrame(planning_rows)
    planning_csv = os.path.join(output_dir, "planning_table_capacity.csv")
    planning_df.to_csv(planning_csv, index=False)
    print(f"Saved capacity planning table to: {planning_csv}")
    
    # Fan chart visualization for Network Total Capacity Planning
    net_plan = planning_df[planning_df["hospital_code"] == "NETWORK_TOTAL"].reset_index(drop=True)
    w_dates = pd.to_datetime(net_plan["week_start_date"])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Beds Plan
    axes[0, 0].plot(w_dates, net_plan["beds_required_nominal"], "b-o", lw=2, label="Nominal Beds Needed")
    axes[0, 0].fill_between(w_dates, net_plan["beds_required_nominal"], net_plan["beds_buffer_80pct"],
                            color="#6baed6", alpha=0.4, label="80% Planning Buffer")
    axes[0, 0].fill_between(w_dates, net_plan["beds_buffer_80pct"], net_plan["beds_surge_95pct"],
                            color="#fc9272", alpha=0.35, label="95% Epidemic Surge Buffer")
    axes[0, 0].set_title("Network Inpatient Bed Allocation (85% Target Occupancy)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Required Operational Beds")
    axes[0, 0].legend(loc="upper left", frameon=True)
    
    # 2. Doctor Shifts
    axes[0, 1].plot(w_dates, net_plan["doctor_shifts_nominal"], "g-s", lw=2, label="Nominal Doctor Shifts")
    axes[0, 1].fill_between(w_dates, net_plan["doctor_shifts_nominal"], net_plan["doctor_shifts_80pct"],
                            color="#a1d99b", alpha=0.5, label="80% Staffing Buffer")
    axes[0, 1].set_title("Weekly OPD Doctor Shifts (6-Hour Clinical Blocks)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Doctor Shifts / Week")
    axes[0, 1].legend(loc="upper left", frameon=True)
    
    # 3. Nurse Shifts
    axes[1, 0].plot(w_dates, net_plan["nurse_shifts_nominal"], "m-^", lw=2, label="Nominal Nurse Shifts")
    axes[1, 0].fill_between(w_dates, net_plan["nurse_shifts_nominal"], net_plan["nurse_shifts_80pct"],
                            color="#bcbddc", alpha=0.5, label="80% Nursing Buffer")
    axes[1, 0].set_title("Weekly Inpatient Nurse Shifts (1:5 General Ward Ratio)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Nurse Shifts / Week")
    axes[1, 0].set_xlabel("Week Starting Date")
    axes[1, 0].legend(loc="upper left", frameon=True)
    
    # 4. Follow-Up Calls
    axes[1, 1].plot(w_dates, net_plan["follow_up_calls_nominal"], "c-d", lw=2, label="Nominal Follow-Up Calls")
    axes[1, 1].fill_between(w_dates, net_plan["follow_up_calls_nominal"], net_plan["follow_up_calls_80pct"],
                            color="#9ecae1", alpha=0.5, label="80% Care Coordination Buffer")
    axes[1, 1].set_title("Weekly Telehealth & Post-Discharge Follow-Up Call Volume", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Follow-Up Patient Calls / Week")
    axes[1, 1].set_xlabel("Week Starting Date")
    axes[1, 1].legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "capacity_planning_schedule_fans.png")
    fig.savefig(fig_path)
    plt.close()
    print(f"Saved capacity planning figure: {fig_path}")
    
    return forecast_df, planning_df
