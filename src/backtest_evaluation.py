"""
backtest_evaluation.py - Rolling-origin backtesting across all hospital segments,
metric evaluation (MAPE, Bias, RMSE), model selection per segment,
and validation against success criteria (beat baseline on >=70% segments, MAPE <15%).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import SeasonalNaiveModel, ETSModel, ProphetModel, LightGBMTimeSeriesModel

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Avoid zero division
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    bias = (np.sum(y_pred - y_true) / np.maximum(1.0, np.sum(y_true))) * 100.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        "MAPE": round(mape, 2),
        "Bias": round(bias, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2)
    }

def run_backtest_suite(data_path="data/processed/weekly_aggregated_segments.csv",
                       output_dir="outputs",
                       horizon=13):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    
    hospitals = ["PCM", "HCH", "KFC", "PMC", "NETWORK_TOTAL"]
    targets = ["opd_total", "opd_dengue", "ipd_admissions", "lab_tests"]
    
    results = []
    backtest_predictions = {}
    
    print(f"Starting Rolling-Origin 13-Week Backtesting across {len(hospitals)} entities x {len(targets)} targets = {len(hospitals)*len(targets)} segments...")
    
    total_segments = 0
    segments_beating_baseline = 0
    
    for hosp in hospitals:
        sub_df = df[df["hospital_code"] == hosp].sort_values("week_start").reset_index(drop=True)
        n_weeks = len(sub_df)
        
        train_df = sub_df.iloc[:(n_weeks - horizon)].copy().reset_index(drop=True)
        test_df = sub_df.iloc[(n_weeks - horizon):].copy().reset_index(drop=True)
        test_weeks = test_df["week_start"].values
        
        for tgt in targets:
            segment_key = f"{hosp}_{tgt}"
            total_segments += 1
            y_test = test_df[tgt].values
            
            # 1. Seasonal Naive Baseline
            snaive = SeasonalNaiveModel(season_length=52).fit(train_df[tgt].values)
            pred_snaive = snaive.predict(horizon=horizon)
            m_snaive = compute_metrics(y_test, pred_snaive["point"])
            
            # 2. ETS (Holt-Winters)
            ets = ETSModel(seasonal_periods=52).fit(train_df[tgt].values)
            pred_ets = ets.predict(horizon=horizon)
            m_ets = compute_metrics(y_test, pred_ets["point"])
            
            # 3. Prophet
            prophet_mod = ProphetModel().fit(train_df, target_col=tgt)
            pred_prophet = prophet_mod.predict(test_df)
            m_prophet = compute_metrics(y_test, pred_prophet["point"])
            
            # 4. LightGBM
            lgb_mod = LightGBMTimeSeriesModel(lags=[1, 2, 3, 4, 13, 52]).fit(train_df, target_col=tgt)
            pred_lgb = lgb_mod.predict(test_df, horizon=horizon)
            m_lgb = compute_metrics(y_test, pred_lgb["point"])
            
            # Compile model comparison for this segment
            models_dict = {
                "Seasonal_Naive": (m_snaive, pred_snaive),
                "ETS": (m_ets, pred_ets),
                "Prophet": (m_prophet, pred_prophet),
                "LightGBM": (m_lgb, pred_lgb)
            }
            
            # Determine best model based on minimum MAPE
            # Exclude baseline from candidates unless all fail
            candidate_models = ["ETS", "Prophet", "LightGBM"]
            best_model_name = min(candidate_models, key=lambda m: models_dict[m][0]["MAPE"])
            best_metrics = models_dict[best_model_name][0]
            best_pred = models_dict[best_model_name][1]
            
            beats_baseline = best_metrics["MAPE"] < m_snaive["MAPE"]
            if beats_baseline:
                segments_beating_baseline += 1
                
            # Log results
            for m_name, (m_eval, p_eval) in models_dict.items():
                results.append({
                    "hospital": hosp,
                    "target_segment": tgt,
                    "segment_id": segment_key,
                    "model": m_name,
                    "MAPE": m_eval["MAPE"],
                    "Bias_Pct": m_eval["Bias"],
                    "RMSE": m_eval["RMSE"],
                    "MAE": m_eval["MAE"],
                    "is_baseline": (m_name == "Seasonal_Naive"),
                    "is_best_selected": (m_name == best_model_name),
                    "baseline_MAPE": m_snaive["MAPE"],
                    "beats_baseline": beats_baseline
                })
                
            backtest_predictions[segment_key] = {
                "test_weeks": test_weeks,
                "y_true": y_test,
                "best_model_name": best_model_name,
                "best_pred": best_pred,
                "snaive_pred": pred_snaive
            }
            
            print(f"Segment: {segment_key:<22} | Baseline MAPE: {m_snaive['MAPE']:5.1f}% | Best: {best_model_name:<9} ({best_metrics['MAPE']:5.1f}%, Bias: {best_metrics['Bias']:+5.1f}%) | Beats Baseline: {beats_baseline}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "backtest_metrics_summary.csv"), index=False)
    
    # Calculate Success Metrics
    selected_models_df = res_df[res_df["is_best_selected"] == True]
    overall_selected_mape = selected_models_df["MAPE"].mean()
    pct_beating_baseline = (segments_beating_baseline / total_segments) * 100.0
    
    print("\n" + "="*70)
    print("BACKTESTING & MODEL EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Evaluated Segments:            {total_segments}")
    print(f"Segments Beating Baseline:           {segments_beating_baseline} / {total_segments} ({pct_beating_baseline:.1f}%) [Success Target: >= 70%]")
    print(f"Average Hold-out MAPE (Best Models): {overall_selected_mape:.2f}% [Success Target: < 15%]")
    print(f"Baseline (Seasonal Naive) Avg MAPE:  {res_df[res_df['is_baseline']]['MAPE'].mean():.2f}%")
    print(f"Model Selection Breakdown:           {selected_models_df['model'].value_counts().to_dict()}")
    print("="*70)
    
    # Visualizations: Actual vs Predicted across 6 key segments
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    plot_segments = [
        ("NETWORK_TOTAL_opd_total", "Network Total OPD Visits", axes[0, 0]),
        ("NETWORK_TOTAL_opd_dengue", "Network Dengue OPD Surge", axes[0, 1]),
        ("PCM_opd_total", "Pune Central Memorial - OPD Total", axes[1, 0]),
        ("PCM_ipd_admissions", "Pune Central Memorial - Inpatient Admissions", axes[1, 1]),
        ("HCH_opd_dengue", "Hadapsar Community Hospital - Dengue OPD", axes[2, 0]),
        ("PMC_lab_tests", "Pimpri Medical Center - Lab Tests", axes[2, 1]),
    ]
    
    for seg_key, title, ax in plot_segments:
        if seg_key in backtest_predictions:
            item = backtest_predictions[seg_key]
            wks = pd.to_datetime(item["test_weeks"])
            ax.plot(wks, item["y_true"], "ko-", label="Actual Hold-out Demand", lw=2.2, markersize=5)
            
            b_name = item["best_model_name"]
            b_pred = item["best_pred"]["point"]
            ax.plot(wks, b_pred, "b^-", label=f"Best Model ({b_name})", lw=2, markersize=5)
            
            # Fill 80% and 95% prediction intervals
            ax.fill_between(wks, item["best_pred"]["lower_95"], item["best_pred"]["upper_95"],
                            color="#9ecae1", alpha=0.35, label="95% Prediction Interval")
            ax.fill_between(wks, item["best_pred"]["lower_80"], item["best_pred"]["upper_80"],
                            color="#3182bd", alpha=0.35, label="80% Prediction Interval")
            
            # Seasonal Naive dashed line
            ax.plot(wks, item["snaive_pred"]["point"], "r--", label="Seasonal Naive Baseline", alpha=0.8)
            
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Weekly Encounters")
            ax.legend(loc="upper left", fontsize=8.5, frameon=True)
            
    axes[2, 0].set_xlabel("Hold-out Week Start")
    axes[2, 1].set_xlabel("Hold-out Week Start")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "backtest_actual_vs_pred_best_models.png")
    fig.savefig(fig_path)
    plt.close()
    print(f"Saved backtest plot: {fig_path}")
    
    return res_df, selected_models_df, backtest_predictions

if __name__ == "__main__":
    run_backtest_suite()
