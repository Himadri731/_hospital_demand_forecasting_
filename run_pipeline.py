"""
run_pipeline.py - Master execution script for Pune Hospital Demand Forecasting.
Executes the end-to-end pipeline:
1. Data generation (~500,000 realistic clinical records across 4 hospitals)
2. Preprocessing & Clinical NLP taxonomy classification
3. Time series decomposition & EDA visualizations
4. Rolling-origin backtesting & per-segment model selection
5. 13-Week production forecasting with 80% & 95% prediction intervals
6. Operational capacity planning conversion (beds, doctors, nurses, follow-up calls)
7. Hierarchical forecast reconciliation (MinT / OLS & Bottom-Up)
8. Scenario stress testing (Dengue shock vs Tariff promotion)
"""

import os
import sys
import time

def main():
    start_time = time.time()
    print("="*80)
    print("PUNE COMMUNITY CLINIC NETWORK - HOSPITAL DEMAND FORECASTING PIPELINE")
    print("="*80)
    
    # 1. Data Generation
    print("\n[Step 1/7] Generating ~500,000 raw clinical records across 4 Pune hospitals...")
    from src.data_generator import generate_pune_hospital_data
    generate_pune_hospital_data(output_dir="data/raw", seed=42)
    
    # 2. Preprocessing & Aggregation
    print("\n[Step 2/7] Preprocessing: Age parsing, Clinical NLP diagnosis categorization, discharge imputation, weekly aggregation...")
    from src.preprocessing import process_and_aggregate_weekly
    process_and_aggregate_weekly(data_dir="data/raw", output_dir="data/processed")
    
    # 3. EDA & Time Series Decomposition
    print("\n[Step 3/7] Performing STL Time Series Decomposition & Pune Festival impact analysis...")
    from src.eda_decomposition import run_eda_and_decomposition
    run_eda_and_decomposition(data_path="data/processed/weekly_aggregated_segments.csv", output_fig_dir="outputs/figures")
    
    # 4. Rolling-Origin Backtesting & Model Selection
    print("\n[Step 4/7] Running 13-week Rolling-Origin Backtesting across all 20 segments...")
    from src.backtest_evaluation import run_backtest_suite
    metrics_df, selected_models_df, backtest_preds = run_backtest_suite(
        data_path="data/processed/weekly_aggregated_segments.csv",
        output_dir="outputs",
        horizon=13
    )
    
    # 5. Production Forecasting & Operational Capacity Planning
    print("\n[Step 5/7] Generating Next Quarter (13-Week) Forecast & Operational Capacity Planning Table...")
    from src.capacity_planning import generate_forecast_and_planning_tables
    forecast_df, planning_df = generate_forecast_and_planning_tables(
        data_path="data/processed/weekly_aggregated_segments.csv",
        calendar_path="data/processed/pune_calendar_features.csv",
        metrics_path="outputs/backtest_metrics_summary.csv",
        output_dir="outputs",
        horizon=13
    )
    
    # 6. Hierarchical Forecast Reconciliation
    print("\n[Step 6/7] Running MinT / OLS Hierarchical Reconciliation...")
    from src.hierarchical_reconciliation import reconcile_hospital_hierarchy
    # Build dictionary of forecasts
    forecast_dict = {}
    for tgt in ["opd_total", "opd_dengue", "ipd_admissions", "lab_tests"]:
        forecast_dict[tgt] = {}
        for hosp in ["NETWORK_TOTAL", "PCM", "HCH", "KFC", "PMC"]:
            sub = forecast_df[(forecast_df["hospital_code"] == hosp) & (forecast_df["target_segment"] == tgt)]
            forecast_dict[tgt][hosp] = sub["point_forecast"].values[:13]
    reconcile_hospital_hierarchy(forecast_dict, horizon=13, output_dir="outputs")
    
    # 7. Scenario Stress Testing
    print("\n[Step 7/7] Simulating Scenarios: Severe Dengue Epidemic Outbreak vs Swasthya Pune Promotion...")
    from src.scenario_analysis import run_scenario_simulations
    run_scenario_simulations(forecast_df, horizon=13, output_dir="outputs")
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"PIPELINE SUCCESSFULLY COMPLETED IN {elapsed:.1f} SECONDS!")
    print("All deliverables generated:")
    print("  - Forecast CSV:          outputs/forecast_next_13_weeks.csv")
    print("  - Planning Table CSV:    outputs/planning_table_capacity.csv")
    print("  - Backtest Metrics CSV:  outputs/backtest_metrics_summary.csv")
    print("  - Reconciled CSV:        outputs/reconciled_forecasts.csv")
    print("  - Scenario Analysis CSV: outputs/scenario_stress_testing.csv")
    print("  - Visualizations:        outputs/figures/*.png")
    print("="*80)

if __name__ == "__main__":
    main()
