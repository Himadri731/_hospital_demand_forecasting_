# Hospital Demand Forecasting: Pune Community Clinic Network
### Quarterly Forecasting, Operational Capacity Planning & Scenario Stress Testing

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://python.org)
[![Models](https://img.shields.io/badge/Models-Prophet%20%7C%20LightGBM%20%7C%20ETS%20%7C%20SNaive-green.svg)](#forecasting-architectures)
[![MAPE](https://img.shields.io/badge/Hold--out%20MAPE-9.42%25-success.svg)](#backtest-performance)
[![Baseline Win Rate](https://img.shields.io/badge/Beat%20Baseline-95.0%25-success.svg)](#success-criteria-audit)

A comprehensive, production-grade clinical demand forecasting and operational planning engine built for a network of 4 community hospitals in Pune, India (**Pune Central Memorial**, **Hadapsar Community Hospital**, **Kothrud Family Clinic**, and **Pimpri Medical Center**).

---

## 🎯 Executive Overview & Problem Context
Planners previously set targets using **"last year's numbers plus a guess"**, leading to severe ward saturation, 4-hour OPD queues, and acute staffing shortages during Pune's post-monsoon dengue season (July–October). 

This project provides an automated, end-to-end forecasting pipeline that ingests **980,000+ raw transactional records**, handles messy unstructured clinical notes and missing dates, decomposes Pune-specific monsoon and festival cycles, trains and benchmarks 4 model architectures, and translates probabilistic forecasts into bed, shift, and patient follow-up schedules.

---

## 🏆 Success Criteria Verification
| Objective / Metric | Target Requirement | Achieved Result | Status |
|:---|:---:|:---:|:---:|
| **Beat Seasonal Naive Baseline** | $\ge 70\%$ of segments | **95.0%** (19 / 20 segments) |  **EXCEEDED** |
| **Overall Hold-out MAPE** | $< 15.0\%$ | **9.42%** |  **EXCEEDED** |
| **Baseline Average MAPE** | Benchmark | 16.44% | *Baseline outperformed by 7.02%* |
| **Rolling-Origin Hold-out** | 13 weeks (1 quarter) | Fully backtested across 20 segments |  **COMPLETE** |
| **Operational Planning Translation** | Beds, Staff, Calls | Formulated with 80% & 95% intervals |  **COMPLETE** |
| **Hierarchical Reconciliation** | Segments add to Total | MinT / OLS & Bottom-Up coherent |  **COMPLETE** |
| **Scenario Stress Testing** | Epidemic & Tariff shocks | Severe Dengue Shock vs Promotion |  **COMPLETE** |

---

## 📁 Repository Directory Structure

```
hospital_demand_forecasting/
├── data/
│   ├── raw/                              # ~980k synthetic transactional clinical records
│   │   ├── opd_visits.csv                # 324k OPD records with free-text diagnosis & mixed age/DOB
│   │   ├── admissions.csv                # 44k Inpatient admissions with missing discharge dates
│   │   ├── lab_tests.csv                 # 570k Diagnostic lab orders (CBC, NS1, Platelet, LFT)
│   │   └── billing_records.csv           # 44k Billing invoices linked to inpatient stays
│   └── processed/
│       ├── weekly_aggregated_segments.csv# Weekly time series across 4 hospitals + Network Total (156 wks)
│       └── pune_calendar_features.csv    # Monsoon index, Diwali, Ganesh Chaturthi, Holi indicators
├── src/
│   ├── data_generator.py                 # Vectorized clinical data generator with realistic quirks
│   ├── preprocessing.py                  # Clinical NLP parser, DOB handler, discharge imputer
│   ├── eda_decomposition.py              # STL decomposition & Pune festival impact analysis
│   ├── models.py                         # Seasonal Naive, ETS, Prophet, LightGBM with prediction intervals
│   ├── backtest_evaluation.py            # Rolling-origin cross-validation, MAPE, Bias, best model selection
│   ├── capacity_planning.py              # Operational translation to beds, shifts, and follow-up calls
│   ├── hierarchical_reconciliation.py    # MinT / OLS optimal hierarchical forecast reconciliation
│   └── scenario_analysis.py              # Severe Dengue Outbreak vs Swasthya Pune Promotion simulations
├── outputs/
│   ├── figures/                          # Publication-grade figures & charts
│   │   ├── backtest_actual_vs_pred_best_models.png
│   │   ├── capacity_planning_schedule_fans.png
│   │   ├── eda_decomposition_stl.png
│   │   ├── eda_dengue_surge_vs_monsoon.png
│   │   ├── eda_festival_impact_boxplots.png
│   │   ├── eda_time_series_overview.png
│   │   ├── reconciliation_comparison.png
│   │   └── scenario_stress_test_comparison.png
│   ├── forecast_next_13_weeks.csv        # Segment-level point, 80%, and 95% forecasts
│   ├── planning_table_capacity.csv       # Beds, doctor shifts, nurse shifts, follow-up calls schedule
│   ├── backtest_metrics_summary.csv      # Complete MAPE, Bias, RMSE per model and segment
│   ├── reconciled_forecasts.csv          # Coherent top-down & bottom-up reconciled series
│   └── scenario_stress_testing.csv       # Baseline vs Scenario A vs Scenario B comparisons
├── reports/
│   └── Executive_Planning_Memo_Q1_2027.md# Two-page strategic planning memorandum for C-suite
├── hospital_demand_forecasting.ipynb     # Interactive Jupyter Notebook walkthrough
├── run_pipeline.py                       # Single-click master execution pipeline
├── requirements.txt                      # Python dependencies
└── README.md                             # This documentation
```

---

## 🚀 Quick Start Guide

Open the project folder in your IDE, then run these commands from the project root (the folder containing `run_pipeline.py`):

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The pipeline prepares the data, evaluates and selects forecasting models, generates 13-week forecasts, and writes the planning, reconciliation, and scenario outputs.

### Open the notebook

From the same project root, run:

```powershell
jupyter notebook hospital_demand_forecasting.ipynb

## 🔬 Core Modeling Architectures

1. **Seasonal Naive Baseline:**  
   $$y_t = y_{t-52}$$  
   Forecasts using the exact observation from the corresponding week in the previous year.
2. **Holt-Winters Exponential Smoothing (ETS):**  
   Additive trend with 52-week additive annual seasonality. Selected for stable clinic-level streams.
3. **Prophet (Generalized Additive Model):**  
   $$y(t) = g(t) + s(t) + h(t) + \beta X_t + \epsilon_t$$  
   Piecewise linear trend + Fourier annual seasonality ($N=10$) + customized Pune holiday regressors (`fest_diwali`, `fest_ganesh_chaturthi`, `fest_holi`, `monsoon_rain_index`). Selected for 17 out of 20 segments.
4. **LightGBM Time-Series Regressor:**  
   Gradient boosted regression with autoregressive lags ($t-1, t-2, t-3, t-4, t-13, t-52$), 4-week and 13-week rolling means/stds/min/max, cyclical calendar transforms, and quantile regressors ($\alpha \in \{0.025, 0.10, 0.90, 0.975\}$) for non-parametric prediction intervals. Selected for complex non-linear segments like Network Dengue.
5. **Hierarchical Reconciliation (MinT / OLS):**  
   $$\tilde{\mathbf{y}} = S (S^T S)^{-1} S^T \hat{\mathbf{y}}$$  
   Projects independent top-level and bottom-level forecasts onto the coherent subspace spanned by summing matrix $S$, eliminating administrative mismatch between hospital rosters and central procurement.

---

## 🏥 Operational Capacity Planning Logic

* **Inpatient Beds Required (85% Occupancy Target):**
  $$\text{Beds} = \frac{\text{Weekly Admissions} \times \text{Average LoS}}{7 \times 0.85}$$
* **Outpatient Doctor Shifts:**
  $$\text{Doctor Shifts (6-Hour Clinical Blocks)} = \left\lceil \frac{\text{Weekly OPD Visits}}{25 \text{ patients/shift}} \right\rceil$$
* **Inpatient Nursing Shifts:**
  $$\text{Nurse Shifts} = \left\lceil \frac{\text{Bed-Days} \times 3 \text{ shifts/day}}{5 \text{ patients/nurse}} \right\rceil$$
* **Care Coordination Calls:**
  $$\text{Follow-Up Calls} = 0.80 \times \text{Admissions} + 0.20 \times \text{Chronic OPD}$$

---

## ⚠️ Critical Reliability Advisory: Where the Forecast is Least Reliable & Why

1. **Hyper-Localized Epidemic Surges (Dengue OPD in Pimpri & Hadapsar):**
   * *Why:* Vector-borne transmission depends on micro-climatic pooling, water tank maintenance, and local municipal fogging drives. A sudden cluster in a single municipal ward can cause a 300% surge within 72 hours that cannot be predicted by macro rainfall regressors alone (MAPE 13.5%–18.6%).
   * *Mitigation:* Operationalize the **95% prediction interval** as an emergency buffer for rapid-mobilization staffing and platelet kit procurement.
2. **Small-Count Poisson Noise in Community Clinics (KFC & PMC Inpatient Admissions):**
   * *Why:* When baseline admissions are small (25–35 per week), small integer fluctuations (e.g. 5 extra admissions from an auto-rickshaw collision on the highway) produce large percentage swings (MAPE ~11%).
   * *Mitigation:* Coordinate network-wide bed management to divert acute overflow to Pune Central Memorial.
3. **Asymmetric Post-Diwali Rebound Timing:**
   * *Why:* The exact week of the festival dip is predictable, but whether patients return in 7 or 14 days depends on school reopenings and agricultural harvest schedules.
   * *Mitigation:* Re-anchor the models weekly every Monday morning during the festival month.

---
*Developed for Pune Community Clinic Network Leadership.*
