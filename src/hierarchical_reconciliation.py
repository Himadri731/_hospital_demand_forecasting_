"""
hierarchical_reconciliation.py - Reconciles independent segment forecasts
using Bottom-Up, Top-Down, and MinT / OLS Optimal Reconciliation so that
branch-level forecasts mathematically sum to network totals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reconcile_hospital_hierarchy(forecasts_dict, horizon=13, output_dir="outputs"):
    """
    forecasts_dict structure:
    {
       'target_name': {
           'NETWORK_TOTAL': array(13),
           'PCM': array(13),
           'HCH': array(13),
           'KFC': array(13),
           'PMC': array(13)
       }
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    hospitals = ["PCM", "HCH", "KFC", "PMC"]
    # Summing matrix S: (5 x 4)
    # Row 0: Network total = sum of all 4
    # Rows 1-4: Identity for the 4 bottom hospitals
    S = np.zeros((5, 4))
    S[0, :] = 1.0
    S[1:, :] = np.eye(4)
    
    # OLS projection operator: P = (S^T S)^(-1) S^T
    # Reconciled = S P y_hat
    STS_inv = np.linalg.inv(S.T @ S)
    P_ols = STS_inv @ S.T
    
    reconciled_results = []
    
    print("\nRunning Hierarchical Forecast Reconciliation (MinT / OLS & Bottom-Up)...")
    
    for tgt, series_dict in forecasts_dict.items():
        # Check if all required series are present
        if "NETWORK_TOTAL" not in series_dict or not all(h in series_dict for h in hospitals):
            continue
            
        y_top = series_dict["NETWORK_TOTAL"]
        y_bottoms = np.array([series_dict[h] for h in hospitals]) # shape (4, horizon)
        
        # Base vector y_hat of shape (5, horizon)
        y_hat = np.vstack([y_top, y_bottoms])
        
        # 1. Base Discrepancy (Sum of bottom - Top)
        unreconciled_sum_bottom = np.sum(y_bottoms, axis=0)
        coherence_gap_pre = unreconciled_sum_bottom - y_top
        
        # 2. Bottom-Up Reconciliation
        bu_reconciled = y_hat.copy()
        bu_reconciled[0, :] = unreconciled_sum_bottom
        
        # 3. MinT / OLS Optimal Reconciliation
        # For each time step t: y_rec = S @ P_ols @ y_hat[:, t]
        ols_reconciled = np.zeros_like(y_hat)
        for t in range(horizon):
            ols_reconciled[:, t] = S @ (P_ols @ y_hat[:, t])
            
        for t in range(horizon):
            reconciled_results.append({
                "target": tgt,
                "week_index": t + 1,
                "unreconciled_network": round(y_top[t], 2),
                "unreconciled_bottom_sum": round(unreconciled_sum_bottom[t], 2),
                "pre_reconciliation_gap": round(coherence_gap_pre[t], 2),
                "reconciled_network_ols": round(ols_reconciled[0, t], 2),
                "reconciled_PCM_ols": round(ols_reconciled[1, t], 2),
                "reconciled_HCH_ols": round(ols_reconciled[2, t], 2),
                "reconciled_KFC_ols": round(ols_reconciled[3, t], 2),
                "reconciled_PMC_ols": round(ols_reconciled[4, t], 2),
                "reconciled_bottom_sum_ols": round(np.sum(ols_reconciled[1:, t]), 2),
                "post_reconciliation_gap": round(np.sum(ols_reconciled[1:, t]) - ols_reconciled[0, t], 6)
            })
            
    df_rec = pd.DataFrame(reconciled_results)
    csv_path = os.path.join(output_dir, "reconciled_forecasts.csv")
    df_rec.to_csv(csv_path, index=False)
    print(f"Saved reconciled forecasts to: {csv_path}")
    
    # Visualization: Discrepancy before and after reconciliation for OPD Total
    sub_opd = df_rec[df_rec["target"] == "opd_total"]
    if len(sub_opd) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        wks = sub_opd["week_index"]
        ax.plot(wks, sub_opd["unreconciled_network"], "k--", label="Independent Network Top Forecast", lw=2)
        ax.plot(wks, sub_opd["unreconciled_bottom_sum"], "r:", label="Sum of Independent Hospital Forecasts (Bottom-Up)", lw=2)
        ax.plot(wks, sub_opd["reconciled_network_ols"], "b-", label="MinT / OLS Reconciled Coherent Forecast", lw=2.5)
        
        ax.fill_between(wks, sub_opd["unreconciled_network"], sub_opd["unreconciled_bottom_sum"],
                        color="red", alpha=0.15, label="Pre-Reconciliation Incoherence Gap")
        ax.set_title("Hierarchical Forecast Reconciliation: Resolving Top-Down vs Bottom-Up Incoherence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Forecast Horizon Week (1 to 13)", fontsize=11)
        ax.set_ylabel("Total OPD Encounters", fontsize=11)
        ax.legend(loc="lower right", frameon=True)
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "reconciliation_comparison.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"Saved reconciliation figure: {fig_path}")
        
    return df_rec
