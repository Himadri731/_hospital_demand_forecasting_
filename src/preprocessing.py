"""
preprocessing.py - Data cleaning, clinical NLP diagnosis classification,
age resolution (DOB vs numeric), missing discharge date imputation,
and weekly time-series aggregation across key hospital segments.
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def parse_patient_age(age_val, encounter_date_str):
    """
    Handles mixed age inputs:
    - If already numeric / integer string, returns integer age.
    - If date string (e.g. '1984-05-12', '15/03/1972', '04-Jul-1995'),
      computes age in completed years at the time of the encounter.
    """
    if pd.isna(age_val):
        return 35 # fallback adult median
    
    # Check if purely digits
    str_val = str(age_val).strip()
    if str_val.isdigit():
        return int(str_val)
    
    # Try parsing as date of birth
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%b-%Y", "%Y/%m/%d", "%d-%m-%Y"
    ]
    dob = None
    for fmt in date_formats:
        try:
            dob = datetime.strptime(str_val, fmt)
            break
        except ValueError:
            continue
            
    if dob is not None:
        try:
            enc_date = datetime.strptime(str(encounter_date_str)[:10], "%Y-%m-%d")
            age = enc_date.year - dob.year - ((enc_date.month, enc_date.day) < (dob.month, dob.day))
            return max(0, min(105, age))
        except Exception:
            return 35
            
    # Try regex extraction of first number
    match = re.search(r'\b\d{1,3}\b', str_val)
    if match:
        val = int(match.group(0))
        if 0 <= val <= 105:
            return val
            
    return 35

def classify_clinical_diagnosis(text):
    """
    Standardizes unstructured doctor clinical notes into defined cohorts:
    - Dengue_Vector
    - Acute_Respiratory
    - Gastrointestinal
    - Chronic_Cardiometabolic
    - Maternal_Pediatric
    - General_Trauma_Surgical
    """
    if pd.isna(text):
        return "General_Trauma_Surgical"
    
    text_lower = str(text).lower()
    
    # Dengue / Vector-borne
    if re.search(r'dengue|\bns1\b|thrombocytopen|platelet|chikungunya|malaria|vector|petechi|retro-orbital', text_lower):
        return "Dengue_Vector"
    
    # Acute Respiratory
    if re.search(r'bronchitis|uri|copd|wheez|cough|breathless|pharyngitis|pneumonia|rhinitis|asthma|smog|influenza|flu|subcostal', text_lower):
        return "Acute_Respiratory"
        
    # Gastrointestinal / Diarrheal
    if re.search(r'gastroenteritis|\bage\b|dehydrat|diarrhea|loose stool|vomit|typhoid|enteric|gastritis|food poison|dysentery|tenesmus', text_lower):
        return "Gastrointestinal"
        
    # Chronic Cardiometabolic
    if re.search(r'diabetes|\bt2dm\b|\bfbs\b|hba1c|hypertens|\bhtn\b|\bbp\b|angina|ischemic|\bchf\b|renal|nephropathy|lipid|cholesterol', text_lower):
        return "Chronic_Cardiometabolic"
        
    # Maternal & Pediatric
    if re.search(r'antenatal|fetal|pediatric|vaccin|immuniz|mmr|booster|neonate|phototherapy|hyperbilirubin|febrile convulsion|growth monitoring', text_lower):
        return "Maternal_Pediatric"
        
    # Default / General Trauma
    return "General_Trauma_Surgical"

def impute_missing_discharge_dates(df_adm):
    """
    Inpatient discharge dates can be missing due to active stays, clerical lag, or transfers.
    Imputes discharge date based on empirical length of stay (LoS) by diagnosis cohort and ward type.
    """
    df = df_adm.copy()
    df["admission_dt"] = pd.to_datetime(df["admission_date"])
    df["discharge_dt"] = pd.to_datetime(df["discharge_date"])
    
    # Calculate observed LoS for records with non-null discharge
    known_mask = df["discharge_dt"].notna()
    df.loc[known_mask, "los_days"] = (df.loc[known_mask, "discharge_dt"] - df.loc[known_mask, "admission_dt"]).dt.days
    df.loc[known_mask & (df["los_days"] < 1), "los_days"] = 1 # min 1 day stay
    
    # Compute median LoS by clinical category & ward type
    los_lookup = df[known_mask].groupby(["clinical_category", "ward_type"])["los_days"].median().to_dict()
    overall_median = df.loc[known_mask, "los_days"].median() or 3.0
    
    # Impute missing discharge dates
    missing_mask = df["discharge_dt"].isna()
    imputed_los = []
    imputed_dates = []
    
    for idx, row in df[missing_mask].iterrows():
        key = (row["clinical_category"], row["ward_type"])
        est_los = int(los_lookup.get(key, overall_median))
        # Add random jitter between -1 and +1 day
        jittered_los = int(max(1, est_los + int(np.random.choice([-1, 0, 1]))))
        adm_date = row["admission_dt"]
        disch_date = adm_date + timedelta(days=int(jittered_los))
        imputed_los.append(jittered_los)
        imputed_dates.append(disch_date)
        
    df.loc[missing_mask, "los_days"] = imputed_los
    df.loc[missing_mask, "discharge_dt"] = imputed_dates
    df["imputed_discharge_flag"] = missing_mask.astype(int)
    
    return df

def generate_pune_calendar(start_date="2023-10-02", end_date="2027-01-31"):
    """
    Builds weekly calendar features including Pune-specific monsoon cycles,
    Ganesh Chaturthi, Diwali, Holi, and seasonal indexes.
    """
    weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    cal_df = pd.DataFrame({"week_start": weeks})
    cal_df["year"] = cal_df["week_start"].dt.year
    cal_df["month"] = cal_df["week_start"].dt.month
    cal_df["week_of_year"] = cal_df["week_start"].dt.isocalendar().week.astype(int)
    cal_df["quarter"] = cal_df["week_start"].dt.quarter
    
    # Pune Monsoon / Dengue Intensity index (Weeks 26 to 42, peaking around week 34)
    # Modeled via Gaussian bell curve reflecting Pune vector-breeding season
    cal_df["monsoon_rain_index"] = 0.0
    for i, row in cal_df.iterrows():
        woy = row["week_of_year"]
        if 26 <= woy <= 42:
            intensity = np.exp(-0.5 * ((woy - 34) / 4.0) ** 2)
            cal_df.at[i, "monsoon_rain_index"] = round(intensity, 4)
            
    # Major Pune Festival indicators with lead and lag windows
    festival_dates = [
        # Diwali
        datetime(2023, 11, 12), datetime(2024, 11, 1), datetime(2025, 10, 20), datetime(2026, 11, 8),
        # Ganesh Chaturthi
        datetime(2024, 9, 7), datetime(2025, 8, 27), datetime(2026, 9, 14),
        # Holi
        datetime(2024, 3, 25), datetime(2025, 3, 14), datetime(2026, 3, 4)
    ]
    
    cal_df["fest_diwali"] = 0
    cal_df["fest_diwali_post_surge"] = 0
    cal_df["fest_ganesh_chaturthi"] = 0
    cal_df["fest_ganesh_post_surge"] = 0
    cal_df["fest_holi"] = 0
    
    for i, row in cal_df.iterrows():
        w_start = row["week_start"]
        w_end = w_start + timedelta(days=6)
        
        # Check Diwali
        diwali_dates = [datetime(2023, 11, 12), datetime(2024, 11, 1), datetime(2025, 10, 20), datetime(2026, 11, 8)]
        for d in diwali_dates:
            if w_start <= d <= w_end:
                cal_df.at[i, "fest_diwali"] = 1
            # 1 to 2 weeks post Diwali
            if 1 <= (w_start - d).days <= 14:
                cal_df.at[i, "fest_diwali_post_surge"] = 1
                
        # Check Ganesh Chaturthi (10-day celebration in Pune)
        ganesh_dates = [datetime(2024, 9, 7), datetime(2025, 8, 27), datetime(2026, 9, 14)]
        for g in ganesh_dates:
            if (w_start - timedelta(days=5)) <= g <= (w_end + timedelta(days=5)):
                cal_df.at[i, "fest_ganesh_chaturthi"] = 1
            if 7 <= (w_start - g).days <= 21:
                cal_df.at[i, "fest_ganesh_post_surge"] = 1
                
        # Check Holi
        holi_dates = [datetime(2024, 3, 25), datetime(2025, 3, 14), datetime(2026, 3, 4)]
        for h in holi_dates:
            if w_start <= h <= w_end:
                cal_df.at[i, "fest_holi"] = 1
                
    return cal_df

def process_and_aggregate_weekly(data_dir="data/raw", output_dir="data/processed"):
    """
    Main preprocessing pipeline:
    1. Loads raw CSVs (OPD, Inpatient, Labs, Billing).
    2. Resolves age (DOB vs integer).
    3. Standardizes diagnosis text into clinical categories.
    4. Imputes missing discharge dates for inpatient records.
    5. Aggregates weekly series per hospital and key clinical segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Loading raw clinical records...")
    
    df_opd = pd.read_csv(os.path.join(data_dir, "opd_visits.csv"))
    df_adm = pd.read_csv(os.path.join(data_dir, "admissions.csv"))
    df_lab = pd.read_csv(os.path.join(data_dir, "lab_tests.csv"))
    df_bil = pd.read_csv(os.path.join(data_dir, "billing_records.csv"))
    
    print(f"Loaded: OPD={len(df_opd):,}, Admissions={len(df_adm):,}, Labs={len(df_lab):,}, Billing={len(df_bil):,}")
    
    # 1. Clean Age
    print("Standardizing patient age (resolving DOB strings vs numeric)...")
    df_opd["clean_age"] = [parse_patient_age(a, d) for a, d in zip(df_opd["patient_age"], df_opd["visit_date"])]
    df_adm["clean_age"] = [parse_patient_age(a, d) for a, d in zip(df_adm["patient_age"], df_adm["admission_date"])]
    
    df_opd["age_group"] = pd.cut(df_opd["clean_age"], bins=[-1, 17, 64, 120], labels=["Pediatric", "Adult", "Geriatric"])
    df_adm["age_group"] = pd.cut(df_adm["clean_age"], bins=[-1, 17, 64, 120], labels=["Pediatric", "Adult", "Geriatric"])
    
    # 2. Classify Free-Text Diagnosis
    print("Applying clinical NLP taxonomy parser to unstructured diagnoses...")
    df_opd["clinical_category"] = df_opd["free_text_diagnosis"].apply(classify_clinical_diagnosis)
    df_adm["clinical_category"] = df_adm["free_text_diagnosis"].apply(classify_clinical_diagnosis)
    
    # 3. Impute Missing Discharge Dates
    print("Imputing missing discharge dates using clinical cohort length of stay...")
    df_adm = impute_missing_discharge_dates(df_adm)
    
    # 4. Weekly Grouping
    print("Aggregating into clean weekly time series...")
    df_opd["visit_dt"] = pd.to_datetime(df_opd["visit_date"])
    # Align to Monday of that week
    df_opd["week_start"] = df_opd["visit_dt"].apply(lambda d: d - timedelta(days=d.weekday()))
    
    df_adm["admission_dt"] = pd.to_datetime(df_adm["admission_date"])
    df_adm["week_start"] = df_adm["admission_dt"].apply(lambda d: d - timedelta(days=d.weekday()))
    
    df_lab["order_dt"] = pd.to_datetime(df_lab["order_date"])
    df_lab["week_start"] = df_lab["order_dt"].apply(lambda d: d - timedelta(days=d.weekday()))
    
    # Aggregate counts per hospital x week
    # OPD Total
    agg_opd = df_opd.groupby(["hospital_code", "week_start"]).size().reset_index(name="opd_total")
    # OPD Dengue
    agg_opd_dengue = df_opd[df_opd["clinical_category"] == "Dengue_Vector"].groupby(
        ["hospital_code", "week_start"]
    ).size().reset_index(name="opd_dengue")
    # Admissions Total
    agg_adm = df_adm.groupby(["hospital_code", "week_start"]).size().reset_index(name="ipd_admissions")
    # Inpatient Bed Days (weekly sum of length of stay)
    agg_bed_days = df_adm.groupby(["hospital_code", "week_start"])["los_days"].sum().reset_index(name="ipd_bed_days")
    # Lab tests
    agg_lab = df_lab.groupby(["hospital_code", "week_start"]).size().reset_index(name="lab_tests")
    # Platelet / Dengue specific tests
    agg_lab_dengue = df_lab[df_lab["test_name"].str.contains("Dengue|Platelet", case=False, na=False)].groupby(
        ["hospital_code", "week_start"]
    ).size().reset_index(name="lab_dengue_platelet")
    
    # Merge hospital-level series
    merged_hosp = pd.merge(agg_opd, agg_opd_dengue, on=["hospital_code", "week_start"], how="left")
    merged_hosp = pd.merge(merged_hosp, agg_adm, on=["hospital_code", "week_start"], how="left")
    merged_hosp = pd.merge(merged_hosp, agg_bed_days, on=["hospital_code", "week_start"], how="left")
    merged_hosp = pd.merge(merged_hosp, agg_lab, on=["hospital_code", "week_start"], how="left")
    merged_hosp = pd.merge(merged_hosp, agg_lab_dengue, on=["hospital_code", "week_start"], how="left")
    merged_hosp = merged_hosp.fillna(0)
    
    # Generate Calendar Features
    start_wk = merged_hosp["week_start"].min()
    end_wk = merged_hosp["week_start"].max() + timedelta(weeks=14)
    cal_features = generate_pune_calendar(start_date=start_wk.strftime("%Y-%m-%d"), end_date=end_wk.strftime("%Y-%m-%d"))
    
    # Merge calendar with hospital series
    merged_hosp = pd.merge(merged_hosp, cal_features, on="week_start", how="left")
    
    # Network Total Aggregation
    agg_network = merged_hosp.groupby("week_start")[
        ["opd_total", "opd_dengue", "ipd_admissions", "ipd_bed_days", "lab_tests", "lab_dengue_platelet"]
    ].sum().reset_index()
    agg_network["hospital_code"] = "NETWORK_TOTAL"
    agg_network = pd.merge(agg_network, cal_features, on="week_start", how="left")
    
    # Combine hospital level and network level
    full_series = pd.concat([merged_hosp, agg_network], ignore_index=True).sort_values(["hospital_code", "week_start"])
    
    # Save cleaned processed outputs
    print("Saving processed weekly time series...")
    full_series.to_csv(os.path.join(output_dir, "weekly_aggregated_segments.csv"), index=False)
    cal_features.to_csv(os.path.join(output_dir, "pune_calendar_features.csv"), index=False)
    
    # Save clean samples for inspectability
    df_opd.head(1000).to_csv(os.path.join(output_dir, "sample_clean_opd.csv"), index=False)
    df_adm.head(1000).to_csv(os.path.join(output_dir, "sample_clean_admissions.csv"), index=False)
    
    print(f"Weekly aggregated dataset created: {len(full_series)} rows across {full_series['hospital_code'].nunique()} entities and {full_series['week_start'].nunique()} weeks.")
    return full_series, cal_features

if __name__ == "__main__":
    process_and_aggregate_weekly()
