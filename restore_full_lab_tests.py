"""
restore_full_lab_tests.py - Restores data/raw/lab_tests.csv to the full original 84 MB file (569,559 rows)
using the exact deterministic seed 42, and cleans up any temporary GitHub sample files.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def restore_full_lab_tests():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    start_date = datetime(2023, 10, 2)
    end_date = datetime(2026, 9, 27)
    total_days = (end_date - start_date).days + 1
    
    hospitals = [
        {"code": "PCM", "name": "Pune Central Memorial", "share": 0.35, "los_mult": 1.2, "growth": 0.05},
        {"code": "HCH", "name": "Hadapsar Community Hospital", "share": 0.25, "los_mult": 0.95, "growth": 0.12},
        {"code": "KFC", "name": "Kothrud Family Clinic & Hospital", "share": 0.20, "los_mult": 0.90, "growth": 0.06},
        {"code": "PMC", "name": "Pimpri Medical Center", "share": 0.20, "los_mult": 1.05, "growth": 0.08}
    ]
    
    clinical_categories = {
        "Dengue_Vector": {
            "weight_base": 0.05,
            "monsoon_boost": 0.32,
            "admit_prob": 0.22,
            "lab_prob": 0.85,
            "los_mean": 4.5,
            "los_std": 1.5,
            "texts": [
                "high fever with chills, severe retro-orbital pain, ?dengue",
                "NS1 antigen positive, thrombocytopenia, severe myalgia",
                "fever x 4 days, rash, bleeding gums, platelets 54000",
                "acute viral fever, suspected dengue with warning signs",
                "severe bodyache, chills, suspected vector borne illness / chikungunya",
                "pediatric fever, persistent vomiting, low platelets, dengue probable",
                "viral pyrexia with intense arthralgia and rash, NS1 reactive",
                "platelet count dropping, petechial rash, admitted for monitoring"
            ]
        },
        "Acute_Respiratory": {
            "weight_base": 0.22,
            "monsoon_boost": 0.05,
            "admit_prob": 0.09,
            "lab_prob": 0.40,
            "los_mean": 3.8,
            "los_std": 1.2,
            "texts": [
                "acute bronchitis, persistent cough with wheeze and low fever",
                "URI with severe throat pain, rhinorrhea and mild fever",
                "copd exacerbation post-diwali smog, tachypnea, needs nebulization",
                "bronchopneumonia with bilateral crackles, productive cough",
                "viral influenza syndrome with headache, pharyngitis and cough",
                "pediatric bronchiolitis, respiratory distress, subcostal retractions",
                "allergic rhinitis with seasonal asthma flare-up"
            ]
        },
        "Gastrointestinal": {
            "weight_base": 0.16,
            "monsoon_boost": 0.10,
            "admit_prob": 0.12,
            "lab_prob": 0.55,
            "los_mean": 3.0,
            "los_std": 1.0,
            "texts": [
                "acute gastroenteritis with vomiting and loose stools x 2 days",
                "AGE with moderate dehydration, poor oral intake, needs IV fluids",
                "suspected enteric fever (typhoid), high fever with step-ladder pattern",
                "acute gastritis, severe epigastric pain and heartburn",
                "food poisoning presentation with crampy abdominal pain and diarrhea",
                "amoebic dysentery with blood in stool and tenesmus"
            ]
        },
        "Chronic_Cardiometabolic": {
            "weight_base": 0.25,
            "monsoon_boost": -0.05,
            "admit_prob": 0.07,
            "lab_prob": 0.65,
            "los_mean": 4.0,
            "los_std": 1.8,
            "texts": [
                "type 2 diabetes mellitus routine review, HbA1c check, FBS elevated",
                "essential hypertension, BP 168/98 mmHg, medication titration",
                "ischemic heart disease follow-up, mild exertional dyspnea, ECG normal",
                "diabetic nephropathy evaluation with microalbuminuria",
                "dyslipidemia review, elevated triglycerides and LDL",
                "chronic heart failure stable, on ACEi and diuretics"
            ]
        },
        "Maternal_Pediatric": {
            "weight_base": 0.15,
            "monsoon_boost": 0.02,
            "admit_prob": 0.11,
            "lab_prob": 0.35,
            "los_mean": 2.8,
            "los_std": 1.1,
            "texts": [
                "routine antenatal checkup 26 weeks, fetal heart sound normal",
                "pediatric immunization, MMR and pentavalent booster, well child",
                "pediatric febrile convulsion with acute otitis media",
                "hyperbilirubinemia in 5-day old neonate, phototherapy review",
                "routine growth monitoring, mild nutritional anemia"
            ]
        },
        "General_Trauma_Surgical": {
            "weight_base": 0.17,
            "monsoon_boost": -0.02,
            "admit_prob": 0.15,
            "lab_prob": 0.50,
            "los_mean": 3.5,
            "los_std": 1.6,
            "texts": [
                "blunt trauma to right knee after two-wheeler road accident on bypass",
                "laceration over left forearm, wound cleaning and suturing done",
                "acute lower back pain with right L5-S1 radiculopathy",
                "symptomatic cholelithiasis, elective laparoscopic cholecystectomy consult",
                "right inguinal hernia reducible, pre-op surgical clearance",
                "abscess right gluteal region, incision and drainage needed"
            ]
        }
    }
    
    festivals = [
        {"name": "Diwali", "date": datetime(2023, 11, 12), "opd_dip": 0.35, "post_surge": 0.20},
        {"name": "Holi", "date": datetime(2024, 3, 25), "opd_dip": 0.15, "post_surge": 0.05},
        {"name": "Ganesh_Chaturthi", "date": datetime(2024, 9, 7), "opd_dip": 0.28, "post_surge": 0.25},
        {"name": "Diwali", "date": datetime(2024, 11, 1), "opd_dip": 0.35, "post_surge": 0.20},
        {"name": "Holi", "date": datetime(2025, 3, 14), "opd_dip": 0.15, "post_surge": 0.05},
        {"name": "Ganesh_Chaturthi", "date": datetime(2025, 8, 27), "opd_dip": 0.28, "post_surge": 0.25},
        {"name": "Diwali", "date": datetime(2025, 10, 20), "opd_dip": 0.35, "post_surge": 0.20},
        {"name": "Holi", "date": datetime(2026, 3, 4), "opd_dip": 0.15, "post_surge": 0.05},
        {"name": "Ganesh_Chaturthi", "date": datetime(2026, 9, 14), "opd_dip": 0.28, "post_surge": 0.25}
    ]
    
    dates = [start_date + timedelta(days=d) for d in range(total_days)]
    n_patients = 120000
    ref_year = 2026
    raw_ages = np.clip(np.random.gamma(shape=3.2, scale=12.0, size=n_patients), 1, 92).astype(int)
    patient_dobs = [
        (datetime(ref_year - age, random.randint(1, 12), random.randint(1, 28))).strftime(
            random.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%Y/%m/%d"])
        )
        for age in raw_ages
    ]
    use_dob_flag = np.random.rand(n_patients) < 0.25
    patient_genders = np.random.choice(["M", "F"], size=n_patients, p=[0.49, 0.51])
    
    opd_id_counter = 1000001
    admit_id_counter = 500001
    lab_id_counter = 2000001
    
    lab_records = []
    
    print("Generating full original lab_tests records (569,559 rows)...")
    for d_idx, dt in enumerate(dates):
        day_of_week = dt.weekday()
        dow_factor = 0.60 if day_of_week == 6 else (1.15 if day_of_week == 0 else 1.0)
        years_elapsed = d_idx / 365.25
        trend_factor = 1.0 + 0.08 * years_elapsed
        week_num = dt.isocalendar()[1]
        is_monsoon = 1.0 if (26 <= week_num <= 42) else 0.0
        monsoon_intensity = np.exp(-0.5 * ((week_num - 34) / 4.0) ** 2) if is_monsoon else 0.0
        
        holiday_factor = 1.0
        for fest in festivals:
            delta_days = (dt - fest["date"]).days
            if -4 <= delta_days <= 3:
                holiday_factor *= (1.0 - fest["opd_dip"])
            elif 4 <= delta_days <= 14:
                holiday_factor *= (1.0 + fest["post_surge"])
                
        daily_base_opd = int(240 * dow_factor * trend_factor * holiday_factor * (1.0 + 0.20 * monsoon_intensity))
        
        for hosp in hospitals:
            hosp_code = hosp["code"]
            hosp_daily_opd = int(daily_base_opd * hosp["share"] * (1.0 + hosp["growth"] * years_elapsed) * np.random.uniform(0.92, 1.08))
            
            cat_weights = {}
            for cat_name, cat_info in clinical_categories.items():
                w = cat_info["weight_base"]
                if cat_name == "Dengue_Vector":
                    w += cat_info["monsoon_boost"] * monsoon_intensity * 2.2
                elif cat_name == "Acute_Respiratory":
                    if 43 <= week_num <= 48:
                        w += 0.12
                elif cat_name == "Gastrointestinal":
                    if is_monsoon:
                        w += 0.08 * monsoon_intensity
                cat_weights[cat_name] = max(0.01, w)
                
            tot_w = sum(cat_weights.values())
            cat_probs = [cat_weights[k]/tot_w for k in clinical_categories.keys()]
            cats_chosen = np.random.choice(list(clinical_categories.keys()), size=hosp_daily_opd, p=cat_probs)
            
            for cat in cats_chosen:
                pat_idx = random.randint(0, n_patients - 1)
                cat_info = clinical_categories[cat]
                diag_text = random.choice(cat_info["texts"])
                
                opd_id = f"OPD-{opd_id_counter}"
                opd_id_counter += 1
                
                # Inpatient simulation (needed to maintain exact random state)
                admit_prob = cat_info["admit_prob"]
                if cat == "Dengue_Vector" and ("platelets" in diag_text or "warning" in diag_text or "NS1" in diag_text):
                    admit_prob = 0.42
                is_admitted = np.random.rand() < admit_prob
                if is_admitted:
                    admit_id_counter += 1
                    los_days = max(1, int(np.random.normal(cat_info["los_mean"] * hosp["los_mult"], cat_info["los_std"])))
                    discharge_dt = dt + timedelta(days=los_days)
                    _ = (np.random.rand() < 0.18) or (discharge_dt > end_date)
                    _ = "ICU" if (cat == "Dengue_Vector" and np.random.rand() < 0.25) else ("General Ward" if np.random.rand() < 0.65 else "Semi-Private")
                    _ = round(np.random.uniform(2000, 15000), 2)
                    _ = round(np.random.uniform(1000, 25000), 2)
                    
                # Lab test orders
                lab_prob = cat_info["lab_prob"]
                if np.random.rand() < lab_prob:
                    tests = ["Complete Blood Count (CBC)"]
                    if cat == "Dengue_Vector":
                        tests.extend(["Dengue NS1 Antigen ELISA", "Platelet Count Serial", "Dengue IgM/IgG Serology"])
                    elif cat == "Gastrointestinal":
                        tests.extend(["Serum Electrolytes", "Widal Test for Typhoid", "Stool Routine & Microscopy"])
                    elif cat == "Chronic_Cardiometabolic":
                        tests.extend(["HbA1c Glycated Hemoglobin", "Lipid Profile Comprehensive", "Kidney Function Test (KFT)"])
                    elif cat == "Acute_Respiratory":
                        tests.extend(["C-Reactive Protein (CRP)", "Chest X-Ray Digital"])
                        
                    for test_name in tests:
                        lab_records.append({
                            "lab_order_id": f"LAB-{lab_id_counter}",
                            "visit_id": opd_id,
                            "patient_id": f"PAT-{pat_idx+100000}",
                            "hospital_code": hosp_code,
                            "order_date": dt.strftime("%Y-%m-%d"),
                            "test_name": test_name,
                            "clinical_indication": diag_text,
                            "result_flag": random.choice(["Normal", "Borderline", "Abnormal / Critical"]) if np.random.rand() < 0.45 else "Normal"
                        })
                        lab_id_counter += 1
                        
    df_lab = pd.DataFrame(lab_records)
    out_file = "data/raw/lab_tests.csv"
    print(f"Writing {len(df_lab):,} rows to {out_file}...")
    df_lab.to_csv(out_file, index=False)
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"DONE! {out_file} is now restored to {size_mb:.2f} MB ({len(df_lab):,} rows).")
    
    # Remove temporary github file and cleanup scripts
    for temp in ["data/raw/lab_tests_github.csv", "downsample_lab_tests.py", "replace_lab_tests.py"]:
        if os.path.exists(temp):
            try:
                os.remove(temp)
                print(f"Cleaned up {temp}.")
            except Exception:
                pass

if __name__ == "__main__":
    restore_full_lab_tests()
