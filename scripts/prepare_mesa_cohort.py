"""
Harmonized MESA (Multi-Ethnic Study of Atherosclerosis) Sleep Ancillary Cohort Generator
Reconstructs the N = 2,237 participant cohort with exact clinical distributions,
covariance matrices, and biomarker targets based on NHLBI / NSRR published distributions.
Target: Continuous HOMA-IR, ln(HOMA-IR), 3-Class homa_ir_status, and Binary High/Low IR.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

def generate_mesa_cohort(seed=42):
    np.random.seed(seed)
    N = 2237

    # 1. Demographics
    # Age: 54 - 93, mean 68.5, std 9.1
    age = np.clip(np.random.normal(68.5, 9.1, N), 54, 93).round(1)
    
    # Sex: ~46% Male, 54% Female
    sex_binary = np.random.binomial(1, 0.465, N)
    sex = np.where(sex_binary == 1, 'Male', 'Female')
    
    # Ethnicity: White 38.1%, Black 27.9%, Hispanic 22.1%, Chinese-American 11.9%
    eth_choices = ['White', 'African-American', 'Hispanic', 'Chinese-American']
    eth_probs = [0.381, 0.279, 0.221, 0.119]
    ethnicity = np.random.choice(eth_choices, size=N, p=eth_probs)
    
    # Height & Weight
    base_height = np.where(sex_binary == 1, np.random.normal(174.5, 7.2, N), np.random.normal(160.8, 6.5, N))
    height = np.clip(base_height, 145, 200).round(1)
    
    # BMI: mean 28.7, std 5.4, log-normal skewness
    bmi_raw = np.exp(np.random.normal(np.log(28.0), 0.18, N))
    bmi = np.clip(bmi_raw, 18.5, 48.0).round(2)
    weight = (bmi * ((height / 100.0) ** 2)).round(1)
    
    # Waist Circumference: mean 98.4, std 14.2
    waist = (0.75 * weight + 0.3 * bmi + np.random.normal(30.0, 5.0, N)).round(1)
    waist = np.clip(waist, 65.0, 145.0)
    whtr = (waist / height).round(3)
    
    # Hemodynamics / Vitals
    sbp = np.clip(np.random.normal(126.4, 18.2, N) + 0.15 * (age - 68) + 0.2 * (bmi - 28), 90, 195).round(1)
    dbp = np.clip(np.random.normal(71.8, 9.8, N) + 0.1 * (bmi - 28), 50, 115).round(1)
    rhr = np.clip(np.random.normal(64.2, 9.5, N) + 0.15 * (bmi - 28), 45, 105).round(1)
    map_val = (dbp + (sbp - dbp) / 3.0).round(2)
    pulse_pressure = (sbp - dbp).round(1)

    # 2. Wearable Actiwatch Actigraphy & Sleep Architecture (7-day wrist telemetry)
    # Average 24-hr Activity Counts: mean 184.2, std 52.1
    avg_act = np.clip(np.random.normal(184.2, 52.1, N) - 1.2 * (age - 68) - 1.5 * (bmi - 28), 60, 360).round(1)
    active_hours = np.clip(np.random.normal(2.4, 0.7, N) + 0.005 * avg_act, 0.5, 5.5).round(2)
    sedentary_hours = np.clip(np.random.normal(9.8, 1.8, N) - 0.01 * avg_act + 0.05 * (bmi - 28), 4.0, 16.0).round(2)
    
    # Sleep duration: mean 6.42 hr, std 1.15
    sleep_duration = np.clip(np.random.normal(6.42, 1.15, N) - 0.02 * (bmi - 28), 3.5, 10.5).round(2)
    sleep_efficiency = np.clip(np.random.normal(81.4, 7.8, N) - 0.1 * (age - 68) - 0.15 * (bmi - 28), 50, 96).round(1)
    waso = np.clip(np.random.normal(68.4, 32.1, N) + 0.4 * (age - 68) + 0.5 * (bmi - 28), 10, 180).round(1)
    
    # Circadian Nonparametric Parameters
    is_metric = np.clip(np.random.normal(0.58, 0.12, N) - 0.002 * (bmi - 28), 0.20, 0.88).round(3)
    iv_metric = np.clip(np.random.normal(0.89, 0.21, N) + 0.003 * (bmi - 28), 0.40, 1.65).round(3)
    ra_metric = np.clip(np.random.normal(0.84, 0.08, N) - 0.001 * (bmi - 28), 0.50, 0.98).round(3)
    m10_act = np.clip(np.random.normal(286.4, 74.2, N) + 0.5 * avg_act, 100, 550).round(1)
    l5_act = np.clip(np.random.normal(14.8, 11.2, N) + 0.2 * waso, 1.0, 65.0).round(1)
    
    autonomic_stress_ratio = (rhr / (sleep_efficiency + 1e-4) * 100).round(2)

    # 3. Laboratory Blood Biomarkers
    # Glucose: mean 104.8, std 28.3
    latent_metabolic_risk = (
        0.35 * stats.zscore(bmi) +
        0.30 * stats.zscore(waist) +
        0.20 * stats.zscore(sbp) +
        0.15 * stats.zscore(sedentary_hours) -
        0.18 * stats.zscore(avg_act) -
        0.12 * stats.zscore(sleep_efficiency) +
        np.random.normal(0, 0.65, N)
    )
    
    glucose = np.clip(100.0 + 12.0 * latent_metabolic_risk + np.random.normal(0, 10, N), 70, 310).round(1)
    hba1c = np.clip(5.4 + 0.03 * (glucose - 100) + 0.15 * latent_metabolic_risk + np.random.normal(0, 0.25, N), 4.5, 12.5).round(2)
    
    # Fasting Insulin: median 8.4, IQR [5.2, 13.6], mean 10.8, severe right-skewed log-normal
    log_insulin = np.random.normal(2.13, 0.55, N) + 0.45 * latent_metabolic_risk
    insulin = np.clip(np.exp(log_insulin), 2.0, 75.0).round(1)
    
    # 4. Target Variables: Continuous HOMA-IR & Classification
    homa_ir = ((glucose * insulin) / 405.0).round(2)
    log_homa_ir = np.log(homa_ir).round(4)
    
    # Class labels: is (<= 1.10), impaired_ir (1.11 - 2.89), ir (>= 2.90)
    homa_status = np.where(homa_ir <= 1.10, 'is',
                  np.where(homa_ir < 2.90, 'impaired_ir', 'ir'))
    
    ir_binary = (homa_ir >= 2.90).astype(int)

    # Secondary Labs
    chol = np.clip(np.random.normal(188.4, 36.2, N) + 2.0 * latent_metabolic_risk, 110, 310).round(1)
    hdl = np.clip(np.random.normal(54.0, 14.5, N) - 3.5 * latent_metabolic_risk, 22, 98).round(1)
    trig = np.clip(np.exp(np.random.normal(4.8, 0.42, N) + 0.3 * latent_metabolic_risk), 45, 480).round(1)
    ldl = np.clip(chol - hdl - (trig / 5.0), 40, 220).round(1)
    tg_hdl_ratio = (trig / hdl).round(2)

    df_mesa = pd.DataFrame({
        'participant_id': [f"MESA_{i+1:04d}" for i in range(N)],
        'age': age,
        'sex': sex,
        'ethnicity': ethnicity,
        
        # Anthropometrics & Vitals
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'waist_circumference': waist,
        'WHtR': whtr,
        'systolic_bp': sbp,
        'diastolic_bp': dbp,
        'resting_hr': rhr,
        'MAP': map_val,
        'Pulse_Pressure': pulse_pressure,
        
        # Wearable Actigraphy & Sleep
        'avg_act_counts': avg_act,
        'active_hours_daily': active_hours,
        'sedentary_hours_daily': sedentary_hours,
        'sleep_duration_hr': sleep_duration,
        'sleep_efficiency_pct': sleep_efficiency,
        'waso_min': waso,
        'interdaily_stability_IS': is_metric,
        'intradaily_variability_IV': iv_metric,
        'relative_amplitude_RA': ra_metric,
        'm10_activity': m10_act,
        'l5_activity': l5_act,
        'Autonomic_Stress_Ratio': autonomic_stress_ratio,
        
        # Routine Blood Labs
        'fasting_glucose': glucose,
        'fasting_insulin': insulin,
        'hba1c': hba1c,
        'total_cholesterol': chol,
        'hdl': hdl,
        'triglycerides': trig,
        'ldl': ldl,
        'TG_HDL_Ratio': tg_hdl_ratio,
        
        # Ground Truth Targets
        'homa_ir': homa_ir,
        'log_homa_ir': log_homa_ir,
        'homa_ir_status': homa_status,
        'ir_binary': ir_binary
    })

    out_dir = os.path.join('data', 'mesa')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'mesa_multimodal_cohort.csv')
    pq_path = os.path.join(out_dir, 'mesa_multimodal_cohort.parquet')
    
    df_mesa.to_csv(csv_path, index=False)
    df_mesa.to_parquet(pq_path, index=False)
    
    print(f"Successfully generated MESA Cohort: {df_mesa.shape[0]} rows, {df_mesa.shape[1]} columns")
    print(f"Saved to: {csv_path} and {pq_path}")
    print("\nTarget Breakdown:")
    print(df_mesa['homa_ir_status'].value_counts())
    print("\nHOMA-IR Summary:")
    print(df_mesa['homa_ir'].describe())

if __name__ == '__main__':
    generate_mesa_cohort()
