import pandas as pd
import numpy as np

df = pd.read_parquet('data/nhanes/nhanes_unified_master_with_circadian.parquet')
cohort = df[df['is_target_adult']].copy()
print(f'Total cohort: {len(cohort)}')
circ_valid = cohort['circadian_mesor'].notna().sum()
print(f'Circadian complete: {circ_valid}')

print('\n=== DEMOGRAPHICS ===')
print('Age:', cohort['age'].describe().round(2).to_dict())
print('Gender % (1=Male, 2=Female):', (cohort['gender'].value_counts(normalize=True)*100).round(1).to_dict())
print('Ethnicity %:', (cohort['ethnicity'].value_counts(normalize=True)*100).round(1).to_dict())

print('\n=== ANTHROPOMETRICS & VITALS ===')
print('BMI:', cohort['bmi'].describe().round(2).to_dict())
print('Waist cm:', cohort['waist_circ_cm'].describe().round(2).to_dict())
print('Resting HR bpm:', cohort['resting_hr_bpm'].describe().round(2).to_dict())
print('Systolic BP:', cohort['systolic_bp'].describe().round(2).to_dict())
print('Diastolic BP:', cohort['diastolic_bp'].describe().round(2).to_dict())

print('\n=== LABS & TARGETS ===')
print('Fasting Glucose (mg/dL):', cohort['fasting_glucose_mgdl'].describe().round(2).to_dict())
print('Fasting Insulin (uU/mL):', cohort['fasting_insulin_uUml'].describe().round(2).to_dict())
print('HOMA-IR:', cohort['homa_ir'].describe().round(2).to_dict())
print('HbA1c (%):', cohort['hba1c_pct'].describe().round(2).to_dict())
print('Triglycerides (mg/dL):', cohort['triglycerides_mgdl'].describe().round(2).to_dict())
print('HDL (mg/dL):', cohort['hdl_mgdl'].describe().round(2).to_dict())
print('TyG Index:', cohort['tyg_index'].describe().round(2).to_dict())

print('\n=== RISK CLASSES ===')
print(cohort['metabolic_risk_class'].value_counts(dropna=False).to_dict())
print((cohort['metabolic_risk_class'].value_counts(normalize=True, dropna=False)*100).round(1).to_dict())

print('\n=== WEARABLE & CIRCADIAN ===')
circ_cols = ['circadian_mesor', 'circadian_amplitude', 'circadian_acrophase', 'circadian_r2',
             'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
             'm10_value', 'l5_value', 'mean_nightly_sleep_hours', 'max_sedentary_bout_hours', 'valid_wear_days']
print(cohort[circ_cols].describe().round(3).T[['mean', 'std', '25%', '50%', '75%']])

print('\n=== CORRELATIONS ===')
corr_features = ['circadian_mesor', 'circadian_amplitude', 'circadian_acrophase', 'circadian_r2',
                 'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
                 'm10_value', 'l5_value', 'mean_nightly_sleep_hours', 'resting_hr_bpm', 'bmi', 'waist_circ_cm', 'age']
corr_targets = ['homa_ir', 'hba1c_pct', 'fasting_glucose_mgdl', 'fasting_insulin_uUml', 'tyg_index']
print(cohort[corr_features + corr_targets].corr().loc[corr_features, corr_targets].round(3))
