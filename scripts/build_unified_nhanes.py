import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nhanes')

def load_and_merge_cycle(cycle_suffix):
    print(f"\n--- Loading and Merging Cycle {cycle_suffix} ---")
    # 1. Demographics
    demo = pd.read_sas(os.path.join(DATA_DIR, f'DEMO_{cycle_suffix}.XPT'), encoding='iso-8859-1')
    demo = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR']].copy()
    demo.rename(columns={
        'RIDAGEYR': 'age',
        'RIAGENDR': 'gender',      # 1=Male, 2=Female
        'RIDRETH3': 'ethnicity',
        'DMDEDUC2': 'education',
        'INDFMPIR': 'poverty_ratio'
    }, inplace=True)
    print(f"Demographics loaded: {len(demo)} participants")

    # 2. Glucose & Insulin
    glu = pd.read_sas(os.path.join(DATA_DIR, f'GLU_{cycle_suffix}.XPT'), encoding='iso-8859-1')
    if 'LBXIN' in glu.columns:
        glu = glu[['SEQN', 'LBXGLU', 'LBXIN', 'WTSAF2YR']].copy()
    else:
        # In 2013-2014, insulin is in INS_H.XPT
        ins = pd.read_sas(os.path.join(DATA_DIR, f'INS_{cycle_suffix}.XPT'), encoding='iso-8859-1')
        glu = glu[['SEQN', 'LBXGLU', 'WTSAF2YR']].merge(ins[['SEQN', 'LBXIN']], on='SEQN', how='inner')
    
    glu.rename(columns={
        'LBXGLU': 'fasting_glucose_mgdl',
        'LBXIN': 'fasting_insulin_uUml',
        'WTSAF2YR': 'fasting_weight'
    }, inplace=True)
    print(f"Glucose & Insulin loaded: {len(glu)} participants with labs")

    # 3. HbA1c (Glycohemoglobin)
    ghb = pd.read_sas(os.path.join(DATA_DIR, f'GHB_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'LBXGH']].copy()
    ghb.rename(columns={'LBXGH': 'hba1c_pct'}, inplace=True)

    # 4. Lipids: HDL, Total Cholesterol, Triglycerides
    hdl = pd.read_sas(os.path.join(DATA_DIR, f'HDL_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'LBDHDD']].copy()
    hdl.rename(columns={'LBDHDD': 'hdl_mgdl'}, inplace=True)
    
    tchol = pd.read_sas(os.path.join(DATA_DIR, f'TCHOL_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'LBXTC']].copy()
    tchol.rename(columns={'LBXTC': 'total_cholesterol_mgdl'}, inplace=True)

    trig = pd.read_sas(os.path.join(DATA_DIR, f'TRIGLY_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'LBXTR', 'LBDLDL']].copy()
    trig.rename(columns={'LBXTR': 'triglycerides_mgdl', 'LBDLDL': 'ldl_mgdl'}, inplace=True)

    # 5. Examination: Body Measures (BMX)
    bmx = pd.read_sas(os.path.join(DATA_DIR, f'BMX_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST']].copy()
    bmx.rename(columns={
        'BMXWT': 'weight_kg',
        'BMXHT': 'height_cm',
        'BMXBMI': 'bmi',
        'BMXWAIST': 'waist_circ_cm'
    }, inplace=True)

    # 6. Blood Pressure & Resting HR (BPX)
    bpx = pd.read_sas(os.path.join(DATA_DIR, f'BPX_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'BPXPLS', 'BPXSY1', 'BPXDI1']].copy()
    bpx.rename(columns={
        'BPXPLS': 'resting_hr_bpm',
        'BPXSY1': 'systolic_bp',
        'BPXDI1': 'diastolic_bp'
    }, inplace=True)

    # 7. Diabetes Questionnaire (DIQ)
    diq = pd.read_sas(os.path.join(DATA_DIR, f'DIQ_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'DIQ010', 'DIQ050']].copy()
    diq.rename(columns={
        'DIQ010': 'diagnosed_diabetes', # 1=Yes, 2=No, 3=Borderline
        'DIQ050': 'taking_insulin'      # 1=Yes, 2=No
    }, inplace=True)

    # 8. Sleep Questionnaire (SLQ)
    slq = pd.read_sas(os.path.join(DATA_DIR, f'SLQ_{cycle_suffix}.XPT'), encoding='iso-8859-1')[['SEQN', 'SLD010H']].copy()
    slq.rename(columns={'SLD010H': 'sleep_hours_weekday'}, inplace=True)

    # 9. Wearable Accelerometry - Day Level (PAXDAY)
    paxday = pd.read_sas(os.path.join(DATA_DIR, f'PAXDAY_{cycle_suffix}.XPT'), encoding='iso-8859-1')
    # Aggregate 7 days into per-participant wearable summaries
    pax_summary = paxday.groupby('SEQN').agg(
        valid_wear_days=('PAXDAYD', 'count'),
        mean_daily_activity_counts=('PAXAISMD', 'mean'),
        mean_daily_triaxial_mims=('PAXMTSD', 'mean'),
        mean_daily_wake_wear_min=('PAXWWMD', 'mean'),
        mean_daily_sleep_wear_min=('PAXSWMD', 'mean'),
        mean_daily_non_wear_min=('PAXNWMD', 'mean'),
        mean_daily_lux=('PAXLXSD', 'mean')
    ).reset_index()
    print(f"Wearable summary aggregated: {len(pax_summary)} participants with 7-day actigraphy")

    # Merge all into single cycle dataframe
    df = demo.merge(glu, on='SEQN', how='inner') \
             .merge(ghb, on='SEQN', how='inner') \
             .merge(hdl, on='SEQN', how='left') \
             .merge(tchol, on='SEQN', how='left') \
             .merge(trig, on='SEQN', how='left') \
             .merge(bmx, on='SEQN', how='left') \
             .merge(bpx, on='SEQN', how='left') \
             .merge(diq, on='SEQN', how='left') \
             .merge(slq, on='SEQN', how='left') \
             .merge(pax_summary, on='SEQN', how='left')

    df['cycle'] = '2011-2012' if cycle_suffix == 'G' else '2013-2014'
    print(f"Cycle {cycle_suffix} unified rows: {len(df)}")
    return df

def build_master():
    df_g = load_and_merge_cycle('G')
    df_h = load_and_merge_cycle('H')
    master = pd.concat([df_g, df_h], ignore_index=True)
    print(f"\nTotal Unified Records Across 2011-2014: {len(master):,}")

    # Compute Gold-Standard HOMA-IR
    # Glucose in mmol/L = LBXGLU / 18.0
    # HOMA-IR = (Insulin * (Glucose / 18.0)) / 22.5
    master['fasting_glucose_mmol'] = master['fasting_glucose_mgdl'] / 18.0
    master['homa_ir'] = (master['fasting_insulin_uUml'] * master['fasting_glucose_mmol']) / 22.5

    # Triglyceride-Glucose Index (TyG) - Another powerful non-insulin IR surrogate
    # TyG = ln(fasting_triglycerides (mg/dL) * fasting_glucose (mg/dL) / 2)
    master['tyg_index'] = np.log((master['triglycerides_mgdl'] * master['fasting_glucose_mgdl']) / 2.0)

    # 3-Class Metabolic Risk Classification
    # Low Risk (Insulin Sensitive): HOMA-IR < 2.0 AND HbA1c < 5.7%
    # Moderate Risk (Prediabetes / Borderline): (HOMA-IR between 2.0 and 2.9) OR (HbA1c 5.7% - 6.4%)
    # High Risk (Insulin Resistant / Diabetic Dysglycemia): HOMA-IR >= 3.0 OR HbA1c >= 6.5%
    def assign_risk_class(row):
        homa = row['homa_ir']
        a1c = row['hba1c_pct']
        if pd.isna(homa) or pd.isna(a1c):
            return np.nan
        if homa >= 3.0 or a1c >= 6.5:
            return 'High_Risk_IR'
        elif homa >= 2.0 or a1c >= 5.7:
            return 'Moderate_Risk_Prediabetes'
        else:
            return 'Low_Risk_Normal'

    master['metabolic_risk_class'] = master.apply(assign_risk_class, axis=1)

    # Binary IR Label (HOMA-IR >= 2.6 as used in Google study and endocrinology consensus)
    master['is_insulin_resistant'] = master['homa_ir'].apply(lambda x: 1 if x >= 2.6 else (0 if pd.notna(x) else np.nan))

    # Adult Cohort Filtering (Ages 18-65, non-pregnant, non-diagnosed T1D)
    master['is_target_adult'] = (
        (master['age'] >= 18) &
        (master['age'] <= 65) &
        (master['diagnosed_diabetes'] != 1) &  # Exclude self-reported known diabetics
        (master['taking_insulin'] != 1) &
        (master['homa_ir'].notna())
    )

    out_csv = os.path.join(DATA_DIR, 'nhanes_unified_master.csv')
    out_parquet = os.path.join(DATA_DIR, 'nhanes_unified_master.parquet')
    master.to_csv(out_csv, index=False)
    master.to_parquet(out_parquet, index=False)
    print(f"\nSaved master dataset to:")
    print(f"  • {out_csv} ({os.path.getsize(out_csv) / (1024*1024):.2f} MB)")
    print(f"  • {out_parquet} ({os.path.getsize(out_parquet) / (1024*1024):.2f} MB)")

    print("\n--- Summary Statistics on Adult Target Cohort (Ages 18-65, Non-Diabetic) ---")
    cohort = master[master['is_target_adult']].copy()
    print(f"Usable Target Participants: {len(cohort):,}")
    print(f"Wearable Monitored Participants: {cohort['valid_wear_days'].notna().sum():,}")
    print("\nMetabolic Risk Class Distribution:")
    print(cohort['metabolic_risk_class'].value_counts(dropna=False))
    print(f"\nBinary Insulin Resistance (HOMA-IR >= 2.6):")
    print(cohort['is_insulin_resistant'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))
    print("\nHOMA-IR Summary:")
    print(cohort['homa_ir'].describe())
    print("\nHbA1c Summary:")
    print(cohort['hba1c_pct'].describe())

if __name__ == '__main__':
    build_master()
