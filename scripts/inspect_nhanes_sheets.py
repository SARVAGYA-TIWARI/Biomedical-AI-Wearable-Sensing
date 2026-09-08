import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nhanes')

files_to_describe = [
    ("DEMO_G.XPT", "Demographics (Age, Sex, Race, Socioeconomic)", [
        ('SEQN', 'Participant ID (Merge Key across all tables)'),
        ('RIDAGEYR', 'Age in years at screening (18-80+)'),
        ('RIAGENDR', 'Gender (1=Male, 2=Female)'),
        ('RIDRETH3', 'Race/Hispanic origin w/ NH Asian'),
        ('DMDEDUC2', 'Education level (Adults 20+)'),
        ('INDFMPIR', 'Ratio of family income to poverty line')
    ]),
    ("GLU_G.XPT", "Fasting Glucose & Insulin (Gold-Standard Lab)", [
        ('SEQN', 'Participant ID'),
        ('LBXGLU', 'Fasting Plasma Glucose (mg/dL)'),
        ('LBDGLUSI', 'Fasting Plasma Glucose (mmol/L)'),
        ('LBXIN', 'Fasting Serum Insulin (uU/mL)'),
        ('WTSAF2YR', 'Fasting Subsample 2-Year Mobile Exam Center Weight')
    ]),
    ("GHB_G.XPT", "Glycohemoglobin / HbA1c (Long-Term Glycemia Lab)", [
        ('SEQN', 'Participant ID'),
        ('LBXGH', 'Glycohemoglobin / HbA1c (%)')
    ]),
    ("HDL_G.XPT", "HDL Cholesterol (Cardiovascular & Metabolic Marker)", [
        ('SEQN', 'Participant ID'),
        ('LBDHDD', 'Direct HDL-Cholesterol (mg/dL)')
    ]),
    ("TRIGLY_G.XPT", "Triglycerides & LDL (Lipid Panel Lab)", [
        ('SEQN', 'Participant ID'),
        ('LBXTR', 'Triglycerides (mg/dL)'),
        ('LBDLDL', 'LDL-Cholesterol calculated (mg/dL)')
    ]),
    ("TCHOL_G.XPT", "Total Cholesterol (Cardiovascular Risk Lab)", [
        ('SEQN', 'Participant ID'),
        ('LBXTC', 'Total Cholesterol (mg/dL)')
    ]),
    ("BMX_G.XPT", "Body Measures (Anthropometric Examination)", [
        ('SEQN', 'Participant ID'),
        ('BMXWT', 'Weight (kg)'),
        ('BMXHT', 'Standing Height (cm)'),
        ('BMXBMI', 'Body Mass Index (kg/m2)'),
        ('BMXWAIST', 'Waist Circumference (cm) - Central Adiposity')
    ]),
    ("BPX_G.XPT", "Blood Pressure (Cardiovascular Autonomic Marker)", [
        ('SEQN', 'Participant ID'),
        ('BPXCHR', '60-second Pulse / Resting Heart Rate (bpm)'),
        ('BPXSY1', 'Systolic Blood Pressure (1st reading, mmHg)'),
        ('BPXDI1', 'Diastolic Blood Pressure (1st reading, mmHg)')
    ]),
    ("DIQ_G.XPT", "Diabetes Questionnaire (Medical History)", [
        ('SEQN', 'Participant ID'),
        ('DIQ010', 'Doctor ever told you have diabetes? (1=Yes, 2=No, 3=Borderline/Prediabetes)'),
        ('DIQ050', 'Taking insulin now? (1=Yes, 2=No)'),
        ('DIQ070', 'Take diabetic pills to lower blood sugar? (1=Yes, 2=No)'),
        ('DID040', 'Age when told had diabetes')
    ]),
    ("PAQ_G.XPT", "Physical Activity Questionnaire (Self-Reported)", [
        ('SEQN', 'Participant ID'),
        ('PAQ605', 'Vigorous work activity (1=Yes, 2=No)'),
        ('PAQ620', 'Moderate work activity (1=Yes, 2=No)'),
        ('PAQ650', 'Vigorous recreational activity (1=Yes, 2=No)'),
        ('PAD680', 'Minutes of sedentary activity per day')
    ]),
    ("SLQ_G.XPT", "Sleep Disorders Questionnaire", [
        ('SEQN', 'Participant ID'),
        ('SLD010H', 'Usual hours of sleep per night (weekdays/workdays)'),
        ('SLQ050', 'Ever told doctor had trouble sleeping? (1=Yes, 2=No)')
    ]),
    ("PAXHD_G.XPT", "Physical Activity Monitor - Header (Wearable Meta)", [
        ('SEQN', 'Participant ID'),
        ('PAXFDMFL', 'First Day Missing Flag'),
        ('PAXSENID', 'Accelerometer Sensor ID')
    ]),
    ("PAXDAY_G.XPT", "Physical Activity Monitor - Day Summary (7 Days Wearable)", [
        ('SEQN', 'Participant ID'),
        ('PAXDAYWD', 'Day of the week (1=Sunday ... 7=Saturday)'),
        ('PAXWDWKD', 'Weekday vs Weekend (1=Weekday, 2=Weekend)'),
        ('PAXWWD', 'Valid Day Flag (Wear time criteria met: 1=Yes, 2=No)'),
        ('PAXWWMIN', 'Wake Wear Time (minutes)'),
        ('PAXNWMIN', 'Non-wear Time (minutes)')
    ])
]

print("=" * 80)
print(f"{'NHANES 2011-2012 COMPONENT INSPECTION REPORT':^80}")
print("=" * 80)

summary_rows = []

for filename, desc, key_vars in files_to_describe:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIPPED] {filename} not yet downloaded.")
        continue
    
    df = pd.read_sas(filepath, encoding='iso-8859-1')
    n_rows, n_cols = df.shape
    n_seqn = df['SEQN'].nunique() if 'SEQN' in df.columns else 'N/A'
    
    summary_rows.append({
        'File': filename,
        'Description': desc,
        'Rows': n_rows,
        'Unique Participants': n_seqn,
        'Columns': n_cols
    })
    
    print(f"\n>>> {filename} : {desc}")
    print(f"    Dimensions: {n_rows:,} rows x {n_cols} columns | Unique SEQN: {n_seqn:,}")
    print("    Key Variables:")
    for col, explanation in key_vars:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct_non_null = (non_null / n_rows) * 100
            sample_val = df[col].dropna().iloc[0] if non_null > 0 else 'All NaN'
            print(f"      • {col:10} | Non-null: {non_null:5,} ({pct_non_null:5.1f}%) | Example: {sample_val:<8} | {explanation}")
        else:
            print(f"      • {col:10} | [NOT FOUND] | {explanation}")

print("\n" + "=" * 80)
print("OVERALL SUMMARY TABLE")
print("=" * 80)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
