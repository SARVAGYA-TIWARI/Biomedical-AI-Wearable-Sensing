import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'nhanes')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print("=" * 75)
print("EXPLAINABLE AI & CLINICAL DECISION SUITE: SHAP INTERPRETABILITY")
print("=" * 75)

# 1. Load Data
parquet_path = os.path.join(DATA_DIR, 'nhanes_unified_master_with_circadian.parquet')
df = pd.read_parquet(parquet_path)
cohort = df[df['is_target_adult'] & df['circadian_mesor'].notna()].copy()
cohort = cohort[cohort['metabolic_risk_class'].notna()].copy()

cohort['acrophase_sin'] = np.sin(2 * np.pi * cohort['circadian_acrophase'] / 24.0)
cohort['acrophase_cos'] = np.cos(2 * np.pi * cohort['circadian_acrophase'] / 24.0)

feature_cols = [
    # Wearable & Circadian Dynamics
    'circadian_mesor', 'circadian_amplitude', 'circadian_acrophase', 'circadian_r2',
    'acrophase_sin', 'acrophase_cos',
    'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
    'm10_value', 'l5_value',
    'mean_nightly_sleep_hours', 'max_sedentary_bout_hours', 'valid_wear_days',
    'mean_daily_activity_counts', 'mean_daily_triaxial_mims',
    'mean_daily_wake_wear_min', 'mean_daily_sleep_wear_min',
    
    # Autonomic Vitals & Anthropometrics (Non-Invasive)
    'resting_hr_bpm', 'bmi', 'waist_circ_cm', 'systolic_bp', 'diastolic_bp',
    
    # Demographics
    'age', 'gender', 'ethnicity', 'poverty_ratio'
]

feature_display_names = {
    'circadian_mesor': 'Circadian Mesor (Activity Baseline)',
    'circadian_amplitude': 'Circadian Amplitude (Peak-Trough)',
    'circadian_acrophase': 'Circadian Acrophase (Peak Time h)',
    'circadian_r2': 'Cosinor Fit Goodness (R²)',
    'acrophase_sin': 'Acrophase Sine Harmonic',
    'acrophase_cos': 'Acrophase Cosine Harmonic',
    'interdaily_stability_IS': 'Interdaily Stability (IS - Routine)',
    'intradaily_variability_IV': 'Intradaily Variability (IV - Frag)',
    'relative_amplitude_RA': 'Relative Circadian Amplitude (RA)',
    'm10_value': 'M10 (10 Most Active Hours MIMS)',
    'l5_value': 'L5 (5 Least Active Hours / Sleep MIMS)',
    'mean_nightly_sleep_hours': 'Nightly Sleep Duration (Hours)',
    'max_sedentary_bout_hours': 'Max Sedentary Bout (Hours)',
    'valid_wear_days': 'Valid Sensor Wear Days',
    'mean_daily_activity_counts': 'Mean Daily Activity Counts',
    'mean_daily_triaxial_mims': 'Triaxial Movement MIMS',
    'mean_daily_wake_wear_min': 'Daily Wake Wear Minutes',
    'mean_daily_sleep_wear_min': 'Daily Sleep Wear Minutes',
    'resting_hr_bpm': 'Resting Heart Rate (BPM)',
    'bmi': 'Body Mass Index (BMI kg/m²)',
    'waist_circ_cm': 'Waist Circumference (cm)',
    'systolic_bp': 'Systolic Blood Pressure (mmHg)',
    'diastolic_bp': 'Diastolic Blood Pressure (mmHg)',
    'age': 'Chronological Age (Years)',
    'gender': 'Biological Sex',
    'ethnicity': 'Race / Ethnicity',
    'poverty_ratio': 'Poverty Income Ratio (SES)'
}

X = cohort[feature_cols].copy()
for col in X.columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

class_mapping = {
    'Low_Risk_Normal': 0,
    'Moderate_Risk_Prediabetes': 1,
    'High_Risk_IR': 2
}
y_clf = cohort['metabolic_risk_class'].map(class_mapping).values
y_reg = cohort['homa_ir'].values

X_display = X.rename(columns=feature_display_names)

print(f"Cohort Size: {len(X)} | Features: {X.shape[1]}")

# 2. Train Models for Interpretation
print("\n--- Training Interpretable Tree Ensembles ---")
X_tr, X_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
    X_display, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# A. XGBoost Regressor for Continuous HOMA-IR
xgb_reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_reg.fit(X_tr, y_reg_tr)

# B. XGBoost Classifier for 3-Class Risk Screening
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    objective='multi:softprob', num_class=3
)
xgb_clf.fit(X_tr, y_clf_tr)
print("Trained XGBoost Regressor and Classifier successfully.")

# 3. Compute Tree SHAP Values
print("\n--- Computing Tree SHAP Values via shap.TreeExplainer ---")
explainer_reg = shap.TreeExplainer(xgb_reg)
shap_values_reg = explainer_reg(X_te)

explainer_clf = shap.TreeExplainer(xgb_clf)
shap_values_clf = explainer_clf(X_te)

print(f"SHAP Values Computed for Regression: {shap_values_reg.shape}")
print(f"SHAP Values Computed for 3-Class Classification: {shap_values_clf.shape}")

# Save Mean Absolute SHAP values to CSV
mean_abs_shap = np.mean(np.abs(shap_values_reg.values), axis=0)
shap_ranking_df = pd.DataFrame({
    'Feature': X_display.columns,
    'Mean_Abs_SHAP_HOMAIR': mean_abs_shap
}).sort_values('Mean_Abs_SHAP_HOMAIR', ascending=False)
shap_csv_path = os.path.join(RES_DIR, 'shap_feature_importance.csv')
shap_ranking_df.to_csv(shap_csv_path, index=False)
print(f"Saved SHAP importance ranking to: {shap_csv_path}")

# 4. Global Summary Beeswarm Plot (Continuous HOMA-IR)
print("\n--- Generating Figure 1: Global SHAP Summary Beeswarm Plot ---")
plt.figure(figsize=(12, 9))
shap.summary_plot(shap_values_reg, X_te, max_display=15, show=False)
plt.title("Global SHAP Value Distribution: Continuous HOMA-IR Biomarker Prediction\nImpact of Wearable Circadian & Autonomic Predictors", fontsize=13, fontweight='bold', pad=15, color='#1F4E79')
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, 'shap_global_beeswarm_homa.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# 5. Multi-Class SHAP Summary Bar Plot (3-Class Risk Screening)
print("\n--- Generating Figure 2: Multiclass SHAP Summary Bar Plot ---")
plt.figure(figsize=(12, 8))
# shap_values_clf: (N, features, 3)
shap.summary_plot(
    [shap_values_clf.values[:, :, 0], shap_values_clf.values[:, :, 1], shap_values_clf.values[:, :, 2]],
    X_te,
    class_names=['Class 0 (Normal)', 'Class 1 (Prediabetes)', 'Class 2 (Insulin Resistant)'],
    max_display=12,
    show=False
)
plt.title("Multiclass Feature Importance: 3-Tier Metabolic Risk Stratification\nRelative Impact on Low Risk, Prediabetes, and Frank Insulin Resistance", fontsize=13, fontweight='bold', pad=15, color='#1F4E79')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'shap_multiclass_bar_screening.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# 6. Feature Interaction Dependence Plots
print("\n--- Generating Figure 3: Clinical Feature Interaction Dependence Plots ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Interaction 1: BMI vs. Relative Circadian Amplitude (RA)
ax1 = axes[0]
bmi_idx = list(X_display.columns).index('Body Mass Index (BMI kg/m²)')
ra_idx = list(X_display.columns).index('Relative Circadian Amplitude (RA)')

scatter1 = ax1.scatter(
    X_te.iloc[:, bmi_idx],
    shap_values_reg.values[:, bmi_idx],
    c=X_te.iloc[:, ra_idx],
    cmap='viridis',
    alpha=0.8,
    edgecolor='none',
    s=35
)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Relative Circadian Amplitude (RA)', fontsize=10, fontweight='bold')
ax1.axhline(0, color='grey', linestyle='--', lw=1)
ax1.set_xlabel('Body Mass Index (BMI kg/m²)', fontsize=11, fontweight='bold')
ax1.set_ylabel('SHAP Value for BMI (Impact on HOMA-IR)', fontsize=11, fontweight='bold')
ax1.set_title('Feature Interaction 1: BMI vs. Circadian Robustness (RA)\nHigh Rhythmicity Buffers Metabolic Elevation', fontsize=11, fontweight='bold', color='#1F4E79')
ax1.grid(True, linestyle=':', alpha=0.6)

# Interaction 2: Waist Circumference vs. Nocturnal Activity L5
ax2 = axes[1]
waist_idx = list(X_display.columns).index('Waist Circumference (cm)')
l5_idx = list(X_display.columns).index('L5 (5 Least Active Hours / Sleep MIMS)')

scatter2 = ax2.scatter(
    X_te.iloc[:, waist_idx],
    shap_values_reg.values[:, waist_idx],
    c=X_te.iloc[:, l5_idx],
    cmap='plasma',
    alpha=0.8,
    edgecolor='none',
    s=35
)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('L5 Nocturnal Movement (Sleep Restlessness)', fontsize=10, fontweight='bold')
ax2.axhline(0, color='grey', linestyle='--', lw=1)
ax2.set_xlabel('Waist Circumference (cm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('SHAP Value for Waist Circumference', fontsize=11, fontweight='bold')
ax2.set_title('Feature Interaction 2: Waist Circumference vs. Nocturnal L5\nCompounding Risk of Central Adiposity & Sleep Fragmentation', fontsize=11, fontweight='bold', color='#1F4E79')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, 'shap_clinical_feature_interactions.png')
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}")

# 7. Patient-Level Clinical Case Studies (Waterfall Visualizations)
print("\n--- Generating Figure 4: Patient-Level Waterfall Case Studies ---")
# Select 3 representative patient archetypes from test set
# Archetype 1: Healthy / Low Risk
normal_indices = np.where((y_clf_te == 0) & (y_reg_te < 1.5))[0]
idx_healthy = normal_indices[0]

# Archetype 2: Borderline Prediabetes / Silent Risk (Normal BMI but disrupted rhythm)
prediab_indices = np.where((y_clf_te == 1) & (X_te['Body Mass Index (BMI kg/m²)'] < 27))[0]
idx_prediab = prediab_indices[0] if len(prediab_indices) > 0 else np.where(y_clf_te == 1)[0][0]

# Archetype 3: High Risk Insulin Resistant
ir_indices = np.where((y_clf_te == 2) & (y_reg_te > 4.5))[0]
idx_ir = ir_indices[0]

fig, axes = plt.subplots(3, 1, figsize=(15, 12))
patient_indices = [idx_healthy, idx_prediab, idx_ir]
patient_titles = [
    f"PATIENT CASE 1: Clinically Healthy / Normal Glycemic Control\nTrue HOMA-IR: {y_reg_te[idx_healthy]:.2f} | Predicted: {xgb_reg.predict(X_te.iloc[[idx_healthy]])[0]:.2f} | Ground Truth: Class 0 (Low Risk)",
    f"PATIENT CASE 2: Borderline Prediabetes with Circadian Disruption\nTrue HOMA-IR: {y_reg_te[idx_prediab]:.2f} | Predicted: {xgb_reg.predict(X_te.iloc[[idx_prediab]])[0]:.2f} | Ground Truth: Class 1 (Prediabetes)",
    f"PATIENT CASE 3: Severe Insulin Resistance & High Cardiometabolic Risk\nTrue HOMA-IR: {y_reg_te[idx_ir]:.2f} | Predicted: {xgb_reg.predict(X_te.iloc[[idx_ir]])[0]:.2f} | Ground Truth: Class 2 (High Risk IR)"
]
box_colors = ['#EBF1F5', '#FFF2CC', '#FCE4D6']
border_colors = ['#1F4E79', '#B25900', '#C00000']

for i, p_idx in enumerate(patient_indices):
    ax = axes[i]
    # Get top 7 features driving this individual's prediction
    vals = shap_values_reg.values[p_idx]
    top_feat_idx = np.argsort(np.abs(vals))[-7:]
    
    y_pos = np.arange(len(top_feat_idx))
    feat_names = [X_display.columns[j] for j in top_feat_idx]
    shap_vals = [vals[j] for j in top_feat_idx]
    bar_colors = ['#C00000' if v > 0 else '#2E75B6' for v in shap_vals]
    
    ax.barh(y_pos, shap_vals, color=bar_colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=10, fontweight='bold')
    ax.axvline(0, color='black', lw=1.2)
    ax.set_title(patient_titles[i], fontsize=11, fontweight='bold', color=border_colors[i], loc='left')
    ax.set_xlabel("SHAP Impact on Predicted HOMA-IR (Red = Elevates Risk | Blue = Lowers Risk)", fontsize=10)
    ax.set_facecolor(box_colors[i])
    
    # Add data annotations
    for b_idx, (y, val) in enumerate(zip(y_pos, shap_vals)):
        orig_val = X_te.iloc[p_idx, top_feat_idx[b_idx]]
        text_str = f" Val = {orig_val:.1f} (Δ {val:+.2f})"
        ax.text(val + (0.02 if val >= 0 else -0.02), y, text_str, va='center',
                ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold',
                color='#7F0000' if val >= 0 else '#003366')

plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, 'shap_patient_case_studies.png')
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved: {fig4_path}")

print("\n--- SHAP Interpretability Suite Execution Complete ---")
