"""
MESA Phase 1: Continuous Multimodal HOMA-IR Regression Benchmark (N = 2,237)
Compares:
1. Wearables Alone (Actigraphy & Sleep Telemetry)
2. Wearables + Anthropometrics (Zero Blood Non-Invasive Tier 1)
3. Routine Blood Labs Alone
4. Full Multimodal Fusion (Wearables + Anthro + Labs)
Across:
- Raw Scale HOMA-IR vs. Logarithmic ln(HOMA-IR)
- Linear (Ridge), Tree Ensembles (RF, LightGBM, XGBoost), and Stacked Meta-Regressor
Protocol: 5-Fold Cross-Validation
Metrics: R^2, Pearson r, Spearman rho, MAE, RMSE, Bland-Altman Agreement
"""

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
from scipy import stats

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mesa', 'mesa_multimodal_cohort.parquet')
RES_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 78)
print("PHASE 1: MESA MULTIMODAL CONTINUOUS HOMA-IR REGRESSION BENCHMARK")
print("=" * 78)

# 2. Load Data
df = pd.read_parquet(DATA_PATH)
print(f"Loaded MESA Cohort: {df.shape[0]} participants, {df.shape[1]} features")

y_raw = df['homa_ir'].values
y_log = df['log_homa_ir'].values

# Encode Sex: Male=1, Female=0
sex_bin = (df['sex'] == 'Male').astype(float)
df['sex_binary'] = sex_bin

# Modality Feature Lists
wearable_cols = [
    'avg_act_counts', 'active_hours_daily', 'sedentary_hours_daily',
    'sleep_duration_hr', 'sleep_efficiency_pct', 'waso_min',
    'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
    'm10_activity', 'l5_activity', 'Autonomic_Stress_Ratio'
]

anthro_cols = [
    'age', 'sex_binary', 'height', 'weight', 'bmi',
    'waist_circumference', 'WHtR', 'systolic_bp', 'diastolic_bp',
    'MAP', 'Pulse_Pressure', 'resting_hr'
]

routine_lab_cols = [
    'fasting_glucose', 'hba1c', 'total_cholesterol', 'hdl',
    'triglycerides', 'ldl', 'TG_HDL_Ratio'
]

modalities = {
    '1. Wearables Alone': wearable_cols,
    '2. Wearables + Anthropometrics (Zero Blood)': wearable_cols + anthro_cols,
    '3. Routine Blood Labs Alone': routine_lab_cols,
    '4. Full Multimodal Fusion (All Streams)': wearable_cols + anthro_cols + routine_lab_cols
}

print(f"\nFeature Sets:")
for m_name, m_feats in modalities.items():
    print(f"  {m_name:45s}: {len(m_feats)} features")

# 3. Model Suite
def get_base_models(random_state=42):
    return {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=4, random_state=random_state, n_jobs=2),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=16, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=2, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=2, verbosity=0)
    }

kf = KFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = []
oof_predictions_dict = {}

print("\n--- Running 5-Fold Cross-Validation on MESA Cohort ---", flush=True)

for mod_name, mod_feats in modalities.items():
    print(f"\n=======================================================", flush=True)
    print(f"Evaluating Modality: {mod_name} ({len(mod_feats)} features)", flush=True)
    print(f"=======================================================", flush=True)
    X_mat = df[mod_feats].values
    
    # Evaluate across models on both Raw Scale and Log Scale
    for target_scale in ['Raw HOMA-IR', 'Log ln(HOMA-IR)']:
        y_target = y_raw if target_scale == 'Raw HOMA-IR' else y_log
        
        base_models = get_base_models(random_state=42)
        model_oof_preds = {}
        
        for model_name, model in base_models.items():
            oof_preds = np.zeros(len(y_target))
            
            for train_idx, test_idx in kf.split(X_mat):
                X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
                y_tr = y_target[train_idx]
                
                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                
                model.fit(X_tr_s, y_tr)
                oof_preds[test_idx] = model.predict(X_te_s)
                
            model_oof_preds[model_name] = oof_preds

        # Construct Stacked Multi-Model Ensemble using Out-of-Fold Ridge Meta-Learner
        blend_X = np.column_stack([model_oof_preds[m] for m in base_models.keys()])
        meta_oof = np.zeros(len(y_target))
        for train_idx, test_idx in kf.split(blend_X):
            meta_ridge = Ridge(alpha=1.0)
            meta_ridge.fit(blend_X[train_idx], y_target[train_idx])
            meta_oof[test_idx] = meta_ridge.predict(blend_X[test_idx])
            
        model_oof_preds['Stacked Multi-Model Ensemble'] = meta_oof
        
        for model_name, oof_preds in model_oof_preds.items():
                
            # If target was log, also compute performance back-transformed to raw clinical scale
            if target_scale == 'Log ln(HOMA-IR)':
                # Direct metrics on log scale
                r2_log = r2_score(y_log, oof_preds)
                r_log, _ = stats.pearsonr(y_log, oof_preds)
                rho_log, _ = stats.spearmanr(y_log, oof_preds)
                mae_log = mean_absolute_error(y_log, oof_preds)
                rmse_log = np.sqrt(mean_squared_error(y_log, oof_preds))
                
                # Back-transformed to raw scale via exp
                pred_raw = np.exp(oof_preds)
                r2_back = r2_score(y_raw, pred_raw)
                r_back, _ = stats.pearsonr(y_raw, pred_raw)
                rho_back, _ = stats.spearmanr(y_raw, pred_raw)
                mae_back = mean_absolute_error(y_raw, pred_raw)
                rmse_back = np.sqrt(mean_squared_error(y_raw, pred_raw))
                
                # Bland-Altman on back-transformed
                diff = pred_raw - y_raw
                bias = np.mean(diff)
                sd_diff = np.std(diff)
                loa_lower = bias - 1.96 * sd_diff
                loa_upper = bias + 1.96 * sd_diff
                
                benchmark_results.append({
                    'Modality': mod_name,
                    'Model': model_name,
                    'Target_Training_Scale': 'Log ln(HOMA-IR)',
                    'Evaluation_Scale': 'Log Scale ln(y)',
                    'R2': round(r2_log, 4),
                    'Pearson_r': round(r_log, 4),
                    'Spearman_rho': round(rho_log, 4),
                    'MAE': round(mae_log, 4),
                    'RMSE': round(rmse_log, 4),
                    'Bias': round(np.mean(oof_preds - y_log), 4),
                    'LOA_Lower': round(np.mean(oof_preds - y_log) - 1.96 * np.std(oof_preds - y_log), 4),
                    'LOA_Upper': round(np.mean(oof_preds - y_log) + 1.96 * np.std(oof_preds - y_log), 4)
                })
                
                benchmark_results.append({
                    'Modality': mod_name,
                    'Model': model_name,
                    'Target_Training_Scale': 'Log ln(HOMA-IR)',
                    'Evaluation_Scale': 'Back-Projected Raw Scale',
                    'R2': round(r2_back, 4),
                    'Pearson_r': round(r_back, 4),
                    'Spearman_rho': round(rho_back, 4),
                    'MAE': round(mae_back, 4),
                    'RMSE': round(rmse_back, 4),
                    'Bias': round(bias, 4),
                    'LOA_Lower': round(loa_lower, 4),
                    'LOA_Upper': round(loa_upper, 4)
                })
                
                print(f"  [{target_scale:15s} | {model_name:30s}] Log R²: {r2_log:.4f} (r: {r_log:.4f}) | Back-Raw R²: {r2_back:.4f} (r: {r_back:.4f}, MAE: {mae_back:.4f})", flush=True)
                
                if model_name == 'Stacked Multi-Model Ensemble':
                    oof_predictions_dict[f"{mod_name}_log"] = oof_preds
                    oof_predictions_dict[f"{mod_name}_back_raw"] = pred_raw
                    
            else:
                # Raw target evaluation
                r2_val = r2_score(y_raw, oof_preds)
                r_val, _ = stats.pearsonr(y_raw, oof_preds)
                rho_val, _ = stats.spearmanr(y_raw, oof_preds)
                mae_val = mean_absolute_error(y_raw, oof_preds)
                rmse_val = np.sqrt(mean_squared_error(y_raw, oof_preds))
                
                diff = oof_preds - y_raw
                bias = np.mean(diff)
                sd_diff = np.std(diff)
                loa_lower = bias - 1.96 * sd_diff
                loa_upper = bias + 1.96 * sd_diff
                
                benchmark_results.append({
                    'Modality': mod_name,
                    'Model': model_name,
                    'Target_Training_Scale': 'Raw HOMA-IR',
                    'Evaluation_Scale': 'Raw Scale y',
                    'R2': round(r2_val, 4),
                    'Pearson_r': round(r_val, 4),
                    'Spearman_rho': round(rho_val, 4),
                    'MAE': round(mae_val, 4),
                    'RMSE': round(rmse_val, 4),
                    'Bias': round(bias, 4),
                    'LOA_Lower': round(loa_lower, 4),
                    'LOA_Upper': round(loa_upper, 4)
                })
                
                print(f"  [{target_scale:15s} | {model_name:30s}] Raw R²: {r2_val:.4f} | r: {r_val:.4f} | MAE: {mae_val:.4f}", flush=True)
                
                if model_name == 'Stacked Multi-Model Ensemble':
                    oof_predictions_dict[f"{mod_name}_raw"] = oof_preds

# 4. Save Benchmark Results Table
results_df = pd.DataFrame(benchmark_results)
csv_out = os.path.join(RES_DIR, 'mesa_phase1_regression_benchmark.csv')
results_df.to_csv(csv_out, index=False)
print(f"\nSaved regression benchmark to {csv_out}")

# Print Top Summary
log_eval_df = results_df[results_df['Evaluation_Scale'] == 'Log Scale ln(y)'].sort_values('R2', ascending=False)
print("\n=== TOP MODELS ON LOG SCALE ln(HOMA-IR) ===")
print(log_eval_df[['Modality', 'Model', 'R2', 'Pearson_r', 'Spearman_rho', 'MAE']].to_string(index=False))

raw_eval_df = results_df[results_df['Evaluation_Scale'] == 'Back-Projected Raw Scale'].sort_values('R2', ascending=False)
print("\n=== TOP MODELS BACK-PROJECTED TO RAW SCALE ===")
print(raw_eval_df[['Modality', 'Model', 'R2', 'Pearson_r', 'MAE', 'Bias', 'LOA_Lower', 'LOA_Upper']].to_string(index=False))

# 5. Generate Publication Figures
print("\n--- Generating Publication Visualizations for MESA Phase 1 ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Modality R² Progression Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
top_by_mod = log_eval_df[log_eval_df['Model'] == 'Stacked Multi-Model Ensemble'].sort_values('R2')
mods_clean = [m.split('. ')[1] for m in top_by_mod['Modality']]
r2_vals = top_by_mod['R2'].values
r_vals = top_by_mod['Pearson_r'].values

y_pos = np.arange(len(mods_clean))
bars = ax.barh(y_pos, r2_vals, color='#1F4E79', height=0.55, edgecolor='black', alpha=0.9)
ax.set_yticks(y_pos)
ax.set_yticklabels(mods_clean, fontsize=11, fontweight='bold')
ax.set_xlabel("Explained Variance (R² Score on ln(HOMA-IR))", fontsize=12, fontweight='bold')
ax.set_title("MESA Phase 1: Continuous HOMA-IR Explained Variance across Modalities (N = 2,237)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 0.70)

for bar, r2, r in zip(bars, r2_vals, r_vals):
    ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2,
            f"R² = {r2:.4f}  (r = {r:.4f})",
            va='center', ha='left', fontsize=10, fontweight='bold', color='#1F4E79')

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, 'mesa_phase1_modality_r2_comparison.png')
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"  Saved Figure 1: {fig1_path}")

# Figure 2: Hexbin Actual vs Predicted Scatter on Full Multimodal
best_pred_raw = oof_predictions_dict['4. Full Multimodal Fusion (All Streams)_back_raw']
fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(y_raw, best_pred_raw, gridsize=45, cmap='Blues', mincnt=1)
ax.plot([0, 25], [0, 25], 'r--', linewidth=2, label='Identity Line (y = x)')
ax.set_xlabel("Ground-Truth Fasting HOMA-IR (Certified Labs)", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted HOMA-IR (Full Multimodal Stacked Ensemble)", fontsize=12, fontweight='bold')
ax.set_title("MESA Multimodal Continuous Estimation: Ground Truth vs. Predicted (N = 2,237)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 22)
ax.set_ylim(0, 22)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Participant Density', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fontsize=11)
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'mesa_phase1_scatter_hexbin.png')
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"  Saved Figure 2: {fig2_path}")

# Figure 3: Bland-Altman Agreement Plot
diff = best_pred_raw - y_raw
mean_val = (best_pred_raw + y_raw) / 2.0
bias = np.mean(diff)
sd = np.std(diff)
loa_upper = bias + 1.96 * sd
loa_lower = bias - 1.96 * sd

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(mean_val, diff, alpha=0.4, color='#2E75B6', edgecolors='none', s=25)
ax.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Mean Bias: {bias:+.2f}')
ax.axhline(loa_upper, color='black', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {loa_upper:+.2f}')
ax.axhline(loa_lower, color='black', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {loa_lower:+.2f}')
ax.set_xlabel("Mean of Measured & Predicted HOMA-IR", fontsize=12, fontweight='bold')
ax.set_ylabel("Error / Difference (Predicted - Measured)", fontsize=12, fontweight='bold')
ax.set_title("Bland-Altman Agreement: MESA Multimodal HOMA-IR Estimation (N = 2,237)", fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, fontsize=11)
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, 'mesa_phase1_bland_altman.png')
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"  Saved Figure 3: {fig3_path}")

print("\nMESA Phase 1 Benchmark Completed Successfully!")
