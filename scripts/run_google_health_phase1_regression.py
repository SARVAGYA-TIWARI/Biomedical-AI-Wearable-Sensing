"""
Phase 1: Multimodal Continuous HOMA-IR Estimation & Regression Benchmark
Using Google Health Nature 2026 Dataset (N = 1,165 Participants)
Evaluates: Wearables Alone vs Labs Alone vs Full Multimodal Fusion
Metrics: R^2, MAE, RMSE, Pearson r, Spearman rho, and Bland-Altman Agreement
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

def run_phase1():
    print("=" * 78, flush=True)
    print("PHASE 1: GOOGLE HEALTH MULTIMODAL CONTINUOUS HOMA-IR REGRESSION BENCHMARK", flush=True)
    print("=" * 78, flush=True)
    
    # 1. Load Data
    data_path = os.path.join("data", "google_health_ir", "ir_prediction_data.pq")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Loaded dataset: {df.shape[0]} participants, {df.shape[1]} features", flush=True)
    
    # Target
    y_raw = df['homa_ir'].values
    y_log = np.log(np.clip(y_raw, 0.1, 100.0))
    
    # Encode Sex (Male: 1, Female: 0)
    sex_binary = (df['sex'].astype(str).str.lower().str.startswith('m')).astype(float)
    
    # 2. Define Feature Groups
    wearable_cols = [
        'RHR_FreeLiving_mean', 'RHR_FreeLiving_median', 'RHR_FreeLiving_std',
        'HRV_FreeLiving_mean', 'HRV_FreeLiving_median', 'HRV_FreeLiving_std',
        'STEPS_Daily_mean', 'STEPS_Daily_median', 'STEPS_Daily_std',
        'SLEEP_Duration_mean', 'SLEEP_Duration_median', 'SLEEP_Duration_std'
    ]
    
    anthro_vitals_cols = [
        'age', 'bmi', 'waist_circumference', 'height', 'weight',
        'systolic_bp', 'diastolic_bp'
    ]
    
    routine_lab_cols = [
        'total cholesterol', 'hdl', 'triglycerides', 'ldl', 'chol/hdl', 'non hdl',
        'glucose', 'creatinine', 'egfr', 'bun', 'alt', 'ast', 'hba1c', 'crp'
    ]
    
    # Engineered interaction features
    df_imputed = df.copy()
    for col in set(wearable_cols + anthro_vitals_cols + routine_lab_cols):
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            
    df_imputed['sex_binary'] = sex_binary
    
    # Domain Ratios
    df_imputed['WHtR'] = df_imputed['waist_circumference'] / df_imputed['height'].replace(0, np.nan)
    df_imputed['WHtR'] = df_imputed['WHtR'].fillna(df_imputed['WHtR'].median())
    
    df_imputed['MAP'] = df_imputed['diastolic_bp'] + (df_imputed['systolic_bp'] - df_imputed['diastolic_bp']) / 3.0
    df_imputed['Pulse_Pressure'] = df_imputed['systolic_bp'] - df_imputed['diastolic_bp']
    
    df_imputed['Autonomic_Stress_Ratio'] = df_imputed['RHR_FreeLiving_mean'] / (df_imputed['HRV_FreeLiving_mean'] + 1e-4)
    df_imputed['TG_HDL_Ratio'] = df_imputed['triglycerides'] / (df_imputed['hdl'] + 1e-4)
    
    engineered_cols = ['WHtR', 'MAP', 'Pulse_Pressure', 'Autonomic_Stress_Ratio', 'TG_HDL_Ratio']
    
    # Modality Dictionaries
    modalities = {
        'Wearables Alone (PPG, HRV, Steps, Sleep)': wearable_cols,
        'Wearables + Anthropometrics & Vitals': wearable_cols + anthro_vitals_cols + ['sex_binary', 'WHtR', 'MAP', 'Autonomic_Stress_Ratio'],
        'Routine Blood Labs Alone (Lipids, Glucose, HbA1c)': routine_lab_cols + ['TG_HDL_Ratio'],
        'Full Multimodal Fusion (Wearables + Anthro + Labs)': wearable_cols + anthro_vitals_cols + ['sex_binary'] + routine_lab_cols + engineered_cols
    }
    
    # 3. Model Benchmark Definitions
    def get_models(random_state=42):
        return {
            'Ridge Regression': Ridge(alpha=10.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1, verbosity=0),
            'Multimodal MLP Encoder': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01, max_iter=150, random_state=random_state)
        }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    
    best_preds_dict = {}
    
    print("\n--- Running 5-Fold Cross-Validation across All Modalities ---", flush=True)
    
    for mod_name, mod_feats in modalities.items():
        print(f"\nEvaluating Modality: {mod_name} ({len(mod_feats)} features)", flush=True)
        X_mat = df_imputed[mod_feats].values
        
        models = get_models(random_state=42)
        
        # Add Stacked Ensemble (Fast 3-fold inner CV)
        estimators = [
            ('ridge', Ridge(alpha=10.0)),
            ('rf', RandomForestRegressor(n_estimators=60, max_depth=5, random_state=42, n_jobs=-1)),
            ('lgb', lgb.LGBMRegressor(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0))
        ]
        models['Stacked Multi-Model Ensemble'] = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
        
        for model_name, model in models.items():
            oof_preds = np.zeros(len(y_raw))
            
            for train_idx, test_idx in kf.split(X_mat):
                X_train, X_test = X_mat[train_idx], X_mat[test_idx]
                y_train_log = y_log[train_idx]
                
                scaler = RobustScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                model.fit(X_train_s, y_train_log)
                pred_log = model.predict(X_test_s)
                oof_preds[test_idx] = np.exp(pred_log)  # Transform back to original scale
                
            # Compute Metrics on original scale
            r2_val = r2_score(y_raw, oof_preds)
            mae_val = mean_absolute_error(y_raw, oof_preds)
            rmse_val = np.sqrt(mean_squared_error(y_raw, oof_preds))
            pearson_r, _ = stats.pearsonr(y_raw, oof_preds)
            spearman_rho, _ = stats.spearmanr(y_raw, oof_preds)
            
            # Bland-Altman statistics
            diff = oof_preds - y_raw
            bias = np.mean(diff)
            sd_diff = np.std(diff)
            loa_lower = bias - 1.96 * sd_diff
            loa_upper = bias + 1.96 * sd_diff
            
            benchmark_results.append({
                'Modality': mod_name,
                'Model': model_name,
                'R2': r2_val,
                'Pearson_r': pearson_r,
                'Spearman_rho': spearman_rho,
                'MAE': mae_val,
                'RMSE': rmse_val,
                'Bias': bias,
                'LOA_Lower': loa_lower,
                'LOA_Upper': loa_upper
            })
            
            print(f"  [{model_name:30s}] R²: {r2_val:.4f} | r: {pearson_r:.4f} | rho: {spearman_rho:.4f} | MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f}", flush=True)
            
            if mod_name.startswith('Full Multimodal') and model_name == 'Stacked Multi-Model Ensemble':
                best_preds_dict['multimodal_stacked'] = oof_preds
            elif mod_name.startswith('Wearables Alone') and model_name == 'Stacked Multi-Model Ensemble':
                best_preds_dict['wearables_stacked'] = oof_preds
            elif mod_name.startswith('Routine Blood Labs') and model_name == 'Stacked Multi-Model Ensemble':
                best_preds_dict['labs_stacked'] = oof_preds
                
    results_df = pd.DataFrame(benchmark_results)
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "google_health_phase1_regression_benchmark.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}", flush=True)
    
    # 4. Generate Publication Figures
    os.makedirs("figures", exist_ok=True)
    
    # Figure 1: Modality & Model Comparison
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    
    stacked_subset = results_df[results_df['Model'] == 'Stacked Multi-Model Ensemble'].sort_values('R2', ascending=True)
    
    colors = ['#3498db', '#2980b9', '#f39c12', '#27ae60']
    bars = plt.barh(stacked_subset['Modality'], stacked_subset['R2'], color=colors, height=0.55, edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2.0, f"R² = {w:.4f}", va='center', fontweight='bold', fontsize=11)
        
    plt.title("Google Health Dataset: Continuous HOMA-IR Estimation Across Input Modalities\n(5-Fold Cross-Validation, Stacked Ensemble)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Coefficient of Determination (Out-of-Sample R²)", fontsize=11, labelpad=8)
    plt.xlim(0, 0.60)
    plt.tight_layout()
    fig1_path = os.path.join("figures", "google_health_phase1_model_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {fig1_path}", flush=True)
    
    # Figure 2: Bland-Altman Agreement Plot for Best Multimodal Model
    plt.figure(figsize=(9, 6))
    pred_best = best_preds_dict['multimodal_stacked']
    mean_val = (y_raw + pred_best) / 2.0
    diff_val = pred_best - y_raw
    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    plt.scatter(mean_val, diff_val, alpha=0.45, color='#2c3e50', edgecolors='none', s=28)
    plt.axhline(mean_diff, color='#e74c3c', linestyle='-', linewidth=2.0, label=f'Mean Bias: {mean_diff:.2f}')
    plt.axhline(upper_loa, color='#2980b9', linestyle='--', linewidth=1.8, label=f'+1.96 SD: {upper_loa:.2f}')
    plt.axhline(lower_loa, color='#2980b9', linestyle='--', linewidth=1.8, label=f'-1.96 SD: {lower_loa:.2f}')
    
    plt.title("Bland-Altman Agreement Analysis: Multimodal Stacked Ensemble vs Gold-Standard HOMA-IR", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Mean of Measured & Predicted HOMA-IR", fontsize=11)
    plt.ylabel("Difference (Predicted - Measured HOMA-IR)", fontsize=11)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig2_path = os.path.join("figures", "google_health_phase1_bland_altman.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {fig2_path}", flush=True)
    
    # Figure 3: Correlation Scatter Hexbin Plot
    plt.figure(figsize=(8, 7))
    plt.hexbin(y_raw, pred_best, gridsize=35, cmap='Blues', mincnt=1)
    cb = plt.colorbar(label='Participant Density')
    
    max_lim = max(np.percentile(y_raw, 99), np.percentile(pred_best, 99)) + 1.0
    plt.plot([0, max_lim], [0, max_lim], 'r--', linewidth=2.0, label='Line of Identity (y = x)')
    
    r_val = results_df.loc[(results_df['Modality'].str.startswith('Full Multimodal')) & (results_df['Model'] == 'Stacked Multi-Model Ensemble'), 'Pearson_r'].values[0]
    r2_val = results_df.loc[(results_df['Modality'].str.startswith('Full Multimodal')) & (results_df['Model'] == 'Stacked Multi-Model Ensemble'), 'R2'].values[0]
    mae_val = results_df.loc[(results_df['Modality'].str.startswith('Full Multimodal')) & (results_df['Model'] == 'Stacked Multi-Model Ensemble'), 'MAE'].values[0]
    
    plt.text(0.05, 0.90, f"Pearson r = {r_val:.4f}\nR² = {r2_val:.4f}\nMAE = {mae_val:.4f}", transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.85, edgecolor='#34495e'))
    
    plt.title("Predicted vs Laboratory Measured HOMA-IR\n(Google Health Multimodal Cohort, N = 1,165)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Certified Laboratory Fasting HOMA-IR", fontsize=11)
    plt.ylabel("Multimodal Ensemble Predicted HOMA-IR", fontsize=11)
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join("figures", "google_health_phase1_scatter_hexbin.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Generated Figure 3: {fig3_path}", flush=True)
    
    print("\n--- PHASE 1 BENCHMARK SUMMARY ---", flush=True)
    summary_table = results_df[results_df['Model'] == 'Stacked Multi-Model Ensemble'][['Modality', 'R2', 'Pearson_r', 'Spearman_rho', 'MAE', 'RMSE']]
    print(summary_table.to_string(index=False), flush=True)

if __name__ == '__main__':
    run_phase1()
