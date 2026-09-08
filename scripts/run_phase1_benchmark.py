import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'nhanes')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

def load_data():
    parquet_path = os.path.join(DATA_DIR, 'nhanes_unified_master_with_circadian.parquet')
    df = pd.read_parquet(parquet_path)
    
    # Target adult cohort with circadian data
    cohort = df[df['is_target_adult'] & df['circadian_mesor'].notna()].copy()
    print(f"Loaded analytical cohort: {len(cohort)} participants with full multi-modal data.")
    return cohort

def prepare_features_and_targets(cohort):
    # Sinusoidal circadian harmonics (SweetDeep encoding)
    cohort['acrophase_sin'] = np.sin(2 * np.pi * cohort['circadian_acrophase'] / 24.0)
    cohort['acrophase_cos'] = np.cos(2 * np.pi * cohort['circadian_acrophase'] / 24.0)

    # Feature List
    feature_cols = [
        # Wearable & Circadian Dynamics
        'circadian_mesor', 'circadian_amplitude', 'circadian_acrophase', 'circadian_r2',
        'acrophase_sin', 'acrophase_cos',
        'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
        'm10_value', 'l5_value',
        'mean_nightly_sleep_hours', 'max_sedentary_bout_hours', 'valid_wear_days',
        'mean_daily_activity_counts', 'mean_daily_triaxial_mims',
        'mean_daily_wake_wear_min', 'mean_daily_sleep_wear_min',
        
        # Autonomic Vitals & Anthropometrics
        'resting_hr_bpm', 'bmi', 'waist_circ_cm', 'systolic_bp', 'diastolic_bp',
        
        # Demographics
        'age', 'gender', 'ethnicity', 'poverty_ratio'
    ]

    target_cols = ['homa_ir', 'hba1c_pct', 'fasting_glucose_mgdl']
    
    # Impute median for missing values in feature set (under 5% in vitals/poverty)
    X = cohort[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    # Stratification label based on metabolic risk class
    strat_label = cohort['metabolic_risk_class'].fillna('Low_Risk_Normal').values

    return X, cohort, feature_cols, target_cols, strat_label

def get_models():
    return {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=0.05, max_iter=2000),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=8, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.01, max_iter=200, random_state=42, early_stopping=True)
    }

def evaluate_bland_altman(y_true, y_pred):
    diff = y_pred - y_true
    mean_bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    lower_loa = float(mean_bias - 1.96 * sd_diff)
    upper_loa = float(mean_bias + 1.96 * sd_diff)
    within_loa = float(np.mean((diff >= lower_loa) & (diff <= upper_loa)) * 100.0)
    return {
        'mean_bias': mean_bias,
        'sd_diff': sd_diff,
        'lower_loa': lower_loa,
        'upper_loa': upper_loa,
        'pct_within_loa': within_loa
    }

def run_benchmark():
    print("=" * 75)
    print("PHASE 1 BENCHMARK: CONTINUOUS METABOLIC BIOMARKER REGRESSION")
    print("=" * 75)

    cohort = load_data()
    X, full_cohort, feature_cols, target_cols, strat_label = prepare_features_and_targets(cohort)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    all_oof_predictions = {}

    target_units = {
        'homa_ir': 'Score',
        'hba1c_pct': '%',
        'fasting_glucose_mgdl': 'mg/dL'
    }

    target_display_names = {
        'homa_ir': 'HOMA-IR (Insulin Resistance)',
        'hba1c_pct': 'HbA1c Glycohemoglobin',
        'fasting_glucose_mgdl': 'Fasting Plasma Glucose'
    }

    for target in target_cols:
        print(f"\n>>> Benchmarking Target: {target_display_names[target]} ({target}) <<<")
        y = full_cohort[target].values
        valid_mask = ~np.isnan(y)
        
        # Subsample to non-NaN targets
        X_curr = X[valid_mask].copy()
        y_curr = y[valid_mask]
        strat_curr = strat_label[valid_mask]
        
        models = get_models()
        all_oof_predictions[target] = {}

        for model_name, model in models.items():
            t0 = time.time()
            oof_preds = np.zeros(len(y_curr))
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_curr, strat_curr)):
                X_train, y_train = X_curr.iloc[train_idx], y_curr[train_idx]
                X_val, y_val = X_curr.iloc[val_idx], y_curr[val_idx]
                
                # Preprocessing: Robust scaling to protect against extreme sensor outliers
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Fit
                if model_name in ['Ridge', 'Lasso', 'Neural Network (MLP)']:
                    model.fit(X_train_scaled, y_train)
                    oof_preds[val_idx] = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    oof_preds[val_idx] = model.predict(X_val)

            elapsed = time.time() - t0
            all_oof_predictions[target][model_name] = oof_preds

            # Metrics
            mae = float(np.mean(np.abs(oof_preds - y_curr)))
            rmse = float(np.sqrt(np.mean((oof_preds - y_curr) ** 2)))
            r, _ = stats.pearsonr(y_curr, oof_preds)
            ss_res = np.sum((y_curr - oof_preds) ** 2)
            ss_tot = np.sum((y_curr - np.mean(y_curr)) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot))

            # Bland-Altman
            ba = evaluate_bland_altman(y_curr, oof_preds)

            benchmark_results.append({
                'Target': target_display_names[target],
                'Target_Code': target,
                'Model': model_name,
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3),
                'Pearson_r': round(r, 3),
                'R2_Score': round(r2, 3),
                'BA_Mean_Bias': round(ba['mean_bias'], 3),
                'BA_Lower_LoA': round(ba['lower_loa'], 3),
                'BA_Upper_LoA': round(ba['upper_loa'], 3),
                'BA_Within_LoA_Pct': round(ba['pct_within_loa'], 1),
                'Unit': target_units[target],
                'Train_Time_Sec': round(elapsed, 2)
            })

            print(f"  • {model_name:22} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | Pearson r: {r:5.3f} | R²: {r2:5.3f} | LoA %: {ba['pct_within_loa']:5.1f}%")

    # Save benchmark table
    results_df = pd.DataFrame(benchmark_results)
    out_csv = os.path.join(RES_DIR, 'phase1_regression_benchmark.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to: {out_csv}")

    # Generate Publication Figures
    generate_figures(full_cohort, all_oof_predictions, target_cols, target_display_names, target_units)
    
    return results_df, all_oof_predictions

def generate_figures(cohort, all_oof_predictions, target_cols, target_names, target_units):
    print("\n--- Generating Publication-Quality Figures ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Target Comparison: Predictions vs True for Best Model (LightGBM)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, target in enumerate(target_cols):
        ax = axes[idx]
        y_true = cohort[cohort[target].notna()][target].values
        y_pred = all_oof_predictions[target]['LightGBM']
        
        # Truncate visual outliers for clean visualization (e.g. HOMA-IR > 20)
        p99 = np.percentile(y_true, 99)
        mask = (y_true <= p99) & (y_pred <= p99 * 1.2)
        
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.3, color='#1F4E79', edgecolors='none', s=20)
        # Identity line
        min_v = min(np.min(y_true[mask]), np.min(y_pred[mask]))
        max_v = max(np.max(y_true[mask]), np.max(y_pred[mask]))
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Prediction')
        
        # Regression trend
        m, b = np.polyfit(y_true[mask], y_pred[mask], 1)
        ax.plot(y_true[mask], m * y_true[mask] + b, color='#E74C3C', lw=1.5, label=f'Fit: y = {m:.2f}x + {b:.2f}')
        
        r, _ = stats.pearsonr(y_true, y_pred)
        ax.set_title(f"{target_names[target]}\nLightGBM (Pearson r = {r:.3f})", fontsize=12, fontweight='bold', color='#1F4E79')
        ax.set_xlabel(f"True Clinical Ground Truth ({target_units[target]})", fontsize=10)
        ax.set_ylabel(f"Model Prediction ({target_units[target]})", fontsize=10)
        ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    scatter_path = os.path.join(FIG_DIR, 'phase1_true_vs_predicted_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")

    # 2. Bland-Altman Agreement Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, target in enumerate(target_cols):
        ax = axes[idx]
        y_true = cohort[cohort[target].notna()][target].values
        y_pred = all_oof_predictions[target]['LightGBM']
        
        mean_val = (y_true + y_pred) / 2.0
        diff = y_pred - y_true
        
        bias = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)
        lower_loa = bias - 1.96 * sd_diff
        upper_loa = bias + 1.96 * sd_diff
        
        # Clip extreme outliers for plot clarity
        p99_mean = np.percentile(mean_val, 99)
        mask = (mean_val <= p99_mean) & (np.abs(diff) <= 3.5 * sd_diff)
        
        ax.scatter(mean_val[mask], diff[mask], alpha=0.3, color='#2E75B6', edgecolors='none', s=20)
        ax.axhline(bias, color='#C00000', lw=2, label=f'Mean Bias: {bias:.2f}')
        ax.axhline(upper_loa, color='#ED7D31', lw=1.5, ls='--', label=f'+1.96 SD: {upper_loa:.2f}')
        ax.axhline(lower_loa, color='#ED7D31', lw=1.5, ls='--', label=f'-1.96 SD: {lower_loa:.2f}')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        
        within_pct = np.mean((diff >= lower_loa) & (diff <= upper_loa)) * 100.0
        ax.set_title(f"Bland-Altman: {target_names[target]}\nLimits of Agreement: {within_pct:.1f}% within LoA", fontsize=12, fontweight='bold', color='#1F4E79')
        ax.set_xlabel(f"Mean of True & Predicted ({target_units[target]})", fontsize=10)
        ax.set_ylabel(f"Difference (Predicted - True) ({target_units[target]})", fontsize=10)
        ax.legend(loc='lower left', frameon=True, fontsize=9)

    plt.tight_layout()
    ba_path = os.path.join(FIG_DIR, 'phase1_bland_altman_agreement.png')
    plt.savefig(ba_path, dpi=300)
    plt.close()
    print(f"Saved: {ba_path}")

    # 3. Model Architecture Comparison Bar Chart
    res_df = pd.read_csv(os.path.join(RES_DIR, 'phase1_regression_benchmark.csv'))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(data=res_df, x='Target_Code', y='Pearson_r', hue='Model', palette='Blues_r', ax=ax)
    ax.set_title("Cross-Validated Predictive Correlation (Pearson r) Across Model Architectures", fontsize=13, fontweight='bold', color='#1F4E79')
    ax.set_xlabel("Metabolic Target Biomarker", fontsize=11, fontweight='bold')
    ax.set_ylabel("Pearson Correlation Coefficient (r)", fontsize=11, fontweight='bold')
    ax.set_xticklabels(['HOMA-IR (Insulin Resistance)', 'HbA1c Glycohemoglobin', 'Fasting Plasma Glucose'], fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, 'phase1_model_comparison_pearson_r.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")

if __name__ == '__main__':
    run_benchmark()
