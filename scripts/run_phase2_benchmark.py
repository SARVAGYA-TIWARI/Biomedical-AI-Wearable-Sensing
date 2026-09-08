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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
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
    cohort = df[df['is_target_adult'] & df['circadian_mesor'].notna()].copy()
    cohort = cohort[cohort['metabolic_risk_class'].notna()].copy()
    print(f"Loaded cohort for Phase 2: {len(cohort)} adults with valid labels and wearable records.")
    return cohort

def prepare_features_and_labels(cohort):
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

    X = cohort[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    class_mapping = {
        'Low_Risk_Normal': 0,
        'Moderate_Risk_Prediabetes': 1,
        'High_Risk_IR': 2
    }
    y_3class = cohort['metabolic_risk_class'].map(class_mapping).values
    y_binary = cohort['is_insulin_resistant'].fillna(0).astype(int).values

    return X, y_3class, y_binary, feature_cols, class_mapping

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=8, class_weight='balanced', random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.01, max_iter=250, random_state=42, early_stopping=True)
    }

def run_phase2():
    print("=" * 75)
    print("PHASE 2 BENCHMARK: NON-INVASIVE 3-CLASS METABOLIC RISK SCREENING")
    print("Zero Blood Access | Multi-Modal Wearables + Vitals + Demographics")
    print("=" * 75)

    cohort = load_data()
    X, y, y_binary, feature_cols, class_mapping = prepare_features_and_labels(cohort)
    class_names = ['Low Risk (Normal)', 'Moderate Risk (Prediabetes)', 'High Risk (IR)']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()

    benchmark_results = []
    all_oof_preds = {}
    all_oof_probs = {}

    print(f"\nClass Distribution across N={len(y)}:")
    for c_name, c_idx in class_mapping.items():
        cnt = np.sum(y == c_idx)
        print(f"  Class {c_idx} ({c_name}): {cnt} ({cnt/len(y)*100:.1f}%)")

    for model_name, model in models.items():
        t0 = time.time()
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros((len(y), 3))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if model_name in ['Logistic Regression', 'Neural Network (MLP)']:
                model.fit(X_train_scaled, y_train)
                oof_preds[val_idx] = model.predict(X_val_scaled)
                oof_probs[val_idx] = model.predict_proba(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                oof_preds[val_idx] = model.predict(X_val)
                oof_probs[val_idx] = model.predict_proba(X_val)

        elapsed = time.time() - t0
        all_oof_preds[model_name] = oof_preds
        all_oof_probs[model_name] = oof_probs

        acc = accuracy_score(y, oof_preds)
        b_acc = balanced_accuracy_score(y, oof_preds)
        macro_f1 = f1_score(y, oof_preds, average='macro')
        weighted_f1 = f1_score(y, oof_preds, average='weighted')
        
        # One-vs-Rest Multiclass AUROC
        y_bin = label_binarize(y, classes=[0, 1, 2])
        try:
            auc_ovr = roc_auc_score(y_bin, oof_probs, multi_class='ovr', average='macro')
        except Exception:
            auc_ovr = np.nan

        benchmark_results.append({
            'Model': model_name,
            'Accuracy': round(acc * 100.0, 2),
            'Balanced_Accuracy': round(b_acc * 100.0, 2),
            'Macro_F1': round(macro_f1 * 100.0, 2),
            'Weighted_F1': round(weighted_f1 * 100.0, 2),
            'Macro_AUROC': round(auc_ovr, 3),
            'Time_Sec': round(elapsed, 2)
        })

        print(f"  • {model_name:22} | Bal Acc: {b_acc*100:5.2f}% | Macro F1: {macro_f1*100:5.2f}% | Weighted F1: {weighted_f1*100:5.2f}% | AUROC: {auc_ovr:5.3f}")

    results_df = pd.DataFrame(benchmark_results)
    out_csv = os.path.join(RES_DIR, 'phase2_classification_benchmark.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved Phase 2 benchmark results to: {out_csv}")

    # Generate Publication Figures
    generate_figures(y, all_oof_preds, all_oof_probs, class_names, feature_cols, X)

    return results_df, all_oof_preds, all_oof_probs

def generate_figures(y, all_oof_preds, all_oof_probs, class_names, feature_cols, X):
    print("\n--- Generating Publication-Quality Phase 2 Figures ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Confusion Matrix Comparison (LightGBM vs. Logistic Regression)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for idx, m_name in enumerate(['LightGBM', 'Logistic Regression']):
        ax = axes[idx]
        cm = confusion_matrix(y, all_oof_preds[m_name])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
        
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', cbar=True, ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 11, 'weight': 'bold'})
        
        ax.set_title(f"Confusion Matrix: {m_name}\nNormalized Percentages (%)", fontsize=12, fontweight='bold', color='#1F4E79')
        ax.set_ylabel("True Clinical Label", fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicted Screening Label", fontsize=10, fontweight='bold')
        ax.set_xticklabels(class_names, rotation=15, ha='right', fontsize=9)

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, 'phase2_confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. Multiclass One-vs-Rest ROC Curves for Best Model (LightGBM)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_bin = label_binarize(y, classes=[0, 1, 2])
    probs = all_oof_probs['LightGBM']
    
    colors = ['#2E75B6', '#ED7D31', '#C00000']
    for c_idx in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, c_idx], probs[:, c_idx])
        c_auc = roc_auc_score(y_bin[:, c_idx], probs[:, c_idx])
        ax.plot(fpr, tpr, color=colors[c_idx], lw=2.2, label=f'{class_names[c_idx]} (AUC = {c_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Baseline (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Phase 2 Multiclass ROC Curves (LightGBM Screening Engine)', fontsize=12, fontweight='bold', color='#1F4E79')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    roc_path = os.path.join(FIG_DIR, 'phase2_multiclass_roc_curves.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")

    # 3. Model Performance Bar Chart (Balanced Accuracy & Macro F1)
    res_df = pd.read_csv(os.path.join(RES_DIR, 'phase2_classification_benchmark.csv'))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    res_plot = pd.melt(res_df, id_vars=['Model'], value_vars=['Balanced_Accuracy', 'Macro_F1'],
                       var_name='Metric', value_name='Score')
    res_plot['Metric'] = res_plot['Metric'].map({'Balanced_Accuracy': 'Balanced Accuracy (%)', 'Macro_F1': 'Macro F1-Score (%)'})
    
    sns.barplot(data=res_plot, x='Model', y='Score', hue='Metric', palette=['#1F4E79', '#ED7D31'], ax=ax)
    ax.axhline(33.33, color='red', ls='--', lw=1.5, label='Random Chance Baseline (33.33%)')
    ax.set_title('Phase 2 Model Performance: 3-Class Metabolic Risk Screening', fontsize=12, fontweight='bold', color='#1F4E79')
    ax.set_ylabel('Percentage Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classification Architecture', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 75])
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, 'phase2_model_comparison_bar_chart.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")

    # 4. Feature Importance Horizontal Bar Chart
    lgb_clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, verbose=-1)
    lgb_clf.fit(X, y)
    imp = pd.Series(lgb_clf.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    imp.tail(15).plot(kind='barh', color='#1F4E79', ax=ax)
    ax.set_title('Top 15 Most Influential Non-Invasive Digital Biomarkers for 3-Class Screening', fontsize=12, fontweight='bold', color='#1F4E79')
    ax.set_xlabel('Feature Importance (LightGBM Split Gain)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Digital Biomarker', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    imp_path = os.path.join(FIG_DIR, 'phase2_feature_importance_top15.png')
    plt.savefig(imp_path, dpi=300)
    plt.close()
    print(f"Saved: {imp_path}")

if __name__ == '__main__':
    run_phase2()
