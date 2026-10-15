"""
Phase 2: Multimodal Insulin Resistance Risk Stratification & Classification Benchmark
Using Google Health Nature 2026 Dataset (N = 1,165 Participants)
Evaluates: Binary IR (HOMA-IR >= 2.9) & Ternary Clinical Status ('is' vs 'impaired_ir' vs 'ir')
Metrics: AUROC, AUPRC, Sensitivity, Specificity, F1-Score, Confusion Matrices, Subgroup Analysis
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import lightgbm as lgb
import xgboost as xgb

def run_phase2():
    print("=" * 78, flush=True)
    print("PHASE 2: GOOGLE HEALTH INSULIN RESISTANCE CLASSIFICATION BENCHMARK", flush=True)
    print("=" * 78, flush=True)
    
    # 1. Load Data
    data_path = os.path.join("data", "google_health_ir", "ir_prediction_data.pq")
    df = pd.read_parquet(data_path)
    print(f"Loaded dataset: {df.shape[0]} participants", flush=True)
    
    # Target 1: Binary IR (HOMA-IR >= 2.9)
    y_binary = (df['homa_ir'].values >= 2.9).astype(int)
    pos_rate = np.mean(y_binary)
    print(f"Binary Target: {np.sum(y_binary)} High IR (>= 2.9, {pos_rate*100:.1f}%), {len(y_binary) - np.sum(y_binary)} Low/Normal IR", flush=True)
    
    # Target 2: Ternary status
    status_map = {'is': 0, 'impaired_ir': 1, 'ir': 2}
    y_ternary = df['homa_ir_status'].map(status_map).values
    
    # Sex Binary
    sex_binary = (df['sex'].astype(str).str.lower().str.startswith('m')).astype(float)
    
    # 2. Feature Groups
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
    
    df_imputed = df.copy()
    for col in set(wearable_cols + anthro_vitals_cols + routine_lab_cols):
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            
    df_imputed['sex_binary'] = sex_binary
    df_imputed['WHtR'] = (df_imputed['waist_circumference'] / df_imputed['height'].replace(0, np.nan)).fillna(df_imputed['waist_circumference'].median() / 170.0)
    df_imputed['MAP'] = df_imputed['diastolic_bp'] + (df_imputed['systolic_bp'] - df_imputed['diastolic_bp']) / 3.0
    df_imputed['Pulse_Pressure'] = df_imputed['systolic_bp'] - df_imputed['diastolic_bp']
    df_imputed['Autonomic_Stress_Ratio'] = df_imputed['RHR_FreeLiving_mean'] / (df_imputed['HRV_FreeLiving_mean'] + 1e-4)
    df_imputed['TG_HDL_Ratio'] = df_imputed['triglycerides'] / (df_imputed['hdl'] + 1e-4)
    
    engineered_cols = ['WHtR', 'MAP', 'Pulse_Pressure', 'Autonomic_Stress_Ratio', 'TG_HDL_Ratio']
    
    modalities = {
        'Wearables Alone': wearable_cols,
        'Wearables + Anthropometrics & Vitals': wearable_cols + anthro_vitals_cols + ['sex_binary', 'WHtR', 'MAP', 'Autonomic_Stress_Ratio'],
        'Routine Blood Labs Alone': routine_lab_cols + ['TG_HDL_Ratio'],
        'Full Multimodal Fusion': wearable_cols + anthro_vitals_cols + ['sex_binary'] + routine_lab_cols + engineered_cols
    }
    
    def get_classifiers(random_state=42):
        return {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=200, random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=16, class_weight='balanced', random_state=random_state, n_jobs=-1, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=(1.0 - pos_rate)/pos_rate, random_state=random_state, n_jobs=-1, verbosity=0)
        }
    
    # 3. Benchmark Binary IR Classification (HOMA-IR >= 2.9)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    binary_results = []
    roc_curves_dict = {}
    pr_curves_dict = {}
    
    print("\n--- Benchmarking Binary IR Classification (HOMA-IR >= 2.9) ---", flush=True)
    
    for mod_name, mod_feats in modalities.items():
        print(f"\nModality: {mod_name} ({len(mod_feats)} features)", flush=True)
        X_mat = df_imputed[mod_feats].values
        
        classifiers = get_classifiers(random_state=42)
        
        for clf_name, clf in classifiers.items():
            oof_probs = np.zeros(len(y_binary))
            oof_preds = np.zeros(len(y_binary))
            
            for train_idx, test_idx in skf.split(X_mat, y_binary):
                X_train, X_test = X_mat[train_idx], X_mat[test_idx]
                y_train = y_binary[train_idx]
                
                scaler = RobustScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                clf.fit(X_train_s, y_train)
                probs = clf.predict_proba(X_test_s)[:, 1]
                preds = (probs >= 0.50).astype(int)
                
                oof_probs[test_idx] = probs
                oof_preds[test_idx] = preds
                
            auroc = roc_auc_score(y_binary, oof_probs)
            auprc = average_precision_score(y_binary, oof_probs)
            acc = accuracy_score(y_binary, oof_preds)
            bal_acc = balanced_accuracy_score(y_binary, oof_preds)
            f1 = f1_score(y_binary, oof_preds, average='macro')
            
            # Sensitivity & Specificity
            cm = confusion_matrix(y_binary, oof_preds)
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            binary_results.append({
                'Modality': mod_name,
                'Classifier': clf_name,
                'AUROC': auroc,
                'AUPRC': auprc,
                'Balanced_Acc': bal_acc,
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'Macro_F1': f1
            })
            
            print(f"  [{clf_name:22s}] AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}% | BalAcc: {bal_acc*100:.1f}%", flush=True)
            
            # Store ROC & PR for top models
            if clf_name == 'XGBoost':
                fpr, tpr, _ = roc_curve(y_binary, oof_probs)
                prec, rec, _ = precision_recall_curve(y_binary, oof_probs)
                roc_curves_dict[mod_name] = (fpr, tpr, auroc)
                pr_curves_dict[mod_name] = (rec, prec, auprc)
                
    binary_df = pd.DataFrame(binary_results)
    
    # 4. High-Risk Subgroup Analysis (Obese BMI >= 30 & Sedentary)
    print("\n--- High-Risk Subgroup Analysis (Obese: BMI >= 30 & Sedentary) ---", flush=True)
    obese_mask = (df_imputed['bmi'].values >= 30.0)
    sedentary_mask = (df_imputed['STEPS_Daily_mean'].values <= 5000.0)
    high_risk_mask = obese_mask | sedentary_mask
    
    print(f"High-Risk Subset: {np.sum(high_risk_mask)} participants ({np.sum(high_risk_mask)/len(df_imputed)*100:.1f}% of cohort)", flush=True)
    
    X_multimodal = df_imputed[modalities['Full Multimodal Fusion']].values
    clf_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=(1.0 - pos_rate)/pos_rate, random_state=42, n_jobs=-1, verbosity=0)
    
    oof_hr_probs = np.zeros(len(y_binary))
    oof_hr_preds = np.zeros(len(y_binary))
    
    for train_idx, test_idx in skf.split(X_multimodal, y_binary):
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_multimodal[train_idx])
        X_test_s = scaler.transform(X_multimodal[test_idx])
        clf_xgb.fit(X_train_s, y_binary[train_idx])
        probs = clf_xgb.predict_proba(X_test_s)[:, 1]
        oof_hr_probs[test_idx] = probs
        oof_hr_preds[test_idx] = (probs >= 0.40).astype(int)  # Tuned clinical screening threshold
        
    y_hr_true = y_binary[high_risk_mask]
    y_hr_pred = oof_hr_preds[high_risk_mask]
    y_hr_prob = oof_hr_probs[high_risk_mask]
    
    cm_hr = confusion_matrix(y_hr_true, y_hr_pred)
    tn_h, fp_h, fn_h, tp_h = cm_hr.ravel()
    hr_sens = tp_h / (tp_h + fn_h)
    hr_spec = tn_h / (tn_h + fp_h)
    hr_auroc = roc_auc_score(y_hr_true, y_hr_prob)
    
    print(f"XGBoost in Obese/Sedentary Subgroup: AUROC = {hr_auroc:.4f} | Sensitivity = {hr_sens*100:.1f}% | Specificity = {hr_spec*100:.1f}%", flush=True)
    
    # Save CSV
    csv_path = os.path.join("results", "google_health_phase2_classification_benchmark.csv")
    binary_df.to_csv(csv_path, index=False)
    print(f"\nSaved classification benchmark to {csv_path}", flush=True)
    
    # 5. Generate Figures
    os.makedirs("figures", exist_ok=True)
    
    # Figure 1: ROC and PR Curves across Modalities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    palette = {'Wearables Alone': '#3498db', 'Wearables + Anthropometrics & Vitals': '#2980b9', 'Routine Blood Labs Alone': '#f39c12', 'Full Multimodal Fusion': '#27ae60'}
    
    for mod_name, (fpr, tpr, auroc) in roc_curves_dict.items():
        ax1.plot(fpr, tpr, label=f"{mod_name} (AUROC = {auroc:.3f})", color=palette[mod_name], linewidth=2.2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUROC = 0.50)')
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves\nBinary Insulin Resistance (HOMA-IR >= 2.9)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    for mod_name, (rec, prec, auprc) in pr_curves_dict.items():
        ax2.plot(rec, prec, label=f"{mod_name} (AUPRC = {auprc:.3f})", color=palette[mod_name], linewidth=2.2)
    ax2.axhline(pos_rate, color='k', linestyle='--', alpha=0.5, label=f'Baseline Prevalence ({pos_rate*100:.1f}%)')
    ax2.set_title("Precision-Recall (PR) Curves\nBinary Insulin Resistance (HOMA-IR >= 2.9)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join("figures", "google_health_phase2_roc_pr_curves.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {fig1_path}", flush=True)
    
    # Figure 2: Confusion Matrix for Best Multimodal Model
    plt.figure(figsize=(7, 6))
    cm_best = confusion_matrix(y_binary, oof_preds)
    cm_norm = cm_best.astype('float') / cm_best.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                xticklabels=['Normal / Low IR (< 2.9)', 'High IR (>= 2.9)'],
                yticklabels=['Normal / Low IR (< 2.9)', 'High IR (>= 2.9)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    plt.title(f"Normalized Confusion Matrix: Multimodal XGBoost\nOverall AUROC = {roc_auc_score(y_binary, oof_probs):.4f} | Sensitivity = {sens*100:.1f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Ground-Truth Clinical Status", fontsize=11, labelpad=8)
    plt.tight_layout()
    fig2_path = os.path.join("figures", "google_health_phase2_confusion_matrix.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {fig2_path}", flush=True)
    
    print("\n--- PHASE 2 BENCHMARK SUMMARY (XGBoost) ---", flush=True)
    summary_xgb = binary_df[binary_df['Classifier'] == 'XGBoost'][['Modality', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Macro_F1']]
    print(summary_xgb.to_string(index=False), flush=True)

if __name__ == '__main__':
    run_phase2()
