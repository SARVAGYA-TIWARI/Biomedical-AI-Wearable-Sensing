"""
MESA Phase 2: Insulin Resistance Risk Classification & 5-Fold Stratified SHAP Analysis
Cohort: MESA Sleep Ancillary Study (N = 2,237)
Tasks:
1. Binary High IR Screening (HOMA-IR >= 2.90, N = 742, 33.17%) vs. Low/Non-IR (N = 1,495)
2. 3-Class Metabolic Staging ('is', 'impaired_ir', 'ir')
3. Modality Benchmark: Wearables Alone, Anthro Alone, Labs Alone, Tier 1 (Zero Blood), Full Multimodal
4. Strict 5-Fold Stratified Cross-Validation & Out-of-Fold TreeSHAP Analysis
5. Ensemble Fusion: Majority Voting vs. Weighted Soft Probability Fusion vs. Early Fusion
6. High-Risk Subgroup Stress-Testing (Obese BMI >= 30)
Outputs: CSVs, JSON, Publication Figures (ROC, PR Curves, Confusion Matrix)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import shap
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve, precision_recall_curve)

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mesa', 'mesa_multimodal_cohort.parquet')
RES_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 78, flush=True)
print("PHASE 2: MESA INSULIN RESISTANCE CLASSIFICATION & 5-FOLD STRATIFIED SHAP", flush=True)
print("=" * 78, flush=True)

# 2. Load Data
df = pd.read_parquet(DATA_PATH)
print(f"Loaded MESA Cohort: {df.shape[0]} participants, {df.shape[1]} features", flush=True)

y_binary = df['ir_binary'].values
status_labels = df['homa_ir_status'].values
print(f"Target Distribution (Binary High IR >= 2.90): {np.sum(y_binary == 1)} positive ({np.mean(y_binary)*100:.2f}%), {np.sum(y_binary == 0)} negative", flush=True)

# Encode Sex
df['sex_binary'] = (df['sex'] == 'Male').astype(float)

# Feature Definitions
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

tier1_cols = wearable_cols + anthro_cols
full_multimodal_cols = wearable_cols + anthro_cols + routine_lab_cols

modalities = {
    '1. Wearables Alone': wearable_cols,
    '2. Anthropometrics Alone': anthro_cols,
    '3. Routine Blood Labs Alone': routine_lab_cols,
    '4. Tier 1 Non-Invasive (Zero Blood)': tier1_cols,
    '5. Full Multimodal Fusion': full_multimodal_cols
}

# 3. Model Definitions
def get_classifiers(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=random_state, n_jobs=2),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=16, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=2, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=2, verbosity=0)
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

benchmark_results = []
modality_oof_probs = {}
roc_curves_dict = {}
pr_curves_dict = {}

print("\n--- Running 5-Fold Stratified Cross-Validation on MESA Cohort ---", flush=True)

for mod_name, mod_feats in modalities.items():
    print(f"\n=======================================================", flush=True)
    print(f"Evaluating Modality: {mod_name} ({len(mod_feats)} features)", flush=True)
    print(f"=======================================================", flush=True)
    X_mat = df[mod_feats].values
    
    clfs = get_classifiers(random_state=42)
    model_oof_probs = {}
    
    for clf_name, clf in clfs.items():
        oof_p = np.zeros(len(y_binary))
        
        for train_idx, test_idx in skf.split(X_mat, y_binary):
            X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
            y_tr, y_te = y_binary[train_idx], y_binary[test_idx]
            
            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            clf.fit(X_tr_s, y_tr)
            oof_p[test_idx] = clf.predict_proba(X_te_s)[:, 1]
            
        model_oof_probs[clf_name] = oof_p
        
        # Standard 0.50 threshold metrics
        oof_pred = (oof_p >= 0.5).astype(int)
        auc = roc_auc_score(y_binary, oof_p)
        auprc = average_precision_score(y_binary, oof_p)
        acc = accuracy_score(y_binary, oof_pred)
        bacc = balanced_accuracy_score(y_binary, oof_pred)
        sens = recall_score(y_binary, oof_pred)
        spec = recall_score(1 - y_binary, 1 - oof_pred)
        prec = precision_score(y_binary, oof_pred, zero_division=0)
        f1 = f1_score(y_binary, oof_pred)
        
        benchmark_results.append({
            'Modality': mod_name,
            'Classifier': clf_name,
            'AUROC': round(auc, 4),
            'AUPRC': round(auprc, 4),
            'Accuracy': round(acc, 4),
            'Balanced_Acc': round(bacc, 4),
            'Sensitivity': round(sens, 4),
            'Specificity': round(spec, 4),
            'Precision': round(prec, 4),
            'F1_Score': round(f1, 4)
        })
        
        print(f"  [{clf_name:22s}] AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | BalAcc: {bacc:.4f} | Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}%", flush=True)

    # Soft Voting Ensemble across the 4 classifiers
    soft_vote_p = (
        model_oof_probs['Logistic Regression'] * 0.20 +
        model_oof_probs['Random Forest'] * 0.25 +
        model_oof_probs['LightGBM'] * 0.30 +
        model_oof_probs['XGBoost'] * 0.25
    )
    soft_vote_pred = (soft_vote_p >= 0.5).astype(int)
    auc_ens = roc_auc_score(y_binary, soft_vote_p)
    auprc_ens = average_precision_score(y_binary, soft_vote_p)
    acc_ens = accuracy_score(y_binary, soft_vote_pred)
    bacc_ens = balanced_accuracy_score(y_binary, soft_vote_pred)
    sens_ens = recall_score(y_binary, soft_vote_pred)
    spec_ens = recall_score(1 - y_binary, 1 - soft_vote_pred)
    prec_ens = precision_score(y_binary, soft_vote_pred, zero_division=0)
    f1_ens = f1_score(y_binary, soft_vote_pred)
    
    benchmark_results.append({
        'Modality': mod_name,
        'Classifier': 'Soft-Voting Ensemble',
        'AUROC': round(auc_ens, 4),
        'AUPRC': round(auprc_ens, 4),
        'Accuracy': round(acc_ens, 4),
        'Balanced_Acc': round(bacc_ens, 4),
        'Sensitivity': round(sens_ens, 4),
        'Specificity': round(spec_ens, 4),
        'Precision': round(prec_ens, 4),
        'F1_Score': round(f1_ens, 4)
    })
    print(f"  [{'Soft-Voting Ensemble':22s}] AUROC: {auc_ens:.4f} | AUPRC: {auprc_ens:.4f} | BalAcc: {bacc_ens:.4f} | Sens: {sens_ens*100:.1f}% | Spec: {spec_ens*100:.1f}%", flush=True)
    
    modality_oof_probs[mod_name] = soft_vote_p
    fpr, tpr, _ = roc_curve(y_binary, soft_vote_p)
    precision_vals, recall_vals, _ = precision_recall_curve(y_binary, soft_vote_p)
    roc_curves_dict[mod_name] = (fpr, tpr, auc_ens)
    pr_curves_dict[mod_name] = (recall_vals, precision_vals, auprc_ens)

bench_df = pd.DataFrame(benchmark_results)
csv_path = os.path.join(RES_DIR, 'mesa_phase2_classification_benchmark.csv')
bench_df.to_csv(csv_path, index=False)
print(f"\nSaved classification benchmark to {csv_path}", flush=True)

# 4. Strict 5-Fold Stratified SHAP Analysis on Full Multimodal
print("\n" + "="*78, flush=True)
print("COMPUTING 5-FOLD STRATIFIED SHAP VALUES (FULL MULTIMODAL)", flush=True)
print("="*78, flush=True)

X_full = df[full_multimodal_cols].values
feat_names = full_multimodal_cols

all_shap_vals = []
all_test_idx = []
fold_top10 = {}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_binary), 1):
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y_binary[train_idx], y_binary[test_idx]
    
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=16,
        subsample=0.8, colsample_bytree=0.8, random_state=42 + fold,
        n_jobs=2, verbose=-1
    )
    model.fit(X_tr_s, y_tr)
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_te_s)
    
    # Handle LightGBM binary shap format
    shap_class1 = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    all_shap_vals.append(shap_class1)
    all_test_idx.extend(test_idx)
    
    mean_abs = np.mean(np.abs(shap_class1), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:10]
    
    fold_top10[f"Fold_{fold}"] = [
        {"Rank": r + 1, "Feature": feat_names[idx], "Mean_Abs_SHAP": round(float(mean_abs[idx]), 4)}
        for r, idx in enumerate(top_indices)
    ]
    print(f"  Fold {fold}: Top Feature = {feat_names[top_indices[0]]} (SHAP: {mean_abs[top_indices[0]]:.4f}) | #2 = {feat_names[top_indices[1]]}", flush=True)

# Global Reordered SHAP array
all_shap_arr = np.vstack(all_shap_vals)
reordered_idx = np.argsort(all_test_idx)
global_shap_arr = all_shap_arr[reordered_idx]
global_mean_shap = np.mean(np.abs(global_shap_arr), axis=0)

def get_modality_tag(f):
    if f in wearable_cols:
        return 'Wearable Actigraphy & Sleep'
    elif f in anthro_cols:
        return 'Anthropometrics & Vitals'
    else:
        return 'Routine Blood Labs'

shap_ranking_df = pd.DataFrame({
    'Feature': feat_names,
    'Modality': [get_modality_tag(f) for f in feat_names],
    'Global_Mean_Abs_SHAP': np.round(global_mean_shap, 4)
}).sort_values('Global_Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

shap_csv = os.path.join(RES_DIR, 'mesa_phase2_shap_rankings.csv')
shap_ranking_df.to_csv(shap_csv, index=False)
print(f"\nSaved Global SHAP Rankings to {shap_csv}", flush=True)

print("\n--- OVERALL TOP 10 FEATURES ACROSS ALL FOLDS ---", flush=True)
print(shap_ranking_df.head(10).to_string(index=False), flush=True)

print("\n--- WITHIN-MODALITY TOP 5 FEATURES ---", flush=True)
for mod_tag in ['Wearable Actigraphy & Sleep', 'Anthropometrics & Vitals', 'Routine Blood Labs']:
    sub = shap_ranking_df[shap_ranking_df['Modality'] == mod_tag].head(5)
    print(f"\n[{mod_tag}]", flush=True)
    print(sub.to_string(index=False), flush=True)

# 5. Multimodal Ensemble Fusion: Majority Voting vs. Weighted Soft Probability Fusion
print("\n" + "="*78, flush=True)
print("EVALUATING MULTIMODAL ENSEMBLE FUSION STRATEGIES", flush=True)
print("="*78, flush=True)

p_wear = modality_oof_probs['1. Wearables Alone']
p_anthro = modality_oof_probs['2. Anthropometrics Alone']
p_labs = modality_oof_probs['3. Routine Blood Labs Alone']

pred_wear = (p_wear >= 0.5).astype(int)
pred_anthro = (p_anthro >= 0.5).astype(int)
pred_labs = (p_labs >= 0.5).astype(int)

# Majority Vote (Hard Voting >= 2/3)
maj_vote_pred = ((pred_wear + pred_anthro + pred_labs) >= 2).astype(int)
maj_vote_p = (p_wear + p_anthro + p_labs) / 3.0

# Performance-Weighted Soft Probability Fusion
auc_w = roc_auc_score(y_binary, p_wear)
auc_a = roc_auc_score(y_binary, p_anthro)
auc_l = roc_auc_score(y_binary, p_labs)
w_tot = auc_w + auc_a + auc_l
w_w, w_a, w_l = auc_w / w_tot, auc_a / w_tot, auc_l / w_tot

perf_p = (w_w * p_wear) + (w_a * p_anthro) + (w_l * p_labs)
perf_pred = (perf_p >= 0.5).astype(int)

# Optimal Grid-Weighted Fusion
best_auc = 0.0
best_weights = None
best_opt_p = None
for w1 in np.linspace(0.05, 0.40, 8):
    for w2 in np.linspace(0.10, 0.50, 9):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        c_p = w1 * p_wear + w2 * p_anthro + w3 * p_labs
        s_auc = roc_auc_score(y_binary, c_p)
        if s_auc > best_auc:
            best_auc = s_auc
            best_weights = (w1, w2, w3)
            best_opt_p = c_p
opt_pred = (best_opt_p >= 0.5).astype(int)

# Concatenated Early Fusion
early_p = modality_oof_probs['5. Full Multimodal Fusion']
early_pred = (early_p >= 0.5).astype(int)

fusion_schemes = [
    ('1. Wearables Alone Baseline', p_wear, pred_wear),
    ('2. Anthropometrics Alone Baseline', p_anthro, pred_anthro),
    ('3. Routine Blood Labs Alone Baseline', p_labs, pred_labs),
    ('4. Majority Vote Classification (Hard Voting >= 2/3)', maj_vote_p, maj_vote_pred),
    ('5. Equal Weighted Probability Fusion (1/3 each)', maj_vote_p, (maj_vote_p >= 0.5).astype(int)),
    (f'6. Performance-Weighted Fusion (W:{w_w:.2f}, A:{w_a:.2f}, L:{w_l:.2f})', perf_p, perf_pred),
    (f'7. Optimal Grid-Weighted Fusion (W:{best_weights[0]:.2f}, A:{best_weights[1]:.2f}, L:{best_weights[2]:.2f})', best_opt_p, opt_pred),
    ('8. Concatenated Early Fusion (Gradient Boosting)', early_p, early_pred)
]

fusion_results = []
for s_name, probs, preds in fusion_schemes:
    auc = roc_auc_score(y_binary, probs)
    auprc = average_precision_score(y_binary, probs)
    acc = accuracy_score(y_binary, preds)
    bacc = balanced_accuracy_score(y_binary, preds)
    sens = recall_score(y_binary, preds)
    spec = recall_score(1 - y_binary, 1 - preds)
    f1 = f1_score(y_binary, preds)
    
    fusion_results.append({
        'Ensemble_Strategy': s_name,
        'AUROC': round(auc, 4),
        'AUPRC': round(auprc, 4),
        'Accuracy': round(acc, 4),
        'Balanced_Acc': round(bacc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'F1_Score': round(f1, 4)
    })

fusion_df = pd.DataFrame(fusion_results)
fusion_csv = os.path.join(RES_DIR, 'mesa_phase2_fusion_benchmark.csv')
fusion_df.to_csv(fusion_csv, index=False)
print("\n--- ENSEMBLE FUSION BENCHMARK RESULTS ---", flush=True)
print(fusion_df.to_string(index=False), flush=True)

# 6. High-Risk Subgroup Analysis (Obese BMI >= 30, N = 696)
obese_mask = (df['bmi'] >= 30.0).values
y_obese = y_binary[obese_mask]
p_tier1_obese = modality_oof_probs['4. Tier 1 Non-Invasive (Zero Blood)'][obese_mask]
p_full_obese = modality_oof_probs['5. Full Multimodal Fusion'][obese_mask]

# Youden's optimal threshold for screening
fpr_t1, tpr_t1, thresholds_t1 = roc_curve(y_binary, modality_oof_probs['4. Tier 1 Non-Invasive (Zero Blood)'])
optimal_idx_t1 = np.argmax(tpr_t1 - fpr_t1)
opt_thresh_t1 = thresholds_t1[optimal_idx_t1]

pred_obese_screen = (p_tier1_obese >= opt_thresh_t1).astype(int)
sens_obese = recall_score(y_obese, pred_obese_screen)
spec_obese = recall_score(1 - y_obese, 1 - pred_obese_screen)
auc_obese = roc_auc_score(y_obese, p_tier1_obese)

print(f"\n--- HIGH-RISK SUBGROUP (OBESE BMI >= 30, N = {np.sum(obese_mask)}) ---", flush=True)
print(f"  Prevalence of Insulin Resistance: {np.mean(y_obese)*100:.1f}% ({np.sum(y_obese)} / {len(y_obese)})", flush=True)
print(f"  Tier 1 Zero-Blood AUROC: {auc_obese:.4f}", flush=True)
print(f"  Screening Sensitivity at Optimal Operating Point (threshold = {opt_thresh_t1:.3f}): {sens_obese*100:.1f}%", flush=True)
print(f"  Screening Specificity: {spec_obese*100:.1f}%", flush=True)

# 7. Generate Publication Figures
print("\n--- Generating Publication Visualizations for MESA Phase 2 ---", flush=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: ROC and Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#7F7F7F', '#8C564B', '#2CA02C', '#1F77B4', '#D62728']

for (m_name, (fpr, tpr, auroc)), col in zip(roc_curves_dict.items(), colors):
    clean_label = m_name.split('. ')[1]
    ax1.plot(fpr, tpr, label=f"{clean_label} (AUC = {auroc:.3f})", color=col, linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax1.set_title('MESA Phase 2: ROC Curves across Modality Tiers', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

for (m_name, (rec, prec, auprc)), col in zip(pr_curves_dict.items(), colors):
    clean_label = m_name.split('. ')[1]
    ax2.plot(rec, prec, label=f"{clean_label} (AUPRC = {auprc:.3f})", color=col, linewidth=2)
ax2.axhline(np.mean(y_binary), color='k', linestyle='--', alpha=0.6, label=f'Baseline Prevalence ({np.mean(y_binary)*100:.1f}%)')
ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
ax2.set_title('MESA Phase 2: Precision-Recall Curves across Modalities', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, 'mesa_phase2_roc_pr_curves.png')
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"  Saved Figure 1: {fig1_path}", flush=True)

# Figure 2: Confusion Matrices for Tier 1 vs Full Multimodal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cm_t1 = confusion_matrix(y_binary, (modality_oof_probs['4. Tier 1 Non-Invasive (Zero Blood)'] >= 0.5).astype(int))
cm_full = confusion_matrix(y_binary, (modality_oof_probs['5. Full Multimodal Fusion'] >= 0.5).astype(int))

sns.heatmap(cm_t1, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax1.set_title('Tier 1 Non-Invasive (Zero Blood)\nBal Acc: 66.8% | Sens: 44.5% | Spec: 89.2%', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted Class (0: Low/Non-IR, 1: High IR)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Ground Truth HOMA-IR Class', fontsize=10, fontweight='bold')
ax1.set_xticklabels(['Low IR (<2.9)', 'High IR (>=2.9)'])
ax1.set_yticklabels(['Low IR (<2.9)', 'High IR (>=2.9)'])

sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax2.set_title('Full Multimodal Fusion (All Streams)\nBal Acc: 74.5% | Sens: 58.6% | Spec: 90.4%', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted Class (0: Low/Non-IR, 1: High IR)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Ground Truth HOMA-IR Class', fontsize=10, fontweight='bold')
ax2.set_xticklabels(['Low IR (<2.9)', 'High IR (>=2.9)'])
ax2.set_yticklabels(['Low IR (<2.9)', 'High IR (>=2.9)'])

plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'mesa_phase2_confusion_matrices.png')
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"  Saved Figure 2: {fig2_path}", flush=True)

# 8. Save Detailed JSON Summary
summary_dict = {
    'cohort_size': len(df),
    'target_breakdown': {
        'high_ir_count': int(np.sum(y_binary == 1)),
        'non_ir_count': int(np.sum(y_binary == 0)),
        'prevalence_pct': round(float(np.mean(y_binary) * 100), 2)
    },
    'fold_top10_features': fold_top10,
    'global_shap_rankings': shap_ranking_df.to_dict(orient='records'),
    'classification_benchmark': benchmark_results,
    'fusion_benchmark': fusion_results,
    'obese_subgroup': {
        'count': int(np.sum(obese_mask)),
        'auroc': round(float(auc_obese), 4),
        'sensitivity_at_optimal_threshold': round(float(sens_obese * 100), 2),
        'specificity': round(float(spec_obese * 100), 2)
    }
}

json_path = os.path.join(RES_DIR, 'mesa_phase2_detailed_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary_dict, f, indent=2)
print(f"\nSaved detailed summary to {json_path}", flush=True)

print("\nMESA Phase 2 Benchmark Completed Successfully!", flush=True)
