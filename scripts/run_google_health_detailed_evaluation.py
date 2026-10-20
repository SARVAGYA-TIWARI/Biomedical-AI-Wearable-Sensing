"""
Comprehensive Analysis Suite for Google Health Nature 2026 Cohort:
1. Exact definitions and statistics of 'is', 'impaired_ir', and 'ir' classes.
2. Strict 5-Fold Subject-Level Stratified Cross-Validation.
3. Top 10 Features across each of the 5 folds via SHAP.
4. Overall Top 5 and Within-Modality Top 5 Features.
5. Modality Importance Evaluation (Wearables, Anthro, Labs, Tier 1, Full Multimodal).
6. Multimodal Ensemble: Weighted Probability Fusion vs Majority Voting vs Early Fusion.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
import lightgbm as lgb
import xgboost as xgb
import shap

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'google_health_ir', 'ir_prediction_data.pq')
res_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(res_dir, exist_ok=True)

df = pd.read_parquet(data_path)
print(f"Loaded Google Health Cohort: {df.shape[0]} participants, {df.shape[1]} features")

# 2. Analyze Target Classes: 'is', 'impaired_ir', 'ir'
status_stats = df.groupby('homa_ir_status')['homa_ir'].agg(['count', 'min', 'max', 'mean', 'median', 'std'])
status_stats['pct'] = round(status_stats['count'] / len(df) * 100, 2)
print("\n--- 3-Class Ground Truth Statistics ---")
print(status_stats)

# Binary target: High IR (homa_ir_status == 'ir', HOMA-IR >= 2.90) vs Low/Non-IR (others)
y_binary = (df['homa_ir_status'] == 'ir').astype(int).values
y_raw_homa = df['homa_ir'].values

# Encode Sex (Male: 1, Female: 0)
sex_binary = (df['sex'].astype(str).str.lower().str.startswith('m')).astype(float)

# 3. Define Modality Feature Sets
wearable_cols = [
    'RHR_FreeLiving_mean', 'RHR_FreeLiving_median', 'RHR_FreeLiving_std',
    'HRV_FreeLiving_mean', 'HRV_FreeLiving_median', 'HRV_FreeLiving_std',
    'STEPS_Daily_mean', 'STEPS_Daily_median', 'STEPS_Daily_std',
    'SLEEP_Duration_mean', 'SLEEP_Duration_median', 'SLEEP_Duration_std'
]

anthro_cols = [
    'age', 'bmi', 'waist_circumference', 'height', 'weight',
    'systolic_bp', 'diastolic_bp'
]

routine_lab_cols = [
    'total cholesterol', 'hdl', 'triglycerides', 'ldl', 'chol/hdl', 'non hdl',
    'glucose', 'creatinine', 'egfr', 'bun', 'alt', 'ast', 'hba1c', 'crp'
]

# Impute medians & engineer ratios
df_feat = df.copy()
for col in set(wearable_cols + anthro_cols + routine_lab_cols):
    if col in df_feat.columns:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())

df_feat['sex_binary'] = sex_binary
df_feat['WHtR'] = df_feat['waist_circumference'] / df_feat['height'].replace(0, np.nan)
df_feat['WHtR'] = df_feat['WHtR'].fillna(df_feat['WHtR'].median())
df_feat['MAP'] = df_feat['diastolic_bp'] + (df_feat['systolic_bp'] - df_feat['diastolic_bp']) / 3.0
df_feat['Pulse_Pressure'] = df_feat['systolic_bp'] - df_feat['diastolic_bp']
df_feat['Autonomic_Stress_Ratio'] = df_feat['RHR_FreeLiving_mean'] / (df_feat['HRV_FreeLiving_mean'] + 1e-4)
df_feat['TG_HDL_Ratio'] = df_feat['triglycerides'] / (df_feat['hdl'] + 1e-4)

# Full clean feature groups
mod_wearables = wearable_cols + ['Autonomic_Stress_Ratio']
mod_anthro = anthro_cols + ['sex_binary', 'WHtR', 'MAP', 'Pulse_Pressure']
mod_labs = routine_lab_cols + ['TG_HDL_Ratio']

tier1_cols = mod_wearables + mod_anthro
full_multimodal_cols = mod_wearables + mod_anthro + mod_labs

print(f"\nFeature counts:")
print(f"  Wearables Modality: {len(mod_wearables)} features")
print(f"  Anthropometrics & Vitals: {len(mod_anthro)} features")
print(f"  Routine Blood Labs: {len(mod_labs)} features")
print(f"  Tier 1 Non-Invasive (Wearables + Anthro): {len(tier1_cols)} features")
print(f"  Full Multimodal Fusion: {len(full_multimodal_cols)} features")

# 4. Strict 5-Fold Stratified Cross-Validation & SHAP Analysis
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_full = df_feat[full_multimodal_cols].values
feature_names = full_multimodal_cols

fold_top10 = {}
all_shap_values = []
all_test_indices = []

oof_preds_full = np.zeros(len(y_binary))
oof_probs_full = np.zeros(len(y_binary))

print("\n--- Running Strict 5-Fold Stratified Cross-Validation & SHAP Computation ---")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_binary), 1):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=16,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    oof_probs_full[test_idx] = probs
    oof_preds_full[test_idx] = (probs >= 0.5).astype(int)
    
    # Compute TreeSHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    
    # If binary, shap_values can be list [class0, class1] or array
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1]
    else:
        shap_class1 = shap_vals
        
    all_shap_values.append(shap_class1)
    all_test_indices.extend(test_idx)
    
    # Mean absolute SHAP per feature for this fold
    mean_abs_shap = np.mean(np.abs(shap_class1), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    
    fold_top10[f"Fold_{fold}"] = [
        {"Rank": rank + 1, "Feature": feature_names[idx], "Mean_Abs_SHAP": float(mean_abs_shap[idx])}
        for rank, idx in enumerate(top_indices)
    ]
    
    fold_auroc = roc_auc_score(y_test, probs)
    print(f"  Fold {fold}: AUROC = {fold_auroc:.4f} | Top Feature = {feature_names[top_indices[0]]} ({mean_abs_shap[top_indices[0]]:.4f})")

# Concatenate all SHAP values aligned with original dataframe index
all_shap_array = np.vstack(all_shap_values)
reordered_indices = np.argsort(all_test_indices)
global_shap_array = all_shap_array[reordered_indices]
global_mean_abs_shap = np.mean(np.abs(global_shap_array), axis=0)

# Global Top Features Ranking
global_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Global_Mean_Abs_SHAP': global_mean_abs_shap
}).sort_values('Global_Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

# Assign Modality
def get_mod(f):
    if f in mod_wearables:
        return 'Wearable Sensor'
    elif f in mod_anthro:
        return 'Anthropometrics & Vitals'
    else:
        return 'Routine Blood Lab'

global_ranking['Modality'] = global_ranking['Feature'].apply(get_mod)

print("\n--- Overall Top 10 Features (Across All Folds) ---")
print(global_ranking.head(10).to_string(index=False))

# Within-Modality Top 5 Features
print("\n--- Within-Modality Top 5 Features ---")
for mod_name in ['Wearable Sensor', 'Anthropometrics & Vitals', 'Routine Blood Lab']:
    sub = global_ranking[global_ranking['Modality'] == mod_name].head(5)
    print(f"\n[{mod_name}]")
    print(sub.to_string(index=False))

# 5. Modality Importance Evaluation
print("\n" + "="*75)
print("5. EVALUATING IMPORTANCE OF EACH MODALITY")
print("="*75)

modality_dict = {
    'Wearables Alone (PPG, HRV, Steps, Sleep)': mod_wearables,
    'Anthropometrics & Vitals Alone': mod_anthro,
    'Routine Blood Labs Alone': mod_labs,
    'Tier 1 Non-Invasive (Wearables + Anthro)': tier1_cols,
    'Full Multimodal Fusion (All Streams)': full_multimodal_cols
}

modality_results = []
modality_oof_probs = {}

for m_name, m_cols in modality_dict.items():
    X_m = df_feat[m_cols].values
    oof_p = np.zeros(len(y_binary))
    
    for train_idx, test_idx in skf.split(X_m, y_binary):
        X_tr, X_te = X_m[train_idx], X_m[test_idx]
        y_tr, y_te = y_binary[train_idx], y_binary[test_idx]
        
        clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                 num_leaves=16, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        oof_p[test_idx] = clf.predict_proba(X_te)[:, 1]
        
    modality_oof_probs[m_name] = oof_p
    
    oof_pred = (oof_p >= 0.5).astype(int)
    auc = roc_auc_score(y_binary, oof_p)
    auprc = average_precision_score(y_binary, oof_p)
    acc = accuracy_score(y_binary, oof_pred)
    bacc = balanced_accuracy_score(y_binary, oof_pred)
    sens = recall_score(y_binary, oof_pred)
    spec = recall_score(1 - y_binary, 1 - oof_pred)
    f1 = f1_score(y_binary, oof_pred)
    
    modality_results.append({
        'Modality': m_name,
        'Num_Features': len(m_cols),
        'AUROC': round(auc, 4),
        'AUPRC': round(auprc, 4),
        'Accuracy': round(acc, 4),
        'Balanced_Acc': round(bacc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'F1_Score': round(f1, 4)
    })

mod_df = pd.DataFrame(modality_results)
print(mod_df.to_string(index=False))

# Calculate Modality SHAP Contribution Percentage
mod_shap_sum = global_ranking.groupby('Modality')['Global_Mean_Abs_SHAP'].sum().reset_index()
total_shap = mod_shap_sum['Global_Mean_Abs_SHAP'].sum()
mod_shap_sum['Contribution_Pct'] = round(mod_shap_sum['Global_Mean_Abs_SHAP'] / total_shap * 100, 2)
print("\n--- Modality SHAP Contribution Breakdown ---")
print(mod_shap_sum.to_string(index=False))

# 6. Weighted Fusion and Majority Vote Based Classification
print("\n" + "="*75)
print("6. WEIGHTED FUSION & MAJORITY VOTE CLASSIFICATION")
print("="*75)

# Extract predictions from the 3 separate modality models
p_wear = modality_oof_probs['Wearables Alone (PPG, HRV, Steps, Sleep)']
p_anthro = modality_oof_probs['Anthropometrics & Vitals Alone']
p_labs = modality_oof_probs['Routine Blood Labs Alone']

pred_wear = (p_wear >= 0.5).astype(int)
pred_anthro = (p_anthro >= 0.5).astype(int)
pred_labs = (p_labs >= 0.5).astype(int)

# A. Majority Vote Classification (Hard Voting: >= 2 votes)
majority_vote_pred = ((pred_wear + pred_anthro + pred_labs) >= 2).astype(int)
majority_vote_prob = (p_wear + p_anthro + p_labs) / 3.0

# B. Equal Weighted Soft Voting (Weights: 0.333, 0.333, 0.333)
soft_vote_equal_prob = (p_wear + p_anthro + p_labs) / 3.0
soft_vote_equal_pred = (soft_vote_equal_prob >= 0.5).astype(int)

# C. Performance-Weighted Probability Fusion
auc_w = roc_auc_score(y_binary, p_wear)
auc_a = roc_auc_score(y_binary, p_anthro)
auc_l = roc_auc_score(y_binary, p_labs)
w_total = auc_w + auc_a + auc_l
w_w, w_a, w_l = auc_w / w_total, auc_a / w_total, auc_l / w_total

perf_weighted_prob = (w_w * p_wear) + (w_a * p_anthro) + (w_l * p_labs)
perf_weighted_pred = (perf_weighted_prob >= 0.5).astype(int)

# D. Grid-Search Optimal Weighted Probability Fusion (Optimizing AUROC)
best_auc = 0.0
best_weights = None
best_opt_prob = None

for w1 in np.linspace(0.05, 0.50, 10):
    for w2 in np.linspace(0.10, 0.60, 11):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        comb_p = w1 * p_wear + w2 * p_anthro + w3 * p_labs
        score = roc_auc_score(y_binary, comb_p)
        if score > best_auc:
            best_auc = score
            best_weights = (w1, w2, w3)
            best_opt_prob = comb_p

opt_weighted_pred = (best_opt_prob >= 0.5).astype(int)

# E. Concatenated Early Fusion (LGBM on all features)
early_fusion_prob = modality_oof_probs['Full Multimodal Fusion (All Streams)']
early_fusion_pred = (early_fusion_prob >= 0.5).astype(int)

# Compile Fusion Benchmark
fusion_schemes = [
    ('1. Wearables Alone Baseline', p_wear, pred_wear),
    ('2. Anthropometrics Alone Baseline', p_anthro, pred_anthro),
    ('3. Routine Labs Alone Baseline', p_labs, pred_labs),
    ('4. Majority Vote Classification (Hard Voting >= 2/3)', majority_vote_prob, majority_vote_pred),
    ('5. Equal Weighted Probability Fusion (1/3 each)', soft_vote_equal_prob, soft_vote_equal_pred),
    (f'6. Performance-Weighted Fusion (W:{w_w:.2f}, A:{w_a:.2f}, L:{w_l:.2f})', perf_weighted_prob, perf_weighted_pred),
    (f'7. Optimal Grid-Weighted Fusion (W:{best_weights[0]:.2f}, A:{best_weights[1]:.2f}, L:{best_weights[2]:.2f})', best_opt_prob, opt_weighted_pred),
    ('8. Concatenated Early Fusion (Gradient Boosting)', early_fusion_prob, early_fusion_pred)
]

fusion_results = []
for scheme_name, probs, preds in fusion_schemes:
    auc = roc_auc_score(y_binary, probs)
    auprc = average_precision_score(y_binary, probs)
    acc = accuracy_score(y_binary, preds)
    bacc = balanced_accuracy_score(y_binary, preds)
    sens = recall_score(y_binary, preds)
    spec = recall_score(1 - y_binary, 1 - preds)
    f1 = f1_score(y_binary, preds)
    
    fusion_results.append({
        'Ensemble_Fusion_Strategy': scheme_name,
        'AUROC': round(auc, 4),
        'AUPRC': round(auprc, 4),
        'Accuracy': round(acc, 4),
        'Balanced_Acc': round(bacc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'F1_Score': round(f1, 4)
    })

fusion_df = pd.DataFrame(fusion_results)
print(fusion_df.to_string(index=False))

# 7. Save Results
results_payload = {
    'target_statistics': status_stats.to_dict(),
    'fold_top10_features': fold_top10,
    'global_ranking': global_ranking.to_dict(orient='records'),
    'modality_benchmark': modality_results,
    'modality_shap_contribution': mod_shap_sum.to_dict(orient='records'),
    'fusion_benchmark': fusion_results,
    'best_fusion_weights': {
        'wearable': round(best_weights[0], 3),
        'anthro': round(best_weights[1], 3),
        'labs': round(best_weights[2], 3),
        'auroc': round(best_auc, 4)
    }
}

json_path = os.path.join(res_dir, 'google_health_detailed_evaluation.json')
with open(json_path, 'w') as f:
    json.dump(results_payload, f, indent=2)

csv_path = os.path.join(res_dir, 'google_health_fusion_benchmark.csv')
fusion_df.to_csv(csv_path, index=False)

ranking_csv = os.path.join(res_dir, 'google_health_shap_rankings.csv')
global_ranking.to_csv(ranking_csv, index=False)

print(f"\nAll results saved to {json_path}, {csv_path}, and {ranking_csv}!")
