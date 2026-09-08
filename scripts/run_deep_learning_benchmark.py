import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'nhanes')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Deep Learning Benchmark on: {DEVICE}")

# ── Model Architectures ──────────────────────────────────────────────────────

class Temporal1DCNN(nn.Module):
    """
    1D-CNN reading the (B, 3, 168) hourly actigraphy sequences.
    Captures multi-scale diurnal motifs (kernel=7: ~7h blocks, kernel=5, kernel=3).
    """
    def __init__(self, in_channels=3, static_dim=9, task='both'):
        super().__init__()
        self.task = task
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 168 -> 84
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 84 -> 42
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 128-dim temporal vector
        )
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.clf_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x_seq, x_static=None):
        # x_seq: (B, 168, 3) -> transpose to (B, 3, 168)
        x_in = x_seq.transpose(1, 2)
        feat = self.conv_net(x_in).squeeze(-1) # (B, 128)
        reg_out = self.reg_head(feat).squeeze(-1)
        clf_out = self.clf_head(feat)
        return reg_out, clf_out


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM reading the 168-hour time series sequentially.
    """
    def __init__(self, in_channels=3, hidden_size=64, task='both'):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        # Bi-directional produces hidden_size * 2 = 128 dims
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x_seq, x_static=None):
        # x_seq: (B, 168, 3)
        out, (hn, cn) = self.lstm(x_seq)
        # Global mean pool across 168 hours
        feat = torch.mean(out, dim=1) # (B, 128)
        reg_out = self.reg_head(feat).squeeze(-1)
        clf_out = self.clf_head(feat)
        return reg_out, clf_out


class DualBranchHybridFusion(nn.Module):
    """
    Dual-Branch Architecture:
    Branch 1: 1D-CNN temporal feature extractor on (B, 3, 168) hourly actigraphy.
    Branch 2: Dense MLP on 9 non-invasive vitals and demographics.
    Fusion Layer: Concatenation + Dense Multi-Task Head.
    """
    def __init__(self, seq_channels=3, static_dim=9):
        super().__init__()
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # (B, 64)
        )
        
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion: 64 (temporal) + 32 (static) = 96
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.reg_head = nn.Linear(64, 1)
        self.clf_head = nn.Linear(64, 3)

    def forward(self, x_seq, x_static):
        x_in = x_seq.transpose(1, 2)
        t_feat = self.temporal_branch(x_in).squeeze(-1) # (B, 64)
        s_feat = self.static_branch(x_static)          # (B, 32)
        
        fused = torch.cat([t_feat, s_feat], dim=1)      # (B, 96)
        rep = self.fusion(fused)                        # (B, 64)
        
        reg_out = self.reg_head(rep).squeeze(-1)
        clf_out = self.clf_head(rep)
        return reg_out, clf_out

# ── Training & Evaluation Functions ──────────────────────────────────────────

def train_model(model_cls, model_name, X_seq, X_stat, y_reg, y_clf, skf, epochs=25, lr=1e-3):
    print(f"\nTraining {model_name} across 5 Folds ...")
    N = len(y_reg)
    oof_reg = np.zeros(N)
    oof_clf_preds = np.zeros(N, dtype=int)
    oof_clf_probs = np.zeros((N, 3))
    
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_stat, y_clf)):
        # Normalize within fold
        # 1. Static features
        scaler_s = RobustScaler()
        X_stat_tr = scaler_s.fit_transform(X_stat[train_idx])
        X_stat_val = scaler_s.transform(X_stat[val_idx])
        
        # 2. Sequence features (channel-wise mean/std)
        X_seq_tr = X_seq[train_idx].copy()
        X_seq_val = X_seq[val_idx].copy()
        for c in range(3):
            c_mean = np.mean(X_seq_tr[:, :, c])
            c_std = np.std(X_seq_tr[:, :, c]) + 1e-6
            X_seq_tr[:, :, c] = (X_seq_tr[:, :, c] - c_mean) / c_std
            X_seq_val[:, :, c] = (X_seq_val[:, :, c] - c_mean) / c_std

        # PyTorch Tensors
        train_ds = TensorDataset(
            torch.tensor(X_seq_tr, dtype=torch.float32),
            torch.tensor(X_stat_tr, dtype=torch.float32),
            torch.tensor(y_reg[train_idx], dtype=torch.float32),
            torch.tensor(y_clf[train_idx], dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_seq_val, dtype=torch.float32),
            torch.tensor(X_stat_val, dtype=torch.float32),
            torch.tensor(y_reg[val_idx], dtype=torch.float32),
            torch.tensor(y_clf[val_idx], dtype=torch.long)
        )
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        
        model = model_cls().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion_reg = nn.SmoothL1Loss() # Robust to HOMA-IR outliers
        criterion_clf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.4, 1.0]).to(DEVICE))
        
        for epoch in range(epochs):
            model.train()
            for b_seq, b_stat, b_reg, b_clf in train_loader:
                b_seq, b_stat = b_seq.to(DEVICE), b_stat.to(DEVICE)
                b_reg, b_clf = b_reg.to(DEVICE), b_clf.to(DEVICE)
                
                optimizer.zero_grad()
                pred_reg, pred_clf = model(b_seq, b_stat)
                loss_reg = criterion_reg(pred_reg, b_reg)
                loss_clf = criterion_clf(pred_clf, b_clf)
                loss = loss_reg + 1.2 * loss_clf
                loss.backward()
                optimizer.step()
                
        # Validation inference
        model.eval()
        fold_reg_preds = []
        fold_clf_probs = []
        with torch.no_grad():
            for b_seq, b_stat, _, _ in val_loader:
                b_seq, b_stat = b_seq.to(DEVICE), b_stat.to(DEVICE)
                p_reg, p_clf = model(b_seq, b_stat)
                probs = torch.softmax(p_clf, dim=-1)
                fold_reg_preds.extend(p_reg.cpu().numpy())
                fold_clf_probs.extend(probs.cpu().numpy())
                
        oof_reg[val_idx] = np.array(fold_reg_preds)
        oof_clf_probs[val_idx] = np.array(fold_clf_probs)
        oof_clf_preds[val_idx] = np.argmax(np.array(fold_clf_probs), axis=1)

    elapsed = time.time() - t0
    
    # Calculate Metrics
    # Regression (HOMA-IR)
    mae = float(np.mean(np.abs(oof_reg - y_reg)))
    rmse = float(np.sqrt(np.mean((oof_reg - y_reg) ** 2)))
    r, _ = stats.pearsonr(y_reg, oof_reg)
    ss_res = np.sum((y_reg - oof_reg) ** 2)
    ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))
    
    # Bland-Altman
    diff = oof_reg - y_reg
    mean_bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    lower_loa = mean_bias - 1.96 * sd_diff
    upper_loa = mean_bias + 1.96 * sd_diff
    within_loa = float(np.mean((diff >= lower_loa) & (diff <= upper_loa)) * 100.0)

    # Classification (3-Class Risk)
    acc = accuracy_score(y_clf, oof_clf_preds) * 100.0
    b_acc = balanced_accuracy_score(y_clf, oof_clf_preds) * 100.0
    macro_f1 = f1_score(y_clf, oof_clf_preds, average='macro') * 100.0
    weighted_f1 = f1_score(y_clf, oof_clf_preds, average='weighted') * 100.0
    y_bin = label_binarize(y_clf, classes=[0, 1, 2])
    auc_ovr = roc_auc_score(y_bin, oof_clf_probs, multi_class='ovr', average='macro')

    print(f"  [{model_name}] Finished in {elapsed:.1f}s")
    print(f"    • HOMA-IR Regression: MAE={mae:.3f} | RMSE={rmse:.3f} | Pearson r={r:.3f} | R²={r2:.3f} | LoA %={within_loa:.1f}%")
    print(f"    • 3-Class Risk Screening: Bal Acc={b_acc:.2f}% | Macro F1={macro_f1:.2f}% | AUROC={auc_ovr:.3f}")

    return {
        'Model': model_name,
        'HOMA_MAE': round(mae, 3),
        'HOMA_RMSE': round(rmse, 3),
        'HOMA_Pearson_r': round(r, 3),
        'HOMA_R2': round(r2, 3),
        'HOMA_LoA_Pct': round(within_loa, 1),
        'Screening_Accuracy': round(acc, 2),
        'Screening_Bal_Acc': round(b_acc, 2),
        'Screening_Macro_F1': round(macro_f1, 2),
        'Screening_AUROC': round(auc_ovr, 3),
        'Time_Sec': round(elapsed, 2),
        'oof_reg': oof_reg,
        'oof_clf_preds': oof_clf_preds,
        'oof_clf_probs': oof_clf_probs
    }

def main():
    print("=" * 75)
    print("STAGE 1: DEEP TEMPORAL SEQUENCE MODELING ON 168-HOUR ACTIGRAPHY")
    print("1D-CNN vs. Bi-LSTM vs. Dual-Branch Hybrid Fusion Network")
    print("=" * 75)

    data_path = os.path.join(DATA_DIR, 'nhanes_168h_sequence_tensors.npz')
    data = np.load(data_path)
    X_seq = data['sequences']          # (N, 168, 3)
    X_stat = data['static_features']    # (N, 9)
    y_clf = data['y_3class']           # (N,) 0, 1, 2
    y_reg = data['y_homa']             # (N,) Continuous HOMA-IR
    
    print(f"Loaded Sequence Data: {X_seq.shape} | Static Features: {X_stat.shape}")
    print(f"Targets: {len(y_reg)} continuous HOMA-IR & {len(y_clf)} 3-class risk labels")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    
    # 1. 1D-CNN
    res_cnn = train_model(lambda: Temporal1DCNN(in_channels=3), "1D-CNN (Temporal ConvNet)", X_seq, X_stat, y_reg, y_clf, skf, epochs=20, lr=1e-3)
    results.append(res_cnn)

    # 2. Bi-LSTM
    res_lstm = train_model(lambda: BiLSTMModel(in_channels=3, hidden_size=64), "Bidirectional LSTM", X_seq, X_stat, y_reg, y_clf, skf, epochs=18, lr=1e-3)
    results.append(res_lstm)

    # 3. Dual-Branch Hybrid Fusion Network
    res_hybrid = train_model(lambda: DualBranchHybridFusion(seq_channels=3, static_dim=9), "Dual-Branch Hybrid Fusion Network", X_seq, X_stat, y_reg, y_clf, skf, epochs=22, lr=1e-3)
    results.append(res_hybrid)

    # Convert to summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Architecture': r['Model'],
            'HOMA-IR Pearson r': r['HOMA_Pearson_r'],
            'HOMA-IR R²': r['HOMA_R2'],
            'HOMA-IR MAE': r['HOMA_MAE'],
            'Screening Bal Acc (%)': r['Screening_Bal_Acc'],
            'Screening Macro F1 (%)': r['Screening_Macro_F1'],
            'Screening AUROC': r['Screening_AUROC'],
            'Training Time (s)': r['Time_Sec']
        })

    summary_df = pd.DataFrame(summary_data)
    out_csv = os.path.join(RES_DIR, 'deep_learning_benchmark.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved Deep Learning benchmark to: {out_csv}")

    # Generate Publication Figures
    generate_dl_figures(results, y_reg, y_clf)

def generate_dl_figures(results, y_reg, y_clf):
    print("\n--- Generating Deep Learning Comparison Figures ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    class_names = ['Low Risk (Normal)', 'Moderate Risk (Prediab)', 'High Risk (IR)']

    # 1. Load Tree Benchmark from Phase 2 to compare
    phase2_csv = os.path.join(RES_DIR, 'phase2_classification_benchmark.csv')
    phase1_csv = os.path.join(RES_DIR, 'phase1_regression_benchmark.csv')
    
    # Comparison Data
    models = ['Dual-Branch Hybrid Fusion', '1D-CNN (Temporal)', 'Bidirectional LSTM', 'XGBoost (Tabular)', 'LightGBM (Tabular)', 'Random Forest (Tabular)']
    bal_accs = [results[2]['Screening_Bal_Acc'], results[0]['Screening_Bal_Acc'], results[1]['Screening_Bal_Acc'], 54.85, 54.05, 54.81]
    aurocs = [results[2]['Screening_AUROC'], results[0]['Screening_AUROC'], results[1]['Screening_AUROC'], 0.737, 0.726, 0.732]
    homa_r = [results[2]['HOMA_Pearson_r'], results[0]['HOMA_Pearson_r'], results[1]['HOMA_Pearson_r'], 0.480, 0.482, 0.467]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar 1: Balanced Accuracy
    ax = axes[0]
    colors = ['#1F4E79', '#2E75B6', '#5B9BD5', '#ED7D31', '#F4B183', '#70AD47']
    bars = ax.barh(models, bal_accs, color=colors)
    ax.axvline(33.33, color='red', ls='--', lw=1.5, label='Chance Baseline (33.33%)')
    ax.set_xlim(0, 65)
    ax.set_title("3-Class Risk Screening: Balanced Accuracy (%)\nDeep Sequence Models vs. Gradient Boosted Trees", fontsize=12, fontweight='bold', color='#1F4E79')
    ax.set_xlabel("Balanced Accuracy (%)", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=10, fontweight='bold')

    # Bar 2: HOMA-IR Pearson r
    ax = axes[1]
    bars2 = ax.barh(models, homa_r, color=colors)
    ax.set_xlim(0, 0.65)
    ax.set_title("Continuous HOMA-IR Prediction: Pearson Correlation (r)\nTemporal Deep Learning vs. Hand-Engineered Features", fontsize=12, fontweight='bold', color='#1F4E79')
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=11, fontweight='bold')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"r = {w:.3f}", va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    comp_path = os.path.join(FIG_DIR, 'deep_vs_tree_model_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")

    # 2. Confusion Matrix for Dual-Branch Hybrid Fusion Network
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_clf, results[2]['oof_clf_preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', cbar=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title("Dual-Branch Hybrid Fusion Network\nNormalized Screening Confusion Matrix (%)", fontsize=12, fontweight='bold', color='#1F4E79')
    ax.set_ylabel("True Clinical Label", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Screening Label", fontsize=11, fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, 'deep_learning_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

if __name__ == '__main__':
    main()
