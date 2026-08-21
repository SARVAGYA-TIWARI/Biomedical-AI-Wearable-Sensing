import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(8)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from clarke_error_grid import evaluate_clarke_grid, plot_clarke_error_grid

# ----------------- Deep Learning PyTorch Architectures ----------------- #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)

class TemporalCNN(nn.Module):
    def __init__(self, input_dim, num_channels=[32, 32, 16], kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_c = out_c
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        conv_out = self.conv_net(x_trans)
        pooled = torch.mean(conv_out, dim=2)
        return self.fc(pooled).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer_encoder(encoded)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)

def train_pytorch_model(model_class, model_kwargs, X_train, y_train, X_val, y_val,
                        epochs=5, batch_size=512, lr=8e-3, patience=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**model_kwargs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_X_t = torch.FloatTensor(X_val).to(device)
    val_y_t = torch.FloatTensor(y_val).to(device)
    
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    
    model.train()
    for epoch in range(epochs):
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_t)
            val_loss = criterion(val_preds, val_y_t).item()
        model.train()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def make_sequence_tensor(df, lag_cols, multimodal_cols, seq_len=6):
    n = len(df)
    n_features = 1 + len(multimodal_cols)
    tensor = np.zeros((n, seq_len, n_features), dtype=np.float32)
    tensor[:, :, 0] = df[lag_cols].values
    for j, col in enumerate(multimodal_cols):
        tensor[:, :, 1 + j] = np.repeat(df[col].values[:, np.newaxis], seq_len, axis=1)
    return tensor

def run_phase1_loso_benchmark():
    os.makedirs(r"d:\BTP\results", exist_ok=True)
    os.makedirs(r"d:\BTP\figures", exist_ok=True)
    t_all0 = time.time()
    
    data_path = r"d:\BTP\data\d1namo_multimodal_master.parquet"
    if not os.path.exists(data_path):
        data_path = r"d:\BTP\data\d1namo_multimodal_master.csv"
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
        
    print(f"Loaded master dataset: {len(df)} rows across {df['subject_id'].nunique()} subjects", flush=True)
    
    glucose_lags = [f"glucose_lag_{i}min" for i in [30, 25, 20, 15, 10, 5]]
    glucose_stats = ["glucose_roll_mean_30min", "glucose_roll_std_30min", "glucose_roc_15min"]
    circadian_features = ["hour_sin", "hour_cos", "sin_2phi", "cos_2phi", "is_night"]
    glucose_only_features = glucose_lags + glucose_stats + circadian_features
    
    wearable_features = [c for c in ["HR_unified", "ECG_SDNN", "ECG_RMSSD", "DeviceTemp", "Activity", "BR",
                                    "HR_unified_roll15", "ECG_SDNN_roll15", "Activity_roll15", "DeviceTemp_roll15"]
                         if c in df.columns]
    multimodal_features = glucose_only_features + wearable_features
    
    df_clean = df.copy()
    for col in wearable_features:
        df_clean[col] = df_clean.groupby("subject_id")[col].ffill().bfill()
        df_clean[col] = df_clean[col].fillna(df_clean[col].median() if df_clean[col].count() > 0 else 0)
        
    subjects = sorted(df_clean["subject_id"].unique())
    print(f"LOSO Subjects: {subjects}", flush=True)
    
    all_results = []
    predictions_store = {}
    
    horizons = [("30min", "target_30min"), ("60min", "target_60min")]
    feature_modalities = [("Multimodal", multimodal_features), ("Glucose-Only", glucose_only_features)]
    
    for horizon_name, target_col in horizons:
        print(f"\n========================================================", flush=True)
        print(f"STARTING LOSO BENCHMARK: HORIZON = {horizon_name} ({target_col})", flush=True)
        print(f"========================================================", flush=True)
        
        valid_df = df_clean.dropna(subset=[target_col] + glucose_only_features).copy()
        
        for feat_name, feat_cols in feature_modalities:
            print(f"\n>>> Evaluating Feature Set: {feat_name} ({len(feat_cols)} features) <<<", flush=True)
            t_feat0 = time.time()
            
            models = {
                "Naive (persistence)": None,
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1),
                "XGBoost": XGBRegressor(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1),
                "LightGBM": LGBMRegressor(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1),
            }
            dl_models = ["LSTM", "GRU", "Temporal CNN", "Transformer"]
            
            pooled_preds = {m: ([], []) for m in list(models.keys()) + dl_models}
            subject_metrics = {m: [] for m in list(models.keys()) + dl_models}
            
            for test_subj in subjects:
                t_f = time.time()
                train_data = valid_df[valid_df["subject_id"] != test_subj]
                test_data = valid_df[valid_df["subject_id"] == test_subj]
                
                if len(test_data) < 10:
                    continue
                    
                X_tr = train_data[feat_cols].values
                y_tr = train_data[target_col].values
                X_te = test_data[feat_cols].values
                y_te = test_data[target_col].values
                
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)
                
                # 1. Classical
                for m_name, model in models.items():
                    if m_name == "Naive (persistence)":
                        y_pred = test_data["glucose_lag_5min"].values
                    else:
                        model.fit(X_tr_scaled, y_tr)
                        y_pred = model.predict(X_te_scaled)
                        
                    mae = mean_absolute_error(y_te, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                    subject_metrics[m_name].append({"subject": test_subj, "mae": mae, "rmse": rmse, "n": len(y_te)})
                    pooled_preds[m_name][0].extend(y_te)
                    pooled_preds[m_name][1].extend(y_pred)
                    
                # 2. Deep Learning
                val_subj = [s for s in subjects if s != test_subj][-1]
                sub_train = train_data[train_data["subject_id"] != val_subj]
                sub_val = train_data[train_data["subject_id"] == val_subj]
                
                extra_feats = [c for c in feat_cols if c not in glucose_lags]
                seq_train = make_sequence_tensor(sub_train, glucose_lags, extra_feats)
                seq_val = make_sequence_tensor(sub_val, glucose_lags, extra_feats)
                seq_test = make_sequence_tensor(test_data, glucose_lags, extra_feats)
                
                y_sub_tr = sub_train[target_col].values
                y_sub_val = sub_val[target_col].values
                
                n_feats = seq_train.shape[2]
                for f_idx in range(n_feats):
                    f_mean = np.mean(seq_train[:, :, f_idx])
                    f_std = np.std(seq_train[:, :, f_idx]) + 1e-6
                    seq_train[:, :, f_idx] = (seq_train[:, :, f_idx] - f_mean) / f_std
                    seq_val[:, :, f_idx] = (seq_val[:, :, f_idx] - f_mean) / f_std
                    seq_test[:, :, f_idx] = (seq_test[:, :, f_idx] - f_mean) / f_std
                    
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                X_test_t = torch.FloatTensor(seq_test).to(device)
                
                dl_configs = {
                    "LSTM": (LSTMModel, {"input_dim": n_feats, "hidden_dim": 24, "num_layers": 1, "dropout": 0.1}),
                    "GRU": (GRUModel, {"input_dim": n_feats, "hidden_dim": 24, "num_layers": 1, "dropout": 0.1}),
                    "Temporal CNN": (TemporalCNN, {"input_dim": n_feats, "num_channels": [24, 16], "kernel_size": 2, "dropout": 0.1}),
                    "Transformer": (TransformerModel, {"input_dim": n_feats, "d_model": 32, "nhead": 4, "num_layers": 1, "dropout": 0.1}),
                }
                
                for dl_name, (cls, kwargs) in dl_configs.items():
                    trained_net = train_pytorch_model(cls, kwargs, seq_train, y_sub_tr, seq_val, y_sub_val,
                                                      epochs=5, batch_size=512, lr=8e-3, patience=2)
                    trained_net.eval()
                    with torch.no_grad():
                        y_pred_dl = trained_net(X_test_t).cpu().numpy()
                        
                    mae = mean_absolute_error(y_te, y_pred_dl)
                    rmse = np.sqrt(mean_squared_error(y_te, y_pred_dl))
                    subject_metrics[dl_name].append({"subject": test_subj, "mae": mae, "rmse": rmse, "n": len(y_te)})
                    pooled_preds[dl_name][0].extend(y_te)
                    pooled_preds[dl_name][1].extend(y_pred_dl)
                    
                print(f"    Fold {test_subj} ({len(test_data)} points) evaluated in {time.time()-t_f:.1f}s", flush=True)
                
            for m_name in pooled_preds:
                y_true_all = np.array(pooled_preds[m_name][0])
                y_pred_all = np.array(pooled_preds[m_name][1])
                
                mae_pooled = mean_absolute_error(y_true_all, y_pred_all)
                rmse_pooled = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
                clarke_stats = evaluate_clarke_grid(y_true_all, y_pred_all)
                
                all_results.append({
                    "Horizon": horizon_name,
                    "Feature_Set": feat_name,
                    "Model": m_name,
                    "MAE": round(mae_pooled, 2),
                    "RMSE": round(rmse_pooled, 2),
                    "Zone_A_pct": round(clarke_stats['A'], 2),
                    "Zone_B_pct": round(clarke_stats['B'], 2),
                    "Zone_C_pct": round(clarke_stats['C'], 2),
                    "Zone_D_pct": round(clarke_stats['D'], 2),
                    "Zone_E_pct": round(clarke_stats['E'], 2),
                    "Clinical_Accuracy_AB": round(clarke_stats['A+B'], 2),
                    "N_Test_Points": len(y_true_all)
                })
                
                predictions_store[(horizon_name, feat_name, m_name)] = {
                    "y_true": y_true_all,
                    "y_pred": y_pred_all,
                    "subject_metrics": subject_metrics[m_name]
                }
                
                print(f"  [{horizon_name} | {feat_name}] {m_name:22s} -> MAE: {mae_pooled:6.2f} | RMSE: {rmse_pooled:6.2f} | Zone A+B: {clarke_stats['A+B']:5.1f}%")
                
            print(f"Feature set {feat_name} evaluated in {time.time()-t_feat0:.1f}s")
            
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(r"d:\BTP\results\phase1_loso_results.csv", index=False)
    with open(r"d:\BTP\results\phase1_predictions.pkl", "wb") as f:
        pickle.dump(predictions_store, f)
        
    print(f"\nPhase 1 LOSO Total Execution Time: {time.time() - t_all0:.1f}s")
    print("\n========================================================")
    print("PHASE 1 LOSO BENCHMARK RESULTS SUMMARY:")
    print("========================================================")
    print(results_df.to_string(index=False))
    
    # ----------------- Visualizations ----------------- #
    print("\nGenerating Phase 1 Figures...")
    
    # Figure 1: MAE Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    for ax, horizon in zip(axes, ["30min", "60min"]):
        sub = results_df[(results_df["Horizon"] == horizon) & (results_df["Feature_Set"] == "Multimodal")].sort_values("MAE")
        colors = ["#2ca02c" if m == "Naive (persistence)" else "#1f77b4" if "Regression" in m else
                  "#ff7f0e" if m in ["Random Forest", "XGBoost", "LightGBM"] else "#d62728" for m in sub["Model"]]
        bars = ax.barh(sub["Model"], sub["MAE"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"Leave-One-Subject-Out MAE — {horizon} Horizon", fontsize=12, fontweight='bold')
        ax.set_xlabel("Mean Absolute Error (mg/dL)", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va='center', fontsize=10, fontweight='bold')
    plt.suptitle("D1NAMO Multimodal Glucose Forecasting Benchmark (LOSO CV)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase1_mae_rmse_comparison.png", bbox_inches='tight')
    plt.close()
    
    # Figure 2: Clarke Error Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), dpi=200)
    for row_idx, horizon in enumerate(["30min", "60min"]):
        for col_idx, model_name in enumerate(["Linear Regression", "Temporal CNN"]):
            ax = axes[row_idx, col_idx]
            entry = predictions_store.get((horizon, "Multimodal", model_name))
            if entry:
                plot_clarke_error_grid(entry["y_true"], entry["y_pred"],
                                       title=f"{model_name} — {horizon} Horizon", ax=ax)
    plt.suptitle("Clarke Error Grid Clinical Analysis (LOSO Cross-Validation)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase1_clarke_grid_best_models.png", bbox_inches='tight')
    plt.close()
    
    # Figure 3: Multimodal vs Glucose-Only Ablation
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ablation_30 = results_df[results_df["Horizon"] == "30min"].copy()
    sns.barplot(data=ablation_30, x="Model", y="MAE", hue="Feature_Set", ax=ax, palette="Set2")
    ax.set_title("Ablation Study: Glucose-Only vs. Multimodal Features (30-min Horizon)", fontsize=12, fontweight='bold')
    ax.set_ylabel("MAE (mg/dL)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase1_multimodal_vs_glucose_ablation.png", bbox_inches='tight')
    plt.close()
    
    print("Phase 1 benchmark finished and figures saved successfully!")

if __name__ == '__main__':
    run_phase1_loso_benchmark()
