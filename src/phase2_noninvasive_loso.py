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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix

# ----------------- MLP PyTorch Classifier ----------------- #

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_classifier(input_dim, num_classes, X_train, y_train, X_val, y_val,
                         epochs=15, batch_size=128, lr=3e-3, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    
    # Class weights for imbalanced cross-entropy
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    weight_tensor = torch.FloatTensor(weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_X_t = torch.FloatTensor(X_val).to(device)
    val_y_t = torch.LongTensor(y_val).to(device)
    
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    
    model.train()
    for epoch in range(epochs):
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss = criterion(val_logits, val_y_t).item()
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

# ----------------- Majority Baseline Classifier ----------------- #

class MajorityClassifier:
    def __init__(self):
        self.majority_class_ = None
        
    def fit(self, X, y):
        counts = np.bincount(y)
        self.majority_class_ = np.argmax(counts)
        return self
        
    def predict(self, X):
        return np.full(len(X), self.majority_class_)

# ----------------- Main Phase 2 LOSO Pipeline ----------------- #

def run_phase2_noninvasive_benchmark():
    os.makedirs(r"d:\BTP\results", exist_ok=True)
    os.makedirs(r"d:\BTP\figures", exist_ok=True)
    
    data_path = r"d:\BTP\data\d1namo_multimodal_master.parquet"
    if not os.path.exists(data_path):
        data_path = r"d:\BTP\data\d1namo_multimodal_master.csv"
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
        
    print(f"Loaded master dataset: {len(df)} rows across {df['subject_id'].nunique()} subjects")
    
    # Strictly non-glucose wearable and circadian features
    non_glucose_features = [
        "hour_sin", "hour_cos", "sin_2phi", "cos_2phi", "is_night", "day_of_week",
        "HR_unified", "ECG_SDNN", "ECG_RMSSD", "DeviceTemp", "Activity", "BR",
        "HR_unified_roll15", "ECG_SDNN_roll15", "Activity_roll15", "DeviceTemp_roll15"
    ]
    non_glucose_features = [c for c in non_glucose_features if c in df.columns]
    
    # Impute missing wearable sensors with forward-fill / median
    df_clean = df.copy()
    for col in non_glucose_features:
        df_clean[col] = df_clean.groupby("subject_id")[col].ffill().bfill()
        df_clean[col] = df_clean[col].fillna(df_clean[col].median() if df_clean[col].count() > 0 else 0)
        
    subjects = sorted(df_clean["subject_id"].unique())
    
    tasks = [
        ("Range_Classification", "range_cgm", ["Hypoglycemic (<70)", "Target (70-180)", "Hyperglycemic (>180)"]),
        ("Trend_30min", "trend_30min", ["Decreasing (<-1.0)", "Stable (-1.0 to 1.0)", "Increasing (>1.0)"]),
        ("Trend_60min", "trend_60min", ["Decreasing (<-1.0)", "Stable (-1.0 to 1.0)", "Increasing (>1.0)"]),
    ]
    
    all_results = []
    confusion_matrices = {}
    feature_importances = {}
    
    for task_name, target_col, class_names in tasks:
        print(f"\n========================================================")
        print(f"STARTING PHASE 2 LOSO EVALUATION: {task_name} ({target_col})")
        print(f"Classes: {class_names}")
        print(f"========================================================")
        
        valid_df = df_clean.dropna(subset=[target_col] + non_glucose_features).copy()
        valid_df[target_col] = valid_df[target_col].astype(int)
        
        class_dist = valid_df[target_col].value_counts().to_dict()
        print(f"Class Distribution: {class_dist}")
        
        models = {
            "Majority Baseline": MajorityClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='mlogloss', n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
            "Neural Network (MLP)": None,  # Handled via PyTorch
        }
        
        pooled_preds = {m: ([], []) for m in models}
        subject_metrics = {m: [] for m in models}
        
        # LOSO Cross-Validation Loop
        for test_subj in subjects:
            train_data = valid_df[valid_df["subject_id"] != test_subj]
            test_data = valid_df[valid_df["subject_id"] == test_subj]
            
            if len(test_data) < 10 or len(train_data) < 20:
                continue
                
            X_tr = train_data[non_glucose_features].values
            y_tr = train_data[target_col].values
            X_te = test_data[non_glucose_features].values
            y_te = test_data[target_col].values
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            # Classical Models
            for m_name, model in models.items():
                if m_name == "Neural Network (MLP)":
                    # PyTorch MLP
                    val_subj = [s for s in subjects if s != test_subj][-1]
                    sub_tr = train_data[train_data["subject_id"] != val_subj]
                    sub_val = train_data[train_data["subject_id"] == val_subj]
                    
                    X_sub_tr = scaler.transform(sub_tr[non_glucose_features].values)
                    y_sub_tr = sub_tr[target_col].values
                    X_sub_val = scaler.transform(sub_val[non_glucose_features].values)
                    y_sub_val = sub_val[target_col].values
                    
                    mlp_net = train_mlp_classifier(input_dim=len(non_glucose_features), num_classes=len(class_names),
                                                   X_train=X_sub_tr, y_train=y_sub_tr,
                                                   X_val=X_sub_val, y_val=y_sub_val,
                                                   epochs=15, batch_size=128, lr=3e-3, patience=3)
                    mlp_net.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    with torch.no_grad():
                        logits = mlp_net(torch.FloatTensor(X_te_scaled).to(device))
                        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    model.fit(X_tr_scaled, y_tr)
                    y_pred = model.predict(X_te_scaled)
                    
                bacc = balanced_accuracy_score(y_te, y_pred)
                macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
                
                subject_metrics[m_name].append({
                    "subject": test_subj,
                    "balanced_acc": bacc,
                    "macro_f1": macro_f1,
                    "n": len(y_te)
                })
                pooled_preds[m_name][0].extend(y_te)
                pooled_preds[m_name][1].extend(y_pred)
                
        # Store Feature Importance for Random Forest
        rf_full = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        X_all_scaled = StandardScaler().fit_transform(valid_df[non_glucose_features].values)
        rf_full.fit(X_all_scaled, valid_df[target_col].values)
        feature_importances[task_name] = pd.Series(rf_full.feature_importances_, index=non_glucose_features).sort_values(ascending=False)
        
        # Calculate Pooled Metrics
        for m_name in models:
            y_true_all = np.array(pooled_preds[m_name][0])
            y_pred_all = np.array(pooled_preds[m_name][1])
            
            bacc_pooled = balanced_accuracy_score(y_true_all, y_pred_all)
            macro_f1_pooled = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
            weighted_f1_pooled = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(class_names))))
            confusion_matrices[(task_name, m_name)] = cm
            
            all_results.append({
                "Task": task_name,
                "Model": m_name,
                "Balanced_Accuracy": round(bacc_pooled * 100.0, 2),
                "Macro_F1": round(macro_f1_pooled * 100.0, 2),
                "Weighted_F1": round(weighted_f1_pooled * 100.0, 2),
                "N_Test_Points": len(y_true_all)
            })
            
            print(f"  {m_name:24s} -> Balanced Acc: {bacc_pooled*100:5.2f}% | Macro F1: {macro_f1_pooled*100:5.2f}%")
            
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(r"d:\BTP\results\phase2_loso_results.csv", index=False)
    
    print("\n========================================================")
    print("PHASE 2 LOSO BENCHMARK RESULTS SUMMARY:")
    print("========================================================")
    print(results_df.to_string(index=False))
    
    # ----------------- Visualizations ----------------- #
    print("\nGenerating Phase 2 Figures...")
    
    # Figure 1: Balanced Accuracy by Task and Model
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    for ax, (task_name, _, _) in zip(axes, tasks):
        sub = results_df[results_df["Task"] == task_name].sort_values("Balanced_Accuracy")
        colors = ["#7f7f7f" if m == "Majority Baseline" else "#1f77b4" if "Logistic" in m else
                  "#ff7f0e" if m in ["Random Forest", "XGBoost", "LightGBM"] else "#2ca02c" for m in sub["Model"]]
        bars = ax.barh(sub["Model"], sub["Balanced_Accuracy"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"{task_name.replace('_', ' ')}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Balanced Accuracy (%)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.axvline(33.3, color="red", linestyle=":", label="Random Chance (33.3%)")
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')
    plt.suptitle("Non-Invasive Glucose Estimation Performance (LOSO Cross-Validation)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase2_balanced_accuracy_comparison.png", bbox_inches='tight')
    plt.close()
    
    # Figure 2: Confusion Matrices for Range Classification and Trend Prediction
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)
    
    cm_range = confusion_matrices.get(("Range_Classification", "Random Forest"))
    if cm_range is not None:
        sns.heatmap(cm_range, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=["Low (<70)", "Target (70-180)", "High (>180)"],
                    yticklabels=["Low (<70)", "Target (70-180)", "High (>180)"])
        axes[0].set_title("Range Classification — Random Forest (LOSO)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Predicted Class", fontsize=10)
        axes[0].set_ylabel("Actual Class", fontsize=10)
        
    cm_trend = confusion_matrices.get(("Trend_30min", "Random Forest"))
    if cm_trend is not None:
        sns.heatmap(cm_trend, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                    xticklabels=["Decreasing", "Stable", "Increasing"],
                    yticklabels=["Decreasing", "Stable", "Increasing"])
        axes[1].set_title("30-min Trend Prediction — Random Forest (LOSO)", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Predicted Class", fontsize=10)
        axes[1].set_ylabel("Actual Class", fontsize=10)
        
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase2_confusion_matrices.png", bbox_inches='tight')
    plt.close()
    
    # Figure 3: Feature Importance Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    for ax, (task_name, title) in zip(axes, [("Range_Classification", "Range Classification"), ("Trend_30min", "30-min Trend Prediction")]):
        if task_name in feature_importances:
            s = feature_importances[task_name].head(10)
            sns.barplot(x=s.values, y=s.index, ax=ax, palette="mako")
            ax.set_title(f"Top 10 Feature Importances — {title}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=10)
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(r"d:\BTP\figures\phase2_feature_importances.png", bbox_inches='tight')
    plt.close()
    
    print("Phase 2 benchmark finished and figures saved successfully!")

if __name__ == '__main__':
    run_phase2_noninvasive_benchmark()
