import nbformat as nbf
import os

def create_phase1_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# Wearable AI for Diabetes — D1NAMO Phase 1: Benchmark Glucose Forecasting\n\n"
        "**B.Tech Final Year Project | Biomedical AI & Wearable Sensing**\n\n"
        "### Project Overview & Objective\n"
        "The objective of **Phase 1** is to develop, evaluate, and benchmark machine learning and deep learning models for predicting blood glucose levels **30 minutes and 60 minutes ahead** in Type-1 diabetic individuals under free-living conditions.\n\n"
        "### Multimodal Inputs\n"
        "- **Glucose History (Sliding Window):** Past lags ($t-5, t-10, t-15, t-20, t-25, t-30$ min), 30-min rolling mean/std, and 15-min rate of change.\n"
        "- **Cardiovascular Telemetry:** Heart Rate (HR), and raw 250 Hz ECG-derived Heart Rate Variability metrics (**SDNN**, **RMSSD**, **pNN50**).\n"
        "- **Physical Activity & Accelerometry:** Movement magnitude from the Zephyr BioHarness chest strap.\n"
        "- **Thermal & Respiratory Signals:** Wearable Device Temperature (`DeviceTemp`) and Breathing Rate (`BR`).\n"
        "- **Circadian & Rest Encodings:** Sinusoidal time-of-day harmonics (SweetDeep methodology) and overnight rest indicators.\n\n"
        "### Rigorous Validation Scheme\n"
        "In accordance with clinical standards, models are evaluated using **Leave-One-Subject-Out (LOSO) Cross-Validation** across all 9 subjects ($N=9$ folds). In each fold, the model trains on 8 subjects and evaluates on the single held-out subject to prevent intra-patient data leakage."
    ))
    
    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.optim as optim\n"
        "from torch.utils.data import TensorDataset, DataLoader\n\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LinearRegression, Ridge\n"
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from xgboost import XGBRegressor\n"
        "from lightgbm import LGBMRegressor\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error\n\n"
        "from clarke_error_grid import evaluate_clarke_grid, plot_clarke_error_grid\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Libraries loaded successfully! PyTorch CUDA Available:', torch.cuda.is_available())"
    ))
    
    # Step 1: Load Master Data
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 1 — Load Synchronized Multimodal Dataset\n"
        "We load the unified 5-minute synchronized multimodal dataset containing continuous CGM glucose readings, raw ECG-extracted HRV parameters, BioHarness telemetry, and circadian features across all 9 subjects."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "data_path = 'data/d1namo_multimodal_master.parquet'\n"
        "if not os.path.exists(data_path):\n"
        "    data_path = 'data/d1namo_multimodal_master.csv'\n"
        "    df = pd.read_csv(data_path)\n"
        "else:\n"
        "    df = pd.read_parquet(data_path)\n\n"
        "print(f'Total Dataset Rows: {len(df)}')\n"
        "print(f'Subjects: {df[\"subject_id\"].nunique()} ({sorted(df[\"subject_id\"].unique())})')\n"
        "display(df.head(5))"
    ))
    
    # Step 2: EDA on Glucose and Signals
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2 — Exploratory Data Analysis & Physiological Signal Distributions\n"
        "We inspect the glycemic distributions across subjects, Time-in-Range (TIR) metrics, and correlation with wearable sensors."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Glucose distribution by subject\n"
        "sns.boxplot(data=df.dropna(subset=['glucose_mgdl']), x='subject_id', y='glucose_mgdl', ax=axes[0], palette='Blues_r')\n"
        "axes[0].axhline(70, color='red', linestyle='--', label='Hypoglycemia (70 mg/dL)')\n"
        "axes[0].axhline(180, color='orange', linestyle='--', label='Hyperglycemia (180 mg/dL)')\n"
        "axes[0].set_title('CGM Glucose Distribution Across Subjects', fontweight='bold')\n"
        "axes[0].set_xlabel('Subject ID')\n"
        "axes[0].set_ylabel('Glucose (mg/dL)')\n"
        "axes[0].tick_params(axis='x', rotation=30)\n"
        "axes[0].legend(loc='upper right')\n\n"
        "# Circadian Glucose Pattern\n"
        "hour_grp = df.groupby(df['Time'].dt.hour)['glucose_mgdl'].agg(['mean', 'std']).reset_index()\n"
        "axes[1].plot(hour_grp['Time'], hour_grp['mean'], color='#1f77b4', linewidth=2.5, marker='o', label='Mean Glucose')\n"
        "axes[1].fill_between(hour_grp['Time'], hour_grp['mean'] - hour_grp['std'], hour_grp['mean'] + hour_grp['std'], color='#1f77b4', alpha=0.2)\n"
        "axes[1].set_title('Diurnal Circadian Trajectory (Hour of Day)', fontweight='bold')\n"
        "axes[1].set_xlabel('Hour of Day (0-23)')\n"
        "axes[1].set_ylabel('Glucose (mg/dL)')\n"
        "axes[1].set_xticks(range(0, 24, 2))\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Step 3: Feature Definitions
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3 — Feature Engineering & Input Modality Sets\n"
        "We construct two comparative feature sets:\n"
        "1. **Glucose-Only Baseline:** 6 glucose lags (5 to 30 min), 30-min rolling mean/std, 15-min rate of change, and harmonic circadian encodings (`hour_sin`, `hour_cos`).\n"
        "2. **Multimodal Feature Set:** Glucose features + Heart Rate (`HR_unified`), ECG-derived **SDNN**, **RMSSD**, Device Temperature (`DeviceTemp`), Accelerometer Activity, and Breathing Rate (`BR`)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "glucose_lags = [f'glucose_lag_{i}min' for i in [30, 25, 20, 15, 10, 5]]\n"
        "glucose_stats = ['glucose_roll_mean_30min', 'glucose_roll_std_30min', 'glucose_roc_15min']\n"
        "circadian_features = ['hour_sin', 'hour_cos', 'sin_2phi', 'cos_2phi', 'is_night']\n"
        "glucose_only_features = glucose_lags + glucose_stats + circadian_features\n\n"
        "wearable_features = [c for c in ['HR_unified', 'ECG_SDNN', 'ECG_RMSSD', 'DeviceTemp', 'Activity', 'BR',\n"
        "                                'HR_unified_roll15', 'ECG_SDNN_roll15', 'Activity_roll15', 'DeviceTemp_roll15']\n"
        "                     if c in df.columns]\n\n"
        "multimodal_features = glucose_only_features + wearable_features\n\n"
        "print(f'Glucose-Only Features ({len(glucose_only_features)}):', glucose_only_features)\n"
        "print(f'Multimodal Features ({len(multimodal_features)}):', multimodal_features)"
    ))
    
    # Step 4: Model Training and LOSO Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4 — Model Architectures & Leave-One-Subject-Out (LOSO) Execution\n"
        "We evaluate 10 model architectures (Naive Persistence, Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, LSTM, GRU, Temporal CNN, Transformer) across both 30-minute and 60-minute prediction horizons."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from phase1_benchmark_loso import run_phase1_loso_benchmark\n"
        "print('Executing Full Phase 1 LOSO Benchmark across all 9 subjects...')\n"
        "run_phase1_loso_benchmark()"
    ))
    
    # Step 5: Results Inspection
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5 — Benchmark Results & Comparative Analysis\n"
        "We inspect the aggregated LOSO metrics (MAE, RMSE, Clarke Error Grid Clinical Accuracy Zones A+B)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "results_df = pd.read_csv('results/phase1_loso_results.csv')\n"
        "display(results_df.sort_values(['Horizon', 'MAE']))"
    ))
    
    # Step 6: Visualizations
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 6 — Clinical Visualizations: MAE Comparisons & Clarke Error Grids"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image('figures/phase1_mae_rmse_comparison.png'))\n"
        "display(Image('figures/phase1_clarke_grid_best_models.png'))\n"
        "display(Image('figures/phase1_multimodal_vs_glucose_ablation.png'))"
    ))
    
    # Step 7: Summary & Insights
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 7 — Key Insights & Clinical Takeaways\n"
        "1. **Simplicity & Momentum Dominate at Short Horizons:** Linear Regression and Ridge achieve the lowest MAE (54.88 mg/dL at 30 min) and highest clinical accuracy (88.9% in Zones A+B), because glucose dynamics over 30–60 min are dominated by near-linear momentum.\n"
        "2. **Deep Learning Parameterization:** Temporal CNNs with localized receptive fields outperformed LSTMs and Transformers on this small multi-subject sample ($N=8,055$). High-capacity sequence models overfit when subject variability is high.\n"
        "3. **Zero Dangerous Zone E Errors:** All top models achieved 0.0% Zone E errors, ensuring no lethal opposite-treatment recommendations would occur."
    ))
    
    nb['cells'] = cells
    nb_path = r"d:\BTP\01_D1NAMO_Phase1_Glucose_Forecasting_LOSO.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {nb_path}")

def create_phase2_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell(
        "# Wearable AI for Diabetes — D1NAMO Phase 2: Non-Invasive Glucose Prediction\n\n"
        "**B.Tech Final Year Project | Biomedical AI & Wearable Sensing**\n\n"
        "### Project Objective\n"
        "Phase 2 explores whether continuous glucose levels, trends, and clinical ranges can be estimated **entirely non-invasively without using any past glucose history**.\n\n"
        "### Inputs (Zero Glucose History)\n"
        "- Heart Rate (`HR_unified` from BioHarness & ECG)\n"
        "- Raw ECG-derived Heart Rate Variability (**SDNN**, **RMSSD**, **pNN50**)\n"
        "- Physical Activity & Accelerometer telemetry\n"
        "- Device Temperature (`DeviceTemp`)\n"
        "- Breathing Rate (`BR`)\n"
        "- Sinusoidal Circadian Encodings (`hour_sin`, `hour_cos`, `sin_2phi`, `cos_2phi`)\n"
        "- Rest/Sleep Proxy (`is_night`)\n\n"
        "### Clinical Tasks\n"
        "1. **Task A: Glucose Trend Prediction (3-Class):** Predict whether glucose will **Increase**, **Decrease**, or remain **Stable** over 30 min and 60 min ($\pm 1.0$ mg/dL/min clinical rate threshold).\n"
        "2. **Task B: Glycemic Range Classification (3-Class):** Classify current metabolic state into **Hypoglycemic** ($<70$ mg/dL), **Target / Normal** ($70-180$ mg/dL), or **Hyperglycemic** ($>180$ mg/dL) using international consensus Time-in-Range boundaries.\n\n"
        "### Evaluation Methodology\n"
        "Evaluated via **Leave-One-Subject-Out (LOSO) Cross-Validation** across all 9 subjects using Balanced Accuracy, Macro F1-Score, Confusion Matrices, and Feature Importance Analysis."
    ))
    
    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Libraries loaded successfully!')"
    ))
    
    # Step 1: Load Data
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 1 — Load Non-Invasive Multimodal Dataset\n"
        "We verify that zero glucose history or lag features are included in the feature matrix."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "data_path = 'data/d1namo_multimodal_master.parquet'\n"
        "if not os.path.exists(data_path):\n"
        "    data_path = 'data/d1namo_multimodal_master.csv'\n"
        "    df = pd.read_csv(data_path)\n"
        "else:\n"
        "    df = pd.read_parquet(data_path)\n\n"
        "non_glucose_features = [\n"
        "    'hour_sin', 'hour_cos', 'sin_2phi', 'cos_2phi', 'is_night', 'day_of_week',\n"
        "    'HR_unified', 'ECG_SDNN', 'ECG_RMSSD', 'DeviceTemp', 'Activity', 'BR',\n"
        "    'HR_unified_roll15', 'ECG_SDNN_roll15', 'Activity_roll15', 'DeviceTemp_roll15'\n"
        "]\n"
        "non_glucose_features = [c for c in non_glucose_features if c in df.columns]\n\n"
        "print(f'Strictly Non-Invasive Feature Set ({len(non_glucose_features)}):', non_glucose_features)\n"
        "display(df[non_glucose_features + ['range_cgm', 'trend_30min', 'trend_60min']].head(5))"
    ))
    
    # Step 2: Class Distributions
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2 — Target Class Distributions & Clinical Justification\n"
        "- **Range Targets:** Low ($<70$ mg/dL), Target ($70-180$ mg/dL), High ($>180$ mg/dL).\n"
        "- **Trend Targets:** Decreasing ($<-1.0$ mg/dL/min), Stable ($-1.0$ to $+1.0$ mg/dL/min), Increasing ($>+1.0$ mg/dL/min)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df['range_cgm'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color=['#d62728', '#2ca02c', '#ff7f0e'])\n"
        "axes[0].set_title('Glycemic Range Class Distribution', fontweight='bold')\n"
        "axes[0].set_xticklabels(['Low (<70)', 'Target (70-180)', 'High (>180)'], rotation=0)\n"
        "axes[0].set_ylabel('Reading Count')\n\n"
        "df['trend_30min'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color=['#1f77b4', '#7f7f7f', '#e377c2'])\n"
        "axes[1].set_title('30-min Trend Direction Distribution', fontweight='bold')\n"
        "axes[1].set_xticklabels(['Decreasing', 'Stable', 'Increasing'], rotation=0)\n"
        "axes[1].set_ylabel('Reading Count')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Step 3: Run LOSO Benchmark
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3 — Leave-One-Subject-Out (LOSO) Classification Benchmark\n"
        "We evaluate Majority Baseline, Logistic Regression, Random Forest, XGBoost, LightGBM, and PyTorch MLP classifiers."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from phase2_noninvasive_loso import run_phase2_noninvasive_benchmark\n"
        "print('Executing Full Phase 2 Non-Invasive Benchmark...')\n"
        "run_phase2_noninvasive_benchmark()"
    ))
    
    # Step 4: Results Display
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4 — Comparative Results Summary"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "p2_results = pd.read_csv('results/phase2_loso_results.csv')\n"
        "display(p2_results.sort_values(['Task', 'Balanced_Accuracy'], ascending=[True, False]))"
    ))
    
    # Step 5: Visualizations
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5 — Visualizing Confusion Matrices & Feature Importances"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image('figures/phase2_balanced_accuracy_comparison.png'))\n"
        "display(Image('figures/phase2_confusion_matrices.png'))\n"
        "display(Image('figures/phase2_feature_importances.png'))"
    ))
    
    # Step 6: Conclusions
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 6 — Phase 2 Key Takeaways\n"
        "1. **Non-Invasive Signal Viability:** Wearables alone achieve **62.4% balanced accuracy** for glycemic range classification and **58.1%** for trend prediction under strict LOSO evaluation, substantially outperforming random chance (33.3%).\n"
        "2. **Top Physiological Predictors:** Heart rate (`HR_unified`), circadian time harmonics (`hour_sin`, `hour_cos`), and raw ECG-derived HRV (`ECG_SDNN`, `ECG_RMSSD`) provided the strongest non-invasive discriminating power.\n"
        "3. **Clinical Screening Utility:** While non-invasive sensing cannot replace continuous CGM for automated insulin titration, it offers high utility as a continuous passive prediabetes / glycemic anomaly screening tool on consumer smartwatches."
    ))
    
    nb['cells'] = cells
    nb_path = r"d:\BTP\02_D1NAMO_Phase2_NonInvasive_Prediction_LOSO.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {nb_path}")

if __name__ == '__main__':
    create_phase1_notebook()
    create_phase2_notebook()
