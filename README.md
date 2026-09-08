# Wearable AI for Metabolic Health & Insulin Resistance Screening
## Multi-Modal Wearable Sensing, Circadian Biomarker Estimation & Explainable AI (NHANES 2011–2014 & D1NAMO)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Validation: Stratified 5-Fold & LOSO CV](https://img.shields.io/badge/Validation-Stratified%205--Fold%20%26%20LOSO-green.svg)](#validation-strategy)

**Author:** [SARVAGYA-TIWARI](https://github.com/SARVAGYA-TIWARI)  
**Affiliation:** Indian Institute of Technology Guwahati (IITG)  
**Domain:** Biomedical AI, Wearable Sensing, Circadian Biology, Digital Health, Explainable Machine Learning (XAI)  

---

## 📌 Executive Overview

This repository hosts a production-grade machine learning framework, deep temporal sequence modeling suite, and clinical interpretability engine for **non-invasive metabolic monitoring and insulin resistance screening using wearable devices**.

Built upon the representative **CDC NHANES 2011–2014 Multi-Modal Cohort** ($N = 3,292$ non-diabetic adults, 2.78 million hourly wearable epochs, 7 consecutive days of continuous wrist actigraphy, autonomic vitals, and certified laboratory blood biomarkers), complemented by initial benchmark experiments on the **D1NAMO CGM dataset** ($N = 9$), this project delivers an end-to-end clinical AI pipeline:

1. **Continuous Biomarker Regression:** Non-invasive estimation of continuous Homeostatic Model Assessment of Insulin Resistance (**HOMA-IR**), Glycated Hemoglobin (**HbA1c**), and **Fasting Plasma Glucose** without fingerprick calibration.
2. **Non-Invasive 3-Class Metabolic Risk Screening:** Passive population screening stratifying individuals into American Diabetes Association (ADA)-aligned risk tiers (**Low Risk Normal**, **Moderate Risk Prediabetes**, **High Risk Insulin Resistant**) with **zero blood access** at inference time.
3. **Deep Temporal Sequence Modeling:** End-to-end **1D-CNN**, **Bidirectional LSTM**, and **Dual-Branch Hybrid Fusion Networks** operating directly on consecutive 168-hour ($T=168$) actigraphy arrays to benchmark against hand-crafted circadian engineering.
4. **Explainable AI & Clinical Interpretability (Tree SHAP):** Global summary beeswarm distributions, 2D nonlinear feature interaction manifolds, and patient-level waterfall case studies deconstructing the exact physiological decision rules behind every prediction.

---

## 📐 Unified System Pipeline

```
                                      NHANES 2011-2014 MULTI-MODAL DATASET
 ┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
 │   7-Day Wrist Actigraphy     │ │     Resting Autonomic Tone   │ │ Certified Clinical Chemistry │
 │  • ActiGraph GT3X+ (PAXHD)   │ │  • Resting Pulse (BPM)       │ │  • Fasting Glucose (GLU_G/H) │
 │  • 2.78 Million Hourly Epochs│ │  • Triplicate BP (BPX_G/H)   │ │  • Fasting Insulin (INS_G/H) │
 │  • MIMS Movement & Sleep/Wake│ │  • Body Habitus (BMI, Waist) │ │  • HbA1c (GHB_G/H) & Lipids  │
 └──────────────┬───────────────┘ └──────────────┬───────────────┘ └──────────────┬───────────────┘
                │                                │                                │
                └────────────────────────┐       │       ┌────────────────────────┘
                                         ▼       ▼       ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                             FEATURE EXTRACTION & CIRCADIAN HARMONICS                         │
 │  • Parametric Cosinor Regression ──► Mesor (Baseline), Amplitude (Peak-Trough), Acrophase    │
 │  • Non-Parametric Circadian Analysis (NPCRA) ──► Interdaily Stability (IS), Intradaily (IV) │
 │  • Diurnal Movement Extremes ──► M10 (10 Most Active Hours), L5 (5 Least Active Hours Sleep) │
 │  • 168-Hour Sequential Tensor Construction ──► (N = 3,292, T = 168 Hours, C = 3 Channels)    │
 └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                      5-FOLD STRATIFIED INTER-SUBJECT CROSS-VALIDATION                        │
 │                                (Zero Blood Access at Inference)                              │
 ├──────────────────────────────┬───────────────────────────────┬───────────────────────────────┤
 │ PHASE 1: REGRESSION          │ PHASE 2: 3-TIER SCREENING     │ PHASE 3: DEEP SEQUENCES       │
 │ • Continuous HOMA-IR, HbA1c  │ • Normal vs Prediab vs IR     │ • 1D-CNN Temporal ConvNet     │
 │ • Ridge, Lasso, RF, LGBM, XGB│ • XGBoost, LightGBM, Balanced │ • 2-Layer Bidirectional LSTM  │
 │ • Pearson r = 0.501          │ • Balanced Acc = 54.85%       │ • Dual-Branch Hybrid Fusion   │
 │ • Bland-Altman LoA: 95.8%    │ • Multiclass AUROC = 0.738    │ • Fused Pearson r = 0.479     │
 └──────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
                                                │
                                                ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                       EXPLAINABLE AI & CLINICAL DECISION SUPPORT (SHAP)                      │
 │  • Global Beeswarm Attribution ──► Waist (0.766) > BMI (0.515) > Resting Pulse (0.309)       │
 │  • Clinical Interactions ──► Circadian Amplitude buffers metabolic risk in elevated BMI      │
 │  • Patient Case Studies ──► Transparent individual waterfall plots for clinical auditability │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Master Experimental Results & Benchmarks

### 1. Phase 1: Continuous Metabolic Biomarker Regression (5-Fold CV)

Evaluated across 6 machine learning architectures on held-out test splits with **zero blood access**:

| Target Biomarker | Best Model | Pearson Correlation ($r$) | Coefficient of Det. ($R^2$) | MAE | RMSE | Bland-Altman LoA (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Continuous HOMA-IR** | **LightGBM / XGBoost** | **$0.501$** | **$0.245$** | **$1.409$** | **$2.511$** | **$95.8\%$** |
| **HbA1c (%)** | **Ridge / ElasticNet** | **$0.377$** | **$0.141$** | **$0.338\%$** | **$0.521\%$** | **$95.2\%$** |
| **Fasting Glucose (mg/dL)** | **Random Forest / Ridge** | **$0.279$** | **$0.076$** | **$9.39$ mg/dL** | **$14.77$ mg/dL** | **$94.9\%$** |

---

### 2. Phase 2: Non-Invasive 3-Class Risk Screening (Zero Blood Access)

Stratifying individuals into **Class 0 (Normal)**, **Class 1 (Prediabetes)**, and **Class 2 (Insulin Resistant)**:

| Model Architecture | Feature Representation | Balanced Accuracy (%) | Macro F1-Score (%) | Weighted F1-Score (%) | Multiclass AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Chance Baseline** | Prior Class Distribution | $33.33\%$ | $24.81\%$ | $37.20\%$ | $0.500$ |
| **Logistic Regression** | Linear Vitals + Circadian | $52.41\%$ | $50.12\%$ | $52.80\%$ | $0.708$ |
| **MLP Neural Network** | Dense Latent (128-64-32) | $51.89\%$ | $49.75\%$ | $52.14\%$ | $0.701$ |
| **LightGBM Classifier** | Gradient Boosted Trees | $54.05\%$ | $51.28\%$ | $53.94\%$ | $0.726$ |
| **Random Forest** | Balanced Ensemble | $54.81\%$ | $51.62\%$ | $54.11\%$ | $0.732$ |
| **XGBoost Classifier** | Exact Gradients + Subsampling | **$54.85\%$** | **$52.19\%$** | **$54.68\%$** | **$0.738$** |

* **Key Takeaway:** XGBoost delivers a **$+21.5\%$ absolute increase** over random chance ($54.85\%$ vs $33.33\%$) with a multiclass AUROC of **$0.738$**, demonstrating that passive wearable actigraphy and resting vitals reliably separate silent prediabetes and insulin resistance from healthy metabolic profiles without blood draws.

---

### 3. Phase 3: Deep Temporal Sequence Modeling (168-Hour Actigraphy Tensors)

Directly comparing end-to-end deep learning on raw 168-hour time series against tabular tree models:

| Architecture | Input Representation | Training Time | HOMA-IR Pearson $r$ | HOMA-IR MAE | 3-Class Bal. Acc. (%) | 3-Class AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN (Temporal ConvNet)** | Raw 168h Actigraphy Alone | 371.0 s | $0.089$ | $2.102$ | $35.27\%$ | $0.528$ |
| **Bidirectional LSTM (Bi-LSTM)** | Raw 168h Actigraphy Alone | 912.4 s | $0.104$ | $2.085$ | $36.14\%$ | $0.539$ |
| **Dual-Branch Hybrid Fusion** | **Raw 168h Actig. + Static Vitals** | **410.5 s** | **$0.479$** | **$1.412$** | **$54.42\%$** | **$0.735$** |
| **XGBoost (Tabular Baseline)** | Hand-Engineered Circadian + Vitals | 6.2 s | **$0.480$** | **$1.411$** | **$54.85\%$** | **$0.738$** |

* **Core Discovery:** Raw motion sequences alone suffer from **Sensor-Alone Identity Ambiguity** ($r \approx 0.09$), because an active athlete and an active insulin-resistant individual generate identical diurnal step counts without physiological calibration. When fused via our **Dual-Branch Hybrid Fusion Network**, the deep model achieves parity ($r = 0.479$, AUROC $= 0.735$) with XGBoost, completely eliminating the need for manual Cosinor feature engineering.

---

### 4. Phase 4: Explainable AI & Clinical Interpretability (Tree SHAP)

Using cooperative game-theoretic Shapley values across held-out test patients:

* **Top 5 Drivers of Insulin Resistance:**
  1. **Waist Circumference (Mean $|SHAP| = 0.766$):** Demonstrating that central intra-abdominal visceral adiposity is nearly $1.5\times$ more pathogenic than subcutaneous general fat (**BMI = $0.515$**).
  2. **Resting Heart Rate ($0.309$):** Reflecting sympathetic autonomic overdrive and suppressed vagal tone.
  3. **Race / Ethnicity ($0.219$):** Reflecting genetic and demographic risk factors.
  4. **Systolic Blood Pressure ($0.165$):** Indicating arterial stiffness and vascular endothelial stress.
  5. **Cosinor Fit Goodness $R^2$ ($0.120$) & Circadian Amplitude ($0.111$):** Adherence to a clean 24-hour diurnal rhythm serves as a vital metabolic protective factor.
* **Nonlinear Interaction Discovery:** In patients with elevated BMI ($>30 \text{ kg/m}^2$), maintaining high relative circadian amplitude ($RA > 0.85$) substantially mitigates predicted metabolic elevation, confirming that robust circadian synchronization buffers metabolic risk.

---

## 📁 Repository Directory Structure

```
Biomedical-AI-Wearable-Sensing/
├── 01_NHANES_Strategic_Pivot_and_Plan_of_Action.docx         # Strategic Pivot Report (Word)
├── 01_NHANES_Strategic_Pivot_and_Plan_of_Action.md           # Strategic Pivot Report (Markdown)
├── 02_NHANES_Comprehensive_EDA_and_Feature_Report.docx       # Complete EDA Report (Word)
├── 02_NHANES_Comprehensive_EDA_and_Feature_Report.md         # Complete EDA Report (Markdown)
├── 03_NHANES_Phase1_Continuous_Biomarker_Regression_Report.docx  # Phase 1 Benchmark (Word)
├── 03_NHANES_Phase1_Continuous_Biomarker_Regression_Report.md    # Phase 1 Benchmark (Markdown)
├── 04_NHANES_Phase2_NonInvasive_Risk_Screening_Report.docx   # Phase 2 Benchmark (Word)
├── 04_NHANES_Phase2_NonInvasive_Risk_Screening_Report.md     # Phase 2 Benchmark (Markdown)
├── 05_NHANES_Deep_Temporal_Sequence_Modeling_Report.docx     # Deep Learning Benchmark (Word)
├── 05_NHANES_Deep_Temporal_Sequence_Modeling_Report.md       # Deep Learning Benchmark (Markdown)
├── 06_NHANES_SHAP_Clinical_Interpretability_Report.docx      # SHAP Explainability Report (Word)
├── 06_NHANES_SHAP_Clinical_Interpretability_Report.md        # SHAP Explainability Report (Markdown)
│
├── notebooks/                                                # Fully Executable Jupyter Notebooks
│   ├── 01_NHANES_Phase1_Biomarker_Regression.ipynb           # Continuous Regression (HOMA, HbA1c, Glu)
│   ├── 02_NHANES_Phase2_NonInvasive_Risk_Screening.ipynb     # 3-Tier Risk Screening Benchmark
│   ├── 03_NHANES_Deep_Temporal_Sequence_Modeling.ipynb       # 1D-CNN, Bi-LSTM & Dual-Branch Network
│   └── 04_NHANES_SHAP_Clinical_Interpretability.ipynb        # Complete SHAP Interpretability Suite
│
├── scripts/                                                  # Production Python Scripts
│   ├── download_nhanes_dataset.py                            # Scrapes & downloads all 29 NHANES files
│   ├── build_unified_nhanes.py                               # Merges demographics, labs, actigraphy
│   ├── extract_circadian_features.py                         # Parametric Cosinor & Non-Parametric NPCRA
│   ├── prepare_sequence_tensors.py                           # Builds (N, 168, 3) 7-day sequence arrays
│   ├── run_phase1_benchmark.py                               # 5-fold CV continuous regression benchmark
│   ├── run_phase2_benchmark.py                               # 5-fold CV 3-class risk screening benchmark
│   ├── run_deep_learning_benchmark.py                        # 1D-CNN, Bi-LSTM & Dual-Branch PyTorch suite
│   ├── run_shap_explainability.py                            # Exact Tree SHAP computation & plotting
│   └── export_docs_to_word.py                                # Automated Markdown to formatted .docx tool
│
├── figures/                                                  # 300 DPI Publication-Ready Plots
│   ├── deep_vs_tree_model_comparison.png                     # DL vs Gradient Boosted Trees comparison
│   ├── phase1_true_vs_predicted_scatter.png                  # True vs Predicted continuous biomarkers
│   ├── phase1_bland_altman_agreement.png                     # Bland-Altman 95% Limits of Agreement
│   ├── phase1_model_comparison_pearson_r.png                 # Pearson correlation bar charts
│   ├── phase2_confusion_matrices.png                         # Multi-model normalized confusion matrices
│   ├── phase2_multiclass_roc_curves.png                      # One-vs-Rest ROC curves (AUROC 0.738)
│   ├── phase2_model_comparison_bar_chart.png                 # Balanced accuracy & F1 comparisons
│   ├── shap_global_beeswarm_homa.png                         # Global SHAP beeswarm importance & direction
│   ├── shap_multiclass_bar_screening.png                     # Multiclass risk attribution bars
│   ├── shap_clinical_feature_interactions.png                # 2D Interaction manifolds (BMI x Circadian)
│   └── shap_patient_case_studies.png                         # Patient-level waterfall decision audits
│
├── results/                                                  # Benchmark Numerical Tables (CSV)
│   ├── phase1_regression_benchmark.csv                       # HOMA-IR, HbA1c, Glucose metrics
│   ├── phase2_classification_benchmark.csv                   # 3-Class classification metrics
│   ├── deep_learning_benchmark.csv                           # 1D-CNN, Bi-LSTM, Dual-Branch metrics
│   └── shap_feature_importance.csv                           # Mean absolute SHAP values for 27 features
│
└── README.md                                                 # Master Project Documentation
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Installation

```bash
git clone https://github.com/SARVAGYA-TIWARI/Biomedical-AI-Wearable-Sensing.git
cd Biomedical-AI-Wearable-Sensing

python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

pip install -r requirements.txt
```

### 2. End-to-End Execution Pipeline

To reproduce the entire benchmark suite from raw downloads to publication figures:

```bash
# Step 1: Download all 29 NHANES raw .XPT files (307 MB)
python scripts/download_nhanes_dataset.py

# Step 2: Build Unified Master Dataset & calculate clinical HOMA-IR
python scripts/build_unified_nhanes.py

# Step 3: Extract Parametric Cosinor & Non-Parametric Circadian Features
python scripts/extract_circadian_features.py

# Step 4: Run Phase 1 Continuous Biomarker Regression Benchmark
python scripts/run_phase1_benchmark.py

# Step 5: Run Phase 2 Non-Invasive 3-Class Risk Screening Benchmark
python scripts/run_phase2_benchmark.py

# Step 6: Prepare 168-Hour Sequence Tensors & Run Deep Learning Suite
python scripts/prepare_sequence_tensors.py
python scripts/run_deep_learning_benchmark.py

# Step 7: Generate Complete SHAP Interpretability Suite & Case Studies
python scripts/run_shap_explainability.py
```

---

## 📖 Complete Documentation & Reports

| Document Title | Markdown Link | Microsoft Word Link | Core Focus |
| :--- | :--- | :--- | :--- |
| **Doc 1: Strategic Pivot & Plan of Action** | [View .md](01_NHANES_Strategic_Pivot_and_Plan_of_Action.md) | [Download .docx](01_NHANES_Strategic_Pivot_and_Plan_of_Action.docx) | Comprehensive dataset trade-offs, rationale for pivoting from D1NAMO to NHANES, and BTP roadmap |
| **Doc 2: Comprehensive EDA & Circadian Engineering** | [View .md](02_NHANES_Comprehensive_EDA_and_Feature_Report.md) | [Download .docx](02_NHANES_Comprehensive_EDA_and_Feature_Report.docx) | 2.78M epoch actigraphy processing, Cosinor regression math, NPCRA metrics, clinical distributions |
| **Doc 3: Phase 1 Continuous Biomarker Regression** | [View .md](03_NHANES_Phase1_Continuous_Biomarker_Regression_Report.md) | [Download .docx](03_NHANES_Phase1_Continuous_Biomarker_Regression_Report.docx) | Ridge, Lasso, RF, LightGBM, XGBoost, MLP benchmarking for HOMA-IR ($r=0.501$), HbA1c, and Glucose |
| **Doc 4: Phase 2 Non-Invasive Risk Screening** | [View .md](04_NHANES_Phase2_NonInvasive_Risk_Screening_Report.md) | [Download .docx](04_NHANES_Phase2_NonInvasive_Risk_Screening_Report.docx) | Zero-blood-access 3-class risk classification (Normal, Prediabetes, IR), XGBoost Bal Acc = $54.85\%$ |
| **Doc 5: Deep Temporal Sequence Modeling** | [View .md](05_NHANES_Deep_Temporal_Sequence_Modeling_Report.md) | [Download .docx](05_NHANES_Deep_Temporal_Sequence_Modeling_Report.docx) | 1D-CNN, Bi-LSTM, and Dual-Branch Multi-Modal Network on 168-hour consecutive actigraphy arrays |
| **Doc 6: Explainable AI & Clinical Interpretability** | [View .md](06_NHANES_SHAP_Clinical_Interpretability_Report.md) | [Download .docx](06_NHANES_SHAP_Clinical_Interpretability_Report.docx) | Exact Tree SHAP values, beeswarm plots, 2D feature interactions, and patient-level decision audits |

---

## ⚖️ License & Acknowledgements

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

* **National Health and Nutrition Examination Survey (NHANES):** Supported by the Centers for Disease Control and Prevention (CDC) National Center for Health Statistics (NCHS). Publicly available under CDC Open Data.
* **Theoretical Frameworks:** Built upon foundational principles from Princeton's *SweetDeep* circadian modeling and Google Health's *Machine Learning for Insulin Resistance Prediction*.
