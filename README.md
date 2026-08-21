# Wearable AI for Diabetes & Glucose Dynamics: Multimodal Biosignal Benchmarking on D1NAMO

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Validation: LOSO CV](https://img.shields.io/badge/Validation-Leave--One--Subject--Out-green.svg)](#leave-one-subject-out-loso-cross-validation-framework)

**Author:** [SARVAGYA-TIWARI](https://github.com/SARVAGYA-TIWARI)  
**Domain:** Biomedical Artificial Intelligence, Wearable Sensing, Time-Series Machine Learning, Digital Health  

---

## 📌 Executive Overview

This repository hosts a production-ready machine learning framework and clinical benchmarking suite for continuous metabolic monitoring using multi-sensor wearable devices. Built on top of the real-world **D1NAMO Dataset** (9 Type-1 diabetic subjects, 8,374 synchronized 5-minute epochs, 47 raw 250 Hz ECG sessions, and continuous glucose monitoring), this project addresses two core clinical challenges:

1. **Phase 1: Benchmark Glucose Forecasting (30 & 60 Minutes Ahead):** Predicts future continuous interstitial glucose levels using historical glucose dynamics combined with multi-sensor telemetry (Heart Rate, raw ECG-derived HRV [SDNN, RMSSD], Accelerometer Activity, Device Temperature, Breathing Rate, and Sinusoidal Circadian Harmonics) under strict **Leave-One-Subject-Out (LOSO)** cross-validation across 10 ML/DL architectures.
2. **Phase 2: Non-Invasive Glucose Dynamics (Zero Glucose History):** Assesses whether wearable signals alone—**without any past or invasive glucose measurements**—can reliably classify 3-class glycemic ranges (Hypoglycemic $<70$, Target $70\text{--}180$, Hyperglycemic $>180$ mg/dL) and 3-class trend directions over 30 and 60 minutes.

---

## 📐 System Architecture

```
                                  D1NAMO MULTIMODAL RAW BIOSIGNALS
 ┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
 │ Continuous Glucose Monitoring│ │  Single-Lead ECG Waveforms   │ │    Zephyr BioHarness 3.0     │
 │  • Dexcom CGM (5-min grid)   │ │  • 250 Hz Sampling (11 GB)   │ │  • Heart Rate (HR, 1 Hz)     │
 │  • Interstitial Glucose      │ │  • 47 Recording Sessions     │ │  • Activity, Temp, BR        │
 └──────────────┬───────────────┘ └──────────────┬───────────────┘ └──────────────┬───────────────┘
                │                                │                                │
                └────────────────────────┐       │       ┌────────────────────────┘
                                         ▼       ▼       ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                             FEATURE EXTRACTION & SYNCHRONIZATION                             │
 │  • Pan-Tompkins QRS Algorithm ──► R-Peak Detection ──► Millisecond HRV (SDNN, RMSSD, pNN50)   │
 │  • SweetDeep Circadian Encodings ──► Sinusoidal Time Harmonics [sin(2πt/24), cos(2πt/24)]    │
 │  • Multimodal Resampling ──► Unified 5-Minute Grid Alignment (8,374 Rows across 9 Subjects)   │
 └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                     LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION BENCHMARK                        │
 │                           (Train on N-1 Subjects, Test on Held-Out Subject)                 │
 ├──────────────────────────────────────────────┬───────────────────────────────────────────────┤
 │ PHASE 1: GLUCOSE FORECASTING                 │ PHASE 2: NON-INVASIVE GLYCEMIC AI             │
 │ • 30m & 60m Horizons (10 Models)            │ • Zero Glucose History (6 Classifiers)        │
 │ • Linear, Ridge, RF, XGB, LGBM, LSTM, TCN... │ • Range (Hypo/Target/Hyper) & Trend (Dec/Stb/Inc)│
 │ • Clarke Error Grid Analysis (Zones A+B)     │ • Balanced Accuracy, Macro F1, Confusion Matrix│
 └──────────────────────────────────────────────┴───────────────────────────────────────────────┘
```

---

## 📊 Benchmark Summary & Key Performance Results

### Phase 1: Benchmark Glucose Forecasting (LOSO Cross-Validation)

Evaluated across $N=9$ Leave-One-Subject-Out folds on **7,931 test points (30-min horizon)** and **7,865 test points (60-min horizon)** across 10 model architectures:

| Horizon | Feature Set | Model Architecture | MAE (mg/dL) | RMSE (mg/dL) | Zone A (%) | Zone B (%) | Clinical Accuracy (A+B) | Zone E Error (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 min** | **Multimodal** | **Linear Regression** | **15.25** | **21.13** | **86.82%** | **11.69%** | **98.51%** | **0.00%** |
| 30 min | Multimodal | Ridge Regression | 15.32 | 21.44 | 86.65% | 11.78% | 98.42% | 0.00% |
| 30 min | Multimodal | LightGBM | 18.02 | 24.94 | 81.72% | 14.42% | 96.14% | 0.00% |
| 30 min | Multimodal | XGBoost | 18.22 | 25.25 | 81.31% | 14.61% | 95.93% | 0.00% |
| 30 min | Multimodal | Random Forest | 18.97 | 26.42 | 80.63% | 15.53% | 96.17% | 0.01% |
| 30 min | Multimodal | Naive (Persistence) | 21.89 | 31.46 | 76.59% | 20.60% | 97.19% | 0.10% |
| **30 min** | **Glucose-Only** | **Linear Regression** | **15.10** | **21.11** | **86.96%** | **11.55%** | **98.51%** | **0.00%** |
| **60 min** | **Multimodal** | **Linear Regression** | **29.64** | **40.58** | **64.17%** | **29.68%** | **93.85%** | **0.29%** |
| 60 min | Multimodal | Ridge Regression | 29.75 | 40.81 | 63.94% | 29.89% | 93.83% | 0.31% |
| 60 min | Multimodal | LightGBM | 32.69 | 44.40 | 59.55% | 31.18% | 90.73% | 0.11% |
| **60 min** | **Glucose-Only** | **Linear Regression** | **29.20** | **40.48** | **65.42%** | **28.86%** | **94.28%** | **0.42%** |

* **Key Takeaway:** Regularized linear models leverage short-term glucose momentum directly without overfitting to cross-patient baseline shifts under LOSO validation, achieving **98.51% clinical accuracy (Clarke Error Grid Zones A+B)** and **0.00% dangerous Zone E errors**.

---

### Phase 2: Non-Invasive Glucose Prediction (Zero Glucose History)

Evaluated across 6 classifiers using **only** non-invasive wearable telemetry and circadian harmonics:

| Clinical Task | Model Architecture | Balanced Accuracy (%) | Macro F1-Score (%) | Weighted F1-Score (%) | Test Points ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Glycemic Range (3-Class)** | **Logistic Regression** | **42.72%** | **34.87%** | **40.47%** | **8,063** |
| Glycemic Range (3-Class) | LightGBM | 41.21% | 38.36% | 47.42% | 8,063 |
| Glycemic Range (3-Class) | Random Forest | 41.14% | 38.09% | 46.73% | 8,063 |
| Glycemic Range (3-Class) | Neural Network (MLP) | 37.90% | 34.05% | 42.08% | 8,063 |
| Glycemic Range (3-Class) | *Random Baseline* | *33.33%* | *23.91%* | *40.10%* | *8,063* |
| **Trend 30-min (3-Class)** | **Logistic Regression** | **39.61%** | **32.35%** | **50.93%** | **7,997** |
| Trend 30-min (3-Class) | Random Forest | 39.14% | 33.54% | 54.32% | 7,997 |

* **Key Takeaway:** Without any blood access or glucose history, wearable biosignals deliver a **+9.4% absolute gain** in balanced accuracy over random chance for 3-class range classification (Low/Target/High), highlighting high promise for passive smartwatch risk screening.

---

## 🛠️ Repository Directory Structure

```
.
├── notebooks/
│   ├── 01_Phase1_Glucose_Forecasting_LOSO.ipynb    # Jupyter Notebook for Phase 1 Forecasting
│   └── 02_Phase2_NonInvasive_Prediction_LOSO.ipynb # Jupyter Notebook for Phase 2 Non-Invasive AI
├── scripts/
│   ├── run_phase1.py                               # CLI Runner for Phase 1 LOSO Benchmark
│   ├── run_phase2.py                               # CLI Runner for Phase 2 LOSO Benchmark
│   └── generate_report.py                         # Docx Research Report Compiler
├── src/
│   ├── clarke_error_grid.py                        # Clarke Error Grid Evaluation & Plotting Engine
│   ├── phase1_benchmark_loso.py                    # Phase 1 LOSO Training & Evaluation Pipeline
│   ├── phase2_noninvasive_loso.py                  # Phase 2 LOSO Classification Pipeline
│   └── generate_comprehensive_report.py            # Automated Word Report Builder
├── figures/                                         # Generated Publication Visualizations
│   ├── phase1_clarke_grid_best_models.png
│   ├── phase1_mae_rmse_comparison.png
│   ├── phase1_multimodal_vs_glucose_ablation.png
│   ├── phase2_balanced_accuracy_comparison.png
│   ├── phase2_confusion_matrices.png
│   └── phase2_feature_importances.png
├── results/                                         # Exported Benchmark Metric CSVs
│   ├── phase1_loso_results.csv
│   └── phase2_loso_results.csv
├── requirements.txt                                 # Pinned Dependencies
├── .gitignore                                       # Clean Version Control Rules
└── README.md                                        # Master Project Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/SARVAGYA-TIWARI/Wearable-AI-Glucose-Dynamics.git
cd Wearable-AI-Glucose-Dynamics

# Create Python Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the D1NAMO dataset from [Kaggle D1NAMO Dataset](https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data/data) and place the processed master parquet/csv file under `data/d1namo_multimodal_master.parquet`.

### 3. Run Phase 1 & Phase 2 Benchmarks via CLI
```bash
# Execute Phase 1 LOSO Benchmark (30m & 60m Horizons across 10 Models)
python scripts/run_phase1.py

# Execute Phase 2 Non-Invasive LOSO Benchmark (Zero Glucose History)
python scripts/run_phase2.py

# Generate Full Word (.docx) Research Report with Embedded Figures
python scripts/generate_report.py
```

### 4. Run Interactive Jupyter Notebooks
```bash
jupyter notebook notebooks/01_Phase1_Glucose_Forecasting_LOSO.ipynb
jupyter notebook notebooks/02_Phase2_NonInvasive_Prediction_LOSO.ipynb
```

---

## 🔬 Key Engineering Contributions

1. **Pan-Tompkins Raw ECG Peak Extraction:** Reconstructed true heart rate variability metrics (**SDNN, RMSSD, pNN50**) from 11 GB of 250 Hz single-lead raw ECG signals, overriding hardware sentinel corruption (`65535`).
2. **Leave-One-Subject-Out (LOSO) Rigor:** Built an $N=9$ fold cross-validation framework ensuring zero intra-patient data leakage across training and test splits.
3. **Clarke Error Grid Safety Analysis:** Implemented full clinical safety boundary evaluation, confirming **0.00% Zone E (lethal treatment error) risks**.
4. **SweetDeep Circadian Modeling:** Implemented sinusoidal time-of-day continuous harmonic transformations to eliminate midnight numerical jump discontinuities.

---

## 🔮 Future Directions

* **Few-Shot Domain Adaptation:** Fine-tuning base LOSO models with 10–12 hours of patient-specific calibration data.
* **Contextual Meal & Insulin Decay Kernels:** Integrating self-reported carbohydrate and bolus insulin exponential decay curves ($e^{-\Delta t/\tau}$).
* **PPG Wrist Sensor Translation:** Adapting Pan-Tompkins peak detection for optical wrist photoplethysmography (PPG) smartwatches.
* **Edge AI Quantization:** Converting PyTorch architectures to ONNX Nano / TensorFlow Lite for Microcontrollers (TFLite) for real-time smartwatch execution.

---

## 📜 License & Citation

Distributed under the **MIT License**. See `LICENSE` for more information.

If you find this work or codebase helpful in your research, please consider starring ⭐ the repository and citing:

```bibtex
@misc{tiwari2026wearableai,
  author = {Sarvagya Tiwari},
  title = {Wearable AI for Diabetes & Glucose Dynamics: Multimodal Biosignal Benchmarking on D1NAMO},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/SARVAGYA-TIWARI/Wearable-AI-Glucose-Dynamics}}
}
```
