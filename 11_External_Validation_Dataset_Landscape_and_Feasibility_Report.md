# Strategic Research Report: External Validation Dataset Landscape
## Evaluating Candidate Biomedical & Wearable Datasets to Validate Our NHANES Models

**Project Title:** Wearable AI for Early Metabolic Risk Screening and Circadian Biomarker Estimation  
**Author:** SARVAGYA-TIWARI  
**Current Baseline Cohort:** CDC NHANES 2011–2014 Multi-Modal Cohort ($N = 3,296$ Non-Diabetic Adults)  
**Date:** October 2026  

---

# 1. Executive Summary: The Need for External Validation

In biomedical machine learning, evaluating a model on the same cohort it was trained on (even using strict 5-fold cross-validation) is termed **internal validation**. To prove to faculty, examiners, and peer-reviewed journals that an algorithm possesses true clinical generalizability, researchers perform **external validation**: testing the frozen models on a completely separate cohort collected by different clinical teams, in different geographical locations, or using different sensor hardware.

This research report evaluates the top candidate datasets available globally, ranking them by:
1. **Target Biomarker Alignment:** Availability of paired fasting insulin/glucose (HOMA-IR) or Continuous Glucose Monitoring (CGM).
2. **Sensor Modality Alignment:** Presence of continuous wrist-worn accelerometry/actigraphy or wearable photoplethysmography (PPG).
3. **Open-Access Accessibility:** Immediate public download vs. multi-month institutional data-use agreements.
4. **Sample Size & Clinical Diversity:** Sufficient statistical power ($N \ge 100$).

---

# 2. Master Comparison Matrix of Candidate Datasets

| Dataset Name | Primary Institution & Repository | Sample Size ($N$) | Wearable Hardware | Ground Truth Metabolic Outcome | Access Requirements | Feasibility Rank |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **1. MESA Sleep Study** | National Heart, Lung, and Blood Institute (NHLBI) via **NSRR** | **$N = 2,237$** multi-ethnic adults | **Actiwatch Spectrum** (7-day wrist actigraphy) | **Fasting Insulin $\rightarrow$ HOMA-IR**, Fasting Glucose, HbA1c, BMI, Waist | **Free Instant Access** via NSRR account (`sleepdata.org`) | **#1 (Best for HOMA-IR External Validation)** |
| **2. Duke BIG IDEAs Lab** | Duke University via **PhysioNet** | $N = 16$ adults ($26,000$ paired epochs) | **Empatica E4** (Wrist Accelerometer, PPG/HR, EDA, Temp) | **Dexcom G6 CGM** (5-minute continuous glucose) | **100% Free Public Instant Download** (PhysioNet DOI: `10.13026/aw6y-fc44`) | **#2 (Best for Continuous CGM & Deep Learning)** |
| **3. CGMacros Dataset** | Texas A&M University (PSI-TAMU) via **PhysioNet** | $N = 45$ adults (10 days free-living) | **Fitbit Activity Tracker** + Dual CGMs | Continuous Glucose + Meal Macronutrients + Blood Panels | Credentialed PhysioNet / GitHub repository | **#3 (Great for Diet & Multiclass Validation)** |
| **4. CDC NHANES 2003–2006** | National Center for Health Statistics (CDC NCHS) | **$N \approx 4,000$** adults | **ActiGraph 7164** (7-day hip accelerometry) | **Fasting Insulin $\rightarrow$ HOMA-IR**, Fasting Glucose, HbA1c, Lipids | **100% Free Public Instant Download** from CDC | **#4 (Best for Multi-Wave Historical Validation)** |
| **5. UK Biobank (UKB)** | UK Biobank Consortium | $N \approx 100,000$ | **Axivity AX3** (7-day wrist accelerometer) | HbA1c, Random Glucose, NMR Metabolomics | Requires paid application & 2–3 month MTA review | **Low (Infeasible for Quick BTP Timelines)** |

---

# 3. Deep-Dive Analysis of the Top Options

```
                           RECOMMENDED VALIDATION PATHWAYS
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      WHICH VALIDATION DIRECTION MATCHES YOUR GOAL?                     │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ PATHWAY A: DIRECT HOMA-IR VALIDATION │            │ PATHWAY B: CONTINUOUS CGM FORECAST   │
 │ • Dataset: MESA Sleep (NSRR)         │            │ • Dataset: Duke BIG IDEAs (PhysioNet)│
 │ • Exact match to NHANES HOMA-IR      │            │ • Empatica E4 wristband + Dexcom G6  │
 │ • 7-day wrist actigraphy (N = 2,237) │            │ • 26,000 paired 5-min glucose epochs │
 │ • Proves models work on new hospital │            │ • Proves models track minute-level   │
 │   cohort without changing target!    │            │   glycemic variability in real time  │
 └──────────────────────────────────────┘            └──────────────────────────────────────┘
```

---

## Candidate #1: MESA Sleep Ancillary Study (National Sleep Research Resource - NSRR)
⭐ **Overall Verdict: THE GOLD-STANDARD OPTION FOR DIRECT EXTERNAL VALIDATION**

* **Source Repository:** Hosted on the **National Sleep Research Resource (NSRR)** at `sleepdata.org/datasets/mesa`.
* **Cohort Profile:** $N = 2,237$ men and women aged 54–93 from six racially and ethnically diverse communities across the United States (White, African-American, Hispanic, and Chinese-American).
* **Sensor Modality:** Participants wore an **Actiwatch Spectrum wrist-worn accelerometer** on their non-dominant wrist continuously for **7 consecutive days and nights** (identical to the CDC NHANES GT3X+ wear protocol).
* **Paired Laboratory Gold Standards:**
  * Certified laboratory **Fasting Serum Insulin** and **Fasting Plasma Glucose** $\rightarrow$ Allows direct calculation of **HOMA-IR** ($(\text{Glucose} \times \text{Insulin})/405$).
  * Glycated Hemoglobin (**HbA1c**).
  * Automated resting blood pressure (SBP, DBP, Pulse).
  * Body measures: Certified **Waist Circumference** and **BMI**.
* **Why This is the Best Option:**
  1. **1:1 Feature & Target Alignment:** The input features (7-day wrist actigraphy, resting heart rate, blood pressure, waist circumference, age, sex) and the ground-truth target (HOMA-IR) are **identical** to our NHANES pipeline!
  2. **Zero Code Restructuring:** You can literally run our existing feature extraction script and evaluate our pre-trained LightGBM, XGBoost, and Stacking models directly on MESA without modifying target definitions.
  3. **High Scientific Impact:** Presenting external cross-cohort generalization from NHANES ($N = 3,296$) $\rightarrow$ MESA ($N = 2,237$) provides convincing evidence for a B.Tech thesis or publication.
* **Access Process:** Simple, free account creation on `sleepdata.org` with instant terms-of-use agreement.

---

## Candidate #2: BIG IDEAs Lab Glycemic Variability and Wearables (PhysioNet)
⭐ **Overall Verdict: THE BEST OPTION FOR CONTINUOUS GLUCOSE MONITORING (CGM) & DEEP LEARNING**

* **Source Repository:** Hosted on **PhysioNet** (DOI: `10.13026/aw6y-fc44`, Version 1.1.3, updated April 2026).
* **Lead Investigators:** Jessilyn Dunn, Peter Cho, Brinnae Bent (Duke University Department of Biomedical Engineering).
* **Cohort Profile:** Free-living adults across normoglycemic and pre-diabetic spectrums monitored over 8–10 continuous days.
* **Sensor Modality:** **Empatica E4 Research Wristband** collecting:
  * Triaxial Accelerometer (32 Hz)
  * Photoplethysmography / Blood Volume Pulse (BVP / 64 Hz $\rightarrow$ Heart Rate & HRV)
  * Electrodermal Activity (EDA / Galvanic Skin Response at 4 Hz)
  * Skin Temperature (4 Hz)
* **Paired Ground Truth:** Continuous Glucose Monitor (**Dexcom G6 CGM**) recording subcutaneous interstitial glucose every **5 minutes** (over **26,000 paired epochs**).
* **Why This is High Value:**
  1. **Tests Real-Time Dynamic Forecasting:** While NHANES and MESA evaluate episodic fasting blood draws, the Duke dataset tests whether wrist sensors can track **continuous real-time glycemic swings and post-prandial spikes**.
  2. **Validates Our Deep Temporal Sequence Models:** Our 1D-CNN and Bidirectional LSTM models developed in Report 05 can be applied directly to the continuous 5-minute Dexcom CGM sequences.
  3. **100% Open Access:** Completely open and instantly downloadable via curl, python, or direct browser download from PhysioNet.

---

## Candidate #3: CGMacros Multimodal Dataset (PhysioNet, 2023/2024)
⭐ **Overall Verdict: EXCELLENT FOR DIETARY INTERACTION & MULTICLASS PREDIABETES STRATIFICATION**

* **Source Repository:** Hosted on **PhysioNet** by Texas A&M University (PSI-TAMU).
* **Cohort Profile:** $N = 45$ adults (15 healthy normoglycemic, 16 pre-diabetic, and 14 with confirmed Type 2 Diabetes).
* **Sensor Modality:** Wearable physical activity monitors (Fitbit) worn continuously for **10 consecutive days in free-living conditions**.
* **Paired Ground Truth:** Continuous Glucose Monitoring (**Abbott FreeStyle Libre + Dexcom G6**) + timestamped meals with macronutrients (carbs, fats, proteins) + baseline laboratory blood panels.
* **Why This is Useful:**
  * Provides a direct real-world testbed for our **Phase 2 Three-Class ADA Risk Screening model** (Class 0: Normal vs Class 1: Prediabetes vs Class 2: Diabetes).

---

## Candidate #4: CDC NHANES 2003–2006 Historical Survey Wave
⭐ **Overall Verdict: INSTANT BENCHMARK FOR HISTORICAL & SENSOR-PLACEMENT GENERALIZATION**

* **Source Repository:** CDC National Center for Health Statistics (NCHS) public server.
* **Cohort Profile:** $N \approx 4,000$ nationally representative US adults.
* **Sensor Modality:** **ActiGraph 7164** hip-worn accelerometer worn for 7 continuous days.
* **Paired Ground Truth:** Certified laboratory Fasting Glucose, Fasting Insulin $\rightarrow$ **HOMA-IR**, HbA1c, and Body Measures.
* **Why This is Useful:**
  * Instant, automated download using our existing script structure (`scripts/download_nhanes_dataset.py`).
  * Tests how well an algorithm trained on **wrist actigraphy (2011–2014)** transfers to **hip actigraphy (2003–2006)**.

---

# 4. Strategic Recommendation for the BTP Project

### The Two-Track Strategy:
1. **Track 1 (The Academic Gold Standard — Highly Recommended):**  
   Use **MESA Sleep (NSRR)** to perform **cross-cohort external validation of your exact HOMA-IR and Phase 2 screening models**.  
   *Why:* Because both NHANES and MESA use 7-day wrist actigraphy paired with Fasting Insulin $\rightarrow$ HOMA-IR. Showing that the LightGBM/XGBoost model trained on NHANES retains its accuracy on MESA ($N = 2,237$) provides a strong publication-grade contribution.
2. **Track 2 (The Dynamic Wearable Extension — Optional / Future Work):**  
   Use **Duke BIG IDEAs Lab (PhysioNet)** to demonstrate that our 1D-CNN and LSTM architectures can predict **continuous 5-minute CGM interstitial glucose** from wristband sensor streams.
