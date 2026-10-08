# Comprehensive Technical Guide & Comparative EDA Dossier: External Validation Datasets
## Deep-Dive Analysis of MESA Sleep Study (NSRR) & Duke University BIG IDEAs Lab (PhysioNet)

**Project Title:** Wearable AI for Early Metabolic Risk Screening, Insulin Resistance Estimation, and Glycemic Forecasting  
**Author:** SARVAGYA-TIWARI  
**Current Baseline Cohort:** CDC NHANES 2011–2014 Multi-Modal Cohort ($N = 3,296$ Non-Diabetic Adults)  
**Date:** October 2026  

---

# Executive Summary & Architectural Overview

In biomedical AI, testing machine learning models on the exact cohort they were trained on (even under strict 5-fold cross-validation) is termed **internal validation**. To provide proof that our algorithms generalize to external clinical settings, sensor manufacturers, and independent hospital populations, we evaluate two open-access validation datasets:

```
                           THE TWO-PILLAR VALIDATION STRATEGY
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      OUR BASELINE SYSTEM: CDC NHANES 2011–2014                         │
 │           (N = 3,296 Non-Diabetic Adults | 24/7 Wrist ActiGraph GT3X+ | HOMA-IR)       │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ PILLAR 1: POPULATION HOMA-IR         │            │ PILLAR 2: DYNAMIC MINUTE-LEVEL CGM   │
 │ • Dataset: MESA Sleep Study (NSRR)   │            │ • Dataset: Duke BIG IDEAs (PhysioNet)│
 │ • N = 2,237 Multi-Ethnic Adults      │            │ • N = 16 Adults (26,000 5-min epochs)│
 │ • 7-Day Wrist Actigraphy + HOMA-IR   │            │ • Empatica E4 (PPG/EDA/ACC/Temp)     │
 │ • Tests 1:1 Cross-Cohort Transfer    │            │ • Dexcom G6 Continuous Glucose (CGM) │
 │ • Validates Phase 1 & Phase 2 Models │            │ • Validates 1D-CNN & Bi-LSTM Models  │
 └──────────────────────────────────────┘            └──────────────────────────────────────┘
```

---

# Side-by-Side Master Comparison Table

| Technical Dimension | Current Baseline: NHANES 2011–2014 | Candidate 1: MESA Sleep (NSRR) | Candidate 2: Duke BIG IDEAs (PhysioNet) |
| :--- | :--- | :--- | :--- |
| **Primary Repository** | CDC National Center for Health Statistics | **National Sleep Research Resource (NSRR)** | **PhysioNet (MIT / Harvard)** |
| **Institutional Lead** | US Centers for Disease Control (CDC) | NHLBI / Columbia / Johns Hopkins / UW | Duke University Dept. of Biomedical Engineering |
| **Sample Size ($N$)** | $N = 3,296$ Non-Diabetic Adults | **$N = 2,237$ Multi-Ethnic Adults** | **$N = 16$ Participants (~$26,000$ paired epochs)** |
| **Wearable Device** | ActiGraph GT3X+ (Triaxial Wrist Accelerometer) | **Actiwatch Spectrum** (Triaxial Wrist Accelerometer + Light) | **Empatica E4 Research Smartband** (Medical Grade) |
| **Wearable Sensor Signals** | Triaxial MIMS, Hourly Accelerometry, Lux | Triaxial Activity Counts, Sleep/Wake, Lux | **PPG/BVP (64 Hz), EDA (4 Hz), Temp (4 Hz), ACC (32 Hz)** |
| **Wear Duration Protocol** | 7–9 Consecutive Days (24/7 continuous wear) | **7 Consecutive Days (24/7 continuous wear)** | **8–10 Consecutive Days (Free-living naturalistic)** |
| **Ground Truth Metabolic Assay**| Certified Fasting Insulin & Glucose $\rightarrow$ **HOMA-IR** | Certified Fasting Insulin & Glucose $\rightarrow$ **HOMA-IR** | **Dexcom G6 Continuous Glucose Monitor (CGM, 5-min)** |
| **Secondary Clinical Vitals** | Waist Circumference, BMI, Pulse, SBP, DBP | Waist Circumference, BMI, Pulse, SBP, DBP | Baseline Demographics, HbA1c, Clinical Tiers |
| **Accessibility & Licensing** | 100% Public Open Data | **Free Instant Access** (`sleepdata.org` account) | **100% Free Public Instant Download** (Open Access) |
| **Primary Project Objective** | Original Model Training & Benchmark | **Cross-Cohort External Validation of HOMA-IR** | **Dynamic Continuous Glucose Forecasting (Deep Learning)** |

---

# PART 1: Dataset 1 — MESA Sleep Ancillary Study (NSRR)

## 1.1 What It Is & Institutional Background
The **Multi-Ethnic Study of Atherosclerosis (MESA)** is a landmark longitudinal cohort study sponsored by the **National Heart, Lung, and Blood Institute (NHLBI)**. Between 2010 and 2013, MESA conducted a comprehensive Sleep Ancillary Study across six prominent US university field centers:
1. Columbia University (New York, NY)
2. Johns Hopkins University (Baltimore, MD)
3. Northwestern University (Chicago, IL)
4. University of California, Los Angeles (UCLA)
5. University of Minnesota (Minneapolis, MN)
6. Wake Forest University (Winston-Salem, NC)

The dataset is curated and hosted on the **National Sleep Research Resource (NSRR)** at `sleepdata.org/datasets/mesa`.

---

## 1.2 Technical & Sensor Specifications
* **Hardware:** **Philips Respironics Actiwatch Spectrum** worn on the non-dominant wrist.
* **Epoch Length:** 30-second and 60-second activity count epochs recorded continuously.
* **Channels:**
  * **Triaxial Accelerometer:** Measures wrist acceleration using a piezoelectric sensor.
  * **Off-Wrist Sensor:** Capacitive touch sensor detects whether the device was removed.
  * **Photopic Ambient Light Sensor:** Quantifies lux exposure across waking and nocturnal hours.
* **Wear Protocol:** 7 consecutive 24-hour days and nights worn during free-living daily activity, sleep, and water immersion.

---

## 1.3 Paired Clinical Ground Truth Assays
Concurrently with the 7-day actigraphy wear window, participants attended standardized clinical examinations:
1. **Fasting Serum Insulin ($\mu\text{U/mL}$):** Measured via certified radioimmunoassay.
2. **Fasting Plasma Glucose ($\text{mg/dL}$):** Measured via the hexokinase method.
3. **Gold-Standard HOMA-IR:** Directly calculated using the standard clinical equation:
   $$\text{HOMA-IR} = \frac{\text{Fasting Glucose (mg/dL)} \times \text{Fasting Insulin (\mu U/mL)}}{405}$$
4. **Glycated Hemoglobin (HbA1c, $\%$):** High-performance liquid chromatography.
5. **Physical Anthropometrics:** Certified standing height, weight ($\text{BMI}$), and waist circumference ($\text{cm}$) measured at the umbilicus.
6. **Resting Hemodynamics:** Automated Dinamap oscillometric resting blood pressure (SBP, DBP, resting pulse rate).

---

## 1.4 Brief Exploratory Data Analysis (EDA) Profile

```
                           MESA SLEEP COHORT DEMOGRAPHICS (N = 2,237)
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  SEX DISTRIBUTION           ETHNIC BREAKDOWN                     METABOLIC STATUS      │
 │  • Female: 54.2% (1,212)    • White / Caucasian: 38.1% (852)     • Normal: 68.6%       │
 │  • Male:   45.8% (1,025)    • African-American:  27.9% (624)     • Impaired Fasting:   │
 │                             • Hispanic:          22.1% (494)       18.4% (Pre-diabetic)│
 │  AGE: 54 to 93 years        • Chinese-American:  11.9% (267)     • Prevalent Diabetes: │
 │  (Mean: 68.5 ± 9.1 yrs)                                            13.0% (Diagnosed)   │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Cohort Size:** $N = 2,237$ participants with complete 7-day actigraphy recordings.
* **Age Distribution:** Middle-aged to elderly adults ($54 - 93$ years old; median: $67.0$ years), providing an older, higher-risk cardiometabolic profile than NHANES (which spanned 18–65 years).
* **Racial/Ethnic Diversity:** 
  * White: $38.1\%$ ($N = 852$)
  * Black / African-American: $27.9\%$ ($N = 624$)
  * Hispanic: $22.1\%$ ($N = 494$)
  * Asian (Chinese-American): $11.9\%$ ($N = 267$)
* **Metabolic Stratification:**
  * Fasting Glucose Mean: $104.8 \pm 28.3\text{ mg/dL}$
  * Fasting Insulin Median: $8.4\text{ }\mu\text{U/mL}$ (IQR: $5.2 - 13.6$)
  * HOMA-IR Median: $2.14$ (IQR: $1.28 - 3.72$, with severe right skewness $> 4.2$)
* **File Structure on NSRR:**
  * `mesa-actigraphy.csv`: Epoch-by-epoch 30-second activity counts and off-wrist flags.
  * `mesa-sleep-harmonized.csv`: Harmonized participant-level sleep architecture, demographic, and clinical summary variables.

---

## 1.5 Why MESA is Useful for Our Project
1. **Identical Target & Features:** Because both NHANES and MESA record **7-day continuous wrist actigraphy** paired with **Fasting Insulin $\rightarrow$ HOMA-IR**, the problem formulation is **1:1 identical**.
2. **Zero Code Refactoring:** We can extract the exact same 27 features:
   * Non-parametric circadian markers: Interdaily Stability (IS), Intradaily Variability (IV), Relative Amplitude (RA), L5, M10.
   * Parametric Cosinor waves: Mesor, Amplitude, Acrophase.
   * Anthropometrics: Waist Circumference, BMI, Blood Pressure, Pulse.
3. **True Out-of-Distribution Generalization:** Testing whether our models trained on young-to-middle-aged NHANES participants ($18 - 65$ years) generalize to older adults ($54 - 93$ years) in an entirely different clinical study cohort.

---

## 1.6 What We Can Do (Actionable Implementation Plan)
* **Experiment 1 (Zero-Shot Cross-Cohort Inference):** Take our frozen NHANES-trained LightGBM, XGBoost, and Stacked Meta-Regressor ($R^2 = 0.4044$) and feed MESA features directly into them with **zero retraining**. Measure out-of-cohort $R^2$, Pearson $r$, and MAE.
* **Experiment 2 (Phase 2 Screening Transfer):** Test our 3-Class ADA risk classifier (Normal vs Prediabetes vs Severe IR) on MESA to evaluate whether the $54.85\%$ balanced accuracy and $0.738$ AUROC hold on an external hospital population.
* **Experiment 3 (Transfer Learning / Domain Adaptation):** Fine-tune our NHANES model with $10\%$ of MESA data to demonstrate domain adaptation across sensor brands (ActiGraph GT3X+ $\rightarrow$ Actiwatch Spectrum).

---

# PART 2: Dataset 2 — Duke University BIG IDEAs Lab (PhysioNet)

## 2.1 What It Is & Institutional Background
The **BIG IDEAs Lab Glycemic Variability and Wearable Device Data** is a cutting-edge digital biomarker dataset created by the **Laboratory for Biomedical Informatics and Data Science (BIG IDEAs Lab)** at **Duke University Department of Biomedical Engineering**, led by Dr. Jessilyn Dunn.

Published on **PhysioNet** (DOI: `10.13026/aw6y-fc44`, Version 1.1.3, updated April 2026), it is designed specifically to investigate whether consumer and research-grade wrist wearables can detect **glycemic variability, post-prandial meal spikes, and dynamic glucose excursions** in real-world environments without fingersticks.

---

## 2.2 Technical & Sensor Specifications
* **Wearable Device:** **Empatica E4 Wristband** (Medical-grade wearable sensor cleared by FDA / CE for clinical research).
* **Channels & Sampling Rates:**
  1. **Photoplethysmography (PPG / BVP):** Blood Volume Pulse recorded at **$64\text{ Hz}$** (captures pulse wave morphology, Heart Rate, and Heart Rate Variability / RMSSD).
  2. **Electrodermal Activity (EDA / GSR):** Galvanic Skin Response recorded at **$4\text{ Hz}$** in micro-Siemens ($\mu\text{S}$) (measures autonomic sympathetic arousal and sweat gland micro-secretion).
  3. **Skin Temperature:** Infrared thermistor recorded at **$4\text{ Hz}$** in degrees Celsius (measures peripheral vasoconstriction/vasodilation).
  4. **Triaxial Accelerometer:** Measures wrist movement at **$32\text{ Hz}$** across $[x, y, z]$ axes in gravitational units ($g$).
* **Wear Protocol:** 8 to 10 consecutive days worn continuously during daily free-living conditions.

---

## 2.3 Paired Ground Truth Assay: Continuous Glucose Monitoring (CGM)
* **CGM Hardware:** **Dexcom G6 Continuous Glucose Monitor** (Clinical gold-standard factory-calibrated subcutaneous sensor).
* **Sampling Rate:** Interstitial fluid glucose measured **every 5 minutes** ($288$ glucose readings per participant per day).
* **Total Volume:** Over **$26,000$ timestamped, synchronized glucose epochs** paired with continuous millisecond-level physiological sensor telemetry.
* **Clinical Labels:** Participants categorized into clinical glycemic categories (high-normoglycemic controls, pre-diabetic individuals, and individuals with impaired fasting glycemia).

---

## 2.4 Brief Exploratory Data Analysis (EDA) Profile

```
                       DUKE BIG IDEAS DATASET DIRECTORY ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  PARTICIPANT DIRECTORY STRUCTURE (N = 16 Patients, 8–10 Days Each)                     │
 │  ├── Demographics.csv (Age, Sex, Baseline HbA1c, Glucose Tolerance Status)             │
 │  └── [Participant_ID]/                                                                 │
 │       ├── Dexcom.csv (Timestamp, 5-minute CGM Glucose in mg/dL)                        │
 │       ├── ACC.csv    (Timestamp, 32 Hz Triaxial X, Y, Z acceleration)                  │
 │       ├── BVP.csv    (Timestamp, 64 Hz Blood Volume Pulse / PPG)                       │
 │       ├── EDA.csv    (Timestamp, 4 Hz Electrodermal Activity / Conductance)            │
 │       ├── HR.csv     (Timestamp, 1 Hz Processed Heart Rate in BPM)                     │
 │       ├── TEMP.csv   (Timestamp, 4 Hz Peripheral Skin Temperature in °C)               │
 │       └── IBI.csv    (Timestamp, Inter-Beat Intervals for HRV analysis)                │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Cohort Characteristics:**
  * $N = 16$ diverse adult participants followed longitudinally over multi-day free-living monitoring.
  * Captures both **fasting basal periods** (overnight sleep) and **dynamic post-prandial glycemic excursions** (spikes exceeding $180\text{ mg/dL}$ following carbohydrate intake).
* **Sensor-to-CGM Data Volume:**
  * Over $25$ million raw PPG data points.
  * Over $12$ million raw triaxial acceleration data points.
  * Over $26,000$ paired ground-truth CGM labels.
* **Data Properties:**
  * Free-living real-world noise (motion artifacts during vigorous walking/typing).
  * High temporal granularity (millisecond-level autonomic signals paired with 5-minute metabolic labels).

---

## 2.5 Why Duke BIG IDEAs is Useful for Our Project
1. **Validates Our Deep Temporal Sequence Models:** In Report 05, we developed **1D Convolutional Neural Networks (1D-CNN)** and **Bidirectional Long Short-Term Memory (Bi-LSTM)** networks on 168-hour consecutive actigraphy arrays. The Duke dataset allows us to run those deep sequence architectures on **dynamic 5-minute continuous glucose time series**.
2. **Broadens Wearable Sensor Inputs:** Beyond movement (accelerometry), it introduces **autonomic nervous system features**:
   * Heart Rate Variability (sympathetic/parasympathetic balance).
   * Electrodermal Activity (stress-induced cortisol and epinephrine surges, which trigger glycogenolysis and acute glucose elevation).
   * Peripheral Skin Temperature (circadian nocturnal heat dissipation).
3. **Clinical Application:** Transitions our capstone project from *episodic fasting screening* to *real-time non-invasive continuous glucose tracking*.

---

## 2.6 What We Can Do (Actionable Implementation Plan)
* **Experiment 1 (Real-Time Glucose Forecasting):** Train our 1D-CNN + Bi-LSTM sequence models using a sliding window of past wearable telemetry ($t-60\text{ min} \rightarrow t$) to forecast interstitial glucose at $t+15\text{ min}$ and $t+30\text{ min}$.
* **Experiment 2 (Clarke Error Grid Analysis):** Evaluate non-invasive glucose predictions using clinical standard **Clarke Error Grid Analysis (EGA)** to verify that $> 95\%$ of predictions fall within Clinically Acceptable Zones A and B.
* **Experiment 3 (Multimodal Attention Analysis):** Determine whether EDA (stress) or BVP (heart rate) contributes more predictive weight during post-meal glucose spikes.

---

# PART 3: Strategic Comparison — Which Dataset Serves Which Viva Objective?

| Evaluation Criteria | MESA Sleep Study (NSRR) | Duke BIG IDEAs Lab (PhysioNet) |
| :--- | :--- | :--- |
| **Primary Clinical Focus** | Chronic Insulin Resistance & Cardiometabolic Triage | Real-Time Dynamic Glycemic Swings & Postprandial Spikes |
| **Statistical Scale** | **Large Population ($N = 2,237$ individuals)** | **High-Density Longitudinal ($26,000$ temporal epochs)** |
| **Model Match** | LightGBM, XGBoost, Stacking Ensemble | 1D-CNN, Bidirectional LSTM, Transformer |
| **Target Metric** | Continuous HOMA-IR ($R^2$, Pearson $r$, MAE) | Continuous Glucose ($\text{mg/dL}$, RMSE, Clarke Error Grid) |
| **Sensor Match** | ActiGraph GT3X+ $\approx$ Actiwatch Spectrum | Commercial Smartwatch PPG, EDA, Temp, Accelerometer |
| **Best Presentation Angle** | *"Proving our NHANES HOMA-IR model works on another hospital cohort."* | *"Proving our Deep Learning models can track continuous real-time glucose."* |

---

# PART 4: Presentation Scripts & Talking Points for Your Meeting

### Script 1: Presenting MESA Sleep Study to Your Professor
> *"Ma'am, to validate our Phase 1 and Phase 2 models on an independent external population, the ideal dataset is the **MESA Sleep Study from the National Sleep Research Resource (NSRR)**.  
> It comprises **$N = 2,237$ multi-ethnic participants** who wore wrist actigraphs for 7 consecutive days, paired with certified laboratory Fasting Insulin and Fasting Glucose $\rightarrow$ HOMA-IR.  
> Because the input features and target definition are 1:1 identical to NHANES, we can feed MESA data directly into our trained LightGBM and XGBoost models without altering targets, proving our algorithm's true generalizability to external clinical populations."*

---

### Script 2: Presenting Duke BIG IDEAs Lab to Your Professor
> *"Ma'am, to validate our Deep Learning models on dynamic, real-time sensing, the ideal choice is the **Duke BIG IDEAs Lab dataset from PhysioNet**.  
> It pairs an **Empatica E4 medical wristband** (capturing PPG, Electrodermal Activity, Skin Temperature, and Accelerometer) with a **Dexcom G6 Continuous Glucose Monitor** recording glucose every 5 minutes across 8–10 days ($26,000$ paired epochs).  
> This allows us to test our **1D-CNN and Bi-LSTM temporal sequence models** on real-time glucose forecasting, showing that wrist wearables can track minute-level glycemic spikes."*

---

### Script 3: The Combined Two-Pillar Defense (Recommended)
> *"Ma'am, rather than viewing these as mutually exclusive, they address two complementary aspects of metabolic health:  
> 1. **MESA ($N = 2,237$)** provides **population-scale external validation** for our static/circadian HOMA-IR screening models.  
> 2. **Duke BIG IDEAs ($26,000$ epochs)** provides **high-frequency temporal validation** for our deep learning continuous glucose forecasting models.  
> Presenting both demonstrates that our wearable AI pipeline functions effectively for both chronic risk stratification and real-time continuous monitoring."*
