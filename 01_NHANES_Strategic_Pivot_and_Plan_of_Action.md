# Research Pivot & Implementation Strategy: Wearable AI for Metabolic Health & Insulin Resistance Screening

**Project Title:** Non-Invasive Metabolic Risk Prediction and Insulin Resistance Screening Using Multi-Modal Wearable Sensors  
**Candidate:** B.Tech Capstone Project  
**Author:** SARVAGYA-TIWARI  
**Domain:** Biomedical AI, Digital Health, Physiological Time-Series Machine Learning  
**Date:** September 2026  

---

## 1. Executive Summary & Why We Pivoted from D1NAMO

At the start of this project, we explored the **D1NAMO dataset** as a testbed for wearable glucose dynamics. D1NAMO was valuable as an engineering sandbox: it allowed us to build signal processing pipelines for electrocardiograms (Pan-Tompkins QRS detection, millisecond-level HRV extraction like SDNN and RMSSD) and construct a Leave-One-Subject-Out (LOSO) cross-validation benchmark across 10 machine learning architectures.

However, as we deepened our clinical literature review—specifically studying Princeton's **SweetDeep** study (published in Nature Digital Medicine) and Google's flagship **Insulin Resistance Prediction** study—we identified severe structural bottlenecks in D1NAMO that would prevent our project from reaching publication-grade or clinically meaningful conclusions.

### Critical Problems Faced in D1NAMO
1. **Microscopic Sample Size ($N = 9$ T1D participants):**
   * D1NAMO contains continuous sensor recordings for only 9 individuals. In medical AI, training complex models (LSTMs, XGBoost, Transformers) on 9 subjects carries an extreme risk of overfitting to subject-specific idiosyncratic baselines. A single outlier drastically skews performance metrics, making population-level generalizability impossible to claim.
2. **Severe Physiological Mismatch (Type 1 Diabetes vs. Prediabetes/Insulin Resistance):**
   * Our primary research goal is the non-invasive early detection of **Insulin Resistance (IR), Prediabetes, and early Type 2 Diabetes**—the silent metabolic dysfunction affecting hundreds of millions of undiagnosed adults.
   * All 9 D1NAMO subjects have **Type 1 Diabetes**, an autoimmune condition characterized by complete pancreatic beta-cell destruction. Their blood glucose fluctuations are driven primarily by **exogenous insulin injections and boluses**. Wearable sensors on the wrist or chest cannot sense or anticipate an unlogged insulin shot. Trying to forecast glucose without knowing insulin doses is attempting to model an unobservable causal factor.
3. **Zero Ground-Truth Metabolic Biomarkers:**
   * D1NAMO has continuous interstitial glucose monitor (CGM) tracings, but **no fasting insulin, no fasting glucose lab draws, and no HbA1c**. Therefore, it is clinically impossible to calculate **HOMA-IR** (the gold standard of insulin resistance) or evaluate against the American Diabetes Association (ADA) diagnostic guidelines.
4. **Unrealistic Device Form Factor:**
   * D1NAMO relies on the **Zephyr BioHarness 3.0**, a clinical sports chest belt that users will not wear in daily life. Consumer digital health is built around **wrist-worn devices** (smartwatches, smartbands).

For these reasons, we decided to pivot to a gold-standard, population-scale dataset with clinical blood chemistry and wrist actigraphy.

---

## 2. The Dataset Landscape: Why NHANES Was the Best Choice

Before selecting our new dataset, we evaluated the entire public medical dataset landscape against a rigorous 6-question framework:
1. **Labels:** Does it have gold-standard laboratory fasting glucose, fasting insulin (HOMA-IR), and HbA1c?
2. **Signals:** Does it have multi-day longitudinal wrist sensor data (accelerometry, activity counts, sleep)?
3. **Cohort Size:** Is the sample size large enough ($N > 1,000$) to train robust deep learning and machine learning models?
4. **Device:** Is the sensor wrist-worn (matching commercial smartwatches)?
5. **Population:** Is it a diverse, real-world population (healthy, prediabetic, undiagnosed)?
6. **Access:** Is it completely free and open without administrative delays?

### Candidate Comparison Table

| Dataset | Cohort Size ($N$) | Wearable Sensor | Clinical Ground Truth | Availability | Overall Suitability |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **NHANES (2011–2014)** | **~4,000 usable adults** | **7-day continuous wrist ActiGraph (80 Hz)** | **Fasting Insulin + Glucose $\rightarrow$ HOMA-IR, HbA1c, Lipids** | **100% Free & Open (Direct CDC download)** | **Primary Choice (Score: 10/10)** |
| **MESA Sleep** | ~2,000 | Wrist actigraphy + PSG sleep staging | Fasting glucose, HbA1c (No insulin $\rightarrow$ No HOMA-IR) | Free with registration approval | Supplementary (Tier 2) |
| **UK Biobank** | ~100,000 | 7-day wrist accelerometer | Full blood panel, HOMA-IR, HbA1c | Requires 3–6 month institutional approval & fees | Infeasible for current timeline |
| **BIDMC PPG** | 53 ICU patients | 125 Hz PPG, ECG, Respiration | No metabolic labels | 100% Free (PhysioNet) | Feature Algorithm Sandbox (Tier 3) |
| **LifeSnaps** | 71 | Fitbit HR, Sleep, Steps | Self-reported survey scores only (No blood tests) | Free (Kaggle) | Exploration only |

### The Trade-Off We Considered (And Why It Is an Overwhelming Win)
* **The Trade-Off:** NHANES does not have continuous raw 125 Hz PPG waveforms; its wearable signal is **7-day continuous triaxial wrist accelerometry (ActiGraph GT3X+)** paired with **resting heart rate** and **clinical blood panels**.
* **Why This Favors Us:** 
  1. In medical AI, models are judged primarily on the validity of their ground-truth labels. Having **HOMA-IR** on 3,745 participants provides an unshakeable clinical anchor.
  2. In Google’s landmark paper on predicting insulin resistance, the strongest non-invasive predictors were **resting heart rate ($r = +0.27$)**, **daily activity volume ($r = -0.25$)**, and **circadian rhythm stability**—all of which NHANES possesses.
  3. Continuous wrist actigraphy is the core operational sensor modality of every commercial smartwatch.
  4. To preserve our ability to demonstrate raw physiological waveform processing in our dissertation, we maintain a 3-tier structure: **NHANES** for core metabolic ML training, and **BIDMC** as a dedicated signal processing sandbox for PPG pulse morphology algorithms.

---

## 3. What NHANES Is Entirely About

The **National Health and Nutrition Examination Survey (NHANES)** is a premier epidemiological study conducted continuously by the **US Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics (NCHS)**.

Unlike typical hospital datasets, NHANES does not sample sick hospital patients; it uses a complex, multistage probability sampling design to survey a nationally representative cross-section of the population. Each participant undergoes:
1. An extensive in-home interview covering health history, lifestyle, and demographics.
2. A comprehensive physical examination in specialized Mobile Examination Centers (MECs).
3. A morning laboratory fasting blood draw conducted by trained phlebotomists.
4. An objective, 24-hour longitudinal wearable monitoring protocol where participants wear a calibrated triaxial accelerometer on their non-dominant wrist for 7 consecutive days.

For our project, we downloaded and unified the two consecutive survey cycles where this wrist-worn protocol was executed:
* **Cycle G (2011–2012)**
* **Cycle H (2013–2014)**

---

## 4. Complete Inventory of Downloaded Data Sheets

We developed an automated pipeline that retrieved all **29 component files (307.34 MB total)** from the CDC servers into `d:\BTP\data\nhanes\`.

### 1. Laboratory Data (Ground Truth Targets)
* **`GLU_G.XPT` & `GLU_H.XPT` / `INS_H.XPT` (Fasting Glucose & Insulin):**
  * *Variables:* `LBXGLU` (fasting plasma glucose, mg/dL), `LBDGLUSI` (glucose in mmol/L), `LBXIN` (fasting serum insulin, $\mu\text{U/mL}$).
  * *Usage:* These two values allow exact computation of **HOMA-IR**, the clinical gold standard for insulin resistance.
* **`GHB_G.XPT` & `GHB_H.XPT` (Glycohemoglobin):**
  * *Variables:* `LBXGH` (HbA1c percentage).
  * *Usage:* Measures long-term (8–12 week) glycemic regulation; used to define ADA prediabetes and diabetes thresholds.
* **`HDL_G.XPT` & `HDL_H.XPT` (HDL Cholesterol):**
  * *Variables:* `LBDHDD` (direct HDL cholesterol, mg/dL).
* **`TRIGLY_G.XPT` & `TRIGLY_H.XPT` (Triglycerides & LDL):**
  * *Variables:* `LBXTR` (triglycerides, mg/dL), `LBDLDL` (calculated LDL).
  * *Usage:* Dyslipidemia is directly linked with insulin resistance; enables computation of the **Triglyceride-Glucose (TyG) Index**.
* **`TCHOL_G.XPT` & `TCHOL_H.XPT` (Total Cholesterol):**
  * *Variables:* `LBXTC` (total serum cholesterol, mg/dL).

### 2. Wearable Accelerometry Signals (The Predictors)
* **`PAXDAY_G.XPT` & `PAXDAY_H.XPT` (Day-Level Activity Summaries):**
  * *Variables:* `PAXDAYD` (day number 1–7), `PAXAISMD` (total daily activity counts), `PAXMTSD` (triaxial MIMS physical activity volume), `PAXWWMD` (wake wear minutes), `PAXSWMD` (sleep wear minutes), `PAXNWMD` (non-wear minutes).
  * *Usage:* Aggregated into weekly wear compliance, daily activity volume, and sleep duration per subject.
* **`PAXHR_G.XPT` & `PAXHR_H.XPT` (Hourly Accelerometry Time-Series — 122 MB & 138 MB):**
  * *Variables:* `PAXTMH` (hour duration), `PAXAISMH` (hourly activity counts), `PAXMTSH` (hourly MIMS physical activity), `PAXWWMH` (hourly wake minutes), `PAXSWMH` (hourly sleep minutes).
  * *Usage:* 168 consecutive hourly time steps per participant. Used for deep learning sequence models (1D-CNN, LSTM) and non-parametric circadian rhythm analysis (IS, IV, M10, L5, Cosinor).
* **`PAXHD_G.XPT` & `PAXHD_H.XPT` (Sensor Header Metadata):**
  * *Variables:* `PAXSTS` (data quality flag), `PAXFTIME` (start time of recording on first day), `PAXHAND` (wear wrist).

### 3. Examination & Anthropometrics
* **`BMX_G.XPT` & `BMX_H.XPT` (Body Measures):**
  * *Variables:* `BMXBMI` (Body Mass Index), `BMXWAIST` (waist circumference in cm — central adiposity marker), `BMXWT` (weight in kg), `BMXHT` (height in cm).
* **`BPX_G.XPT` & `BPX_H.XPT` (Blood Pressure & Resting Vitals):**
  * *Variables:* `BPXPLS` (60-second resting pulse / heart rate in bpm), `BPXSY1` (systolic blood pressure), `BPXDI1` (diastolic blood pressure).
  * *Usage:* Resting pulse serves as our primary non-invasive autonomic tone digital biomarker.

### 4. Medical Questionnaires
* **`DIQ_G.XPT` & `DIQ_H.XPT` (Diabetes Questionnaire):**
  * *Variables:* `DIQ010` (doctor ever diagnosed diabetes: 1=Yes, 2=No, 3=Borderline), `DIQ050` (currently taking insulin).
  * *Usage:* Serves as an essential **exclusion filter** so our models are trained strictly on non-diabetic and undiagnosed individuals.
* **`SLQ_G.XPT` & `SLQ_H.XPT` (Sleep Questionnaire):**
  * *Variables:* `SLD010H` (self-reported usual sleep hours on weekdays).
* **`PAQ_G.XPT` & `PAQ_H.XPT` (Physical Activity Questionnaire):**
  * *Variables:* `PAD680` (self-reported daily sedentary minutes).

---

## 5. The Unified Master Dataset

By joining these 29 sheets on participant identifier `SEQN` and applying clinical inclusion/exclusion criteria, we created the master analytical dataset:
* **Storage Location:** [`data/nhanes/nhanes_unified_master_with_circadian.csv`](file:///d:/BTP/data/nhanes/nhanes_unified_master_with_circadian.csv) (3.13 MB) and `.parquet` (1.12 MB).

### Inclusion and Exclusion Rules Applied
1. **Adult Focus:** Age between 18 and 65 years inclusive.
2. **Exclusion of Diagnosed Diabetics:** Excluded individuals with a prior physician diagnosis of diabetes (`DIQ010 == 1`) or currently on insulin (`DIQ050 == 1`). This forces the models to learn subtle early metabolic dysfunction rather than drug effects.
3. **Complete Lab Panels:** Must have valid morning fasting plasma glucose (`LBXGLU`) and serum insulin (`LBXIN`).

### Final Cohort Dimensions
* **Total Usable Target Adults:** **3,745 participants**
* **Participants with 7-Day Continuous Accelerometry + Circadian Extraction:** **3,296 participants**
* **Total Features per Subject:** 54 columns spanning demographics, blood chemistry, anthropometrics, vitals, daily activity metrics, and 12 engineered circadian parameters.

---

## 6. How the Original Work Plan Adapts to NHANES

We have retained the exact philosophical structure of the original two phases approved by the professor, while adapting the targets to real-world clinical laboratory standards:

```
                            TRANSFORMATION OF THE WORK PLAN
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 1: CONTINUOUS METABOLIC BIOMARKER REGRESSION                                     │
 │ Objective: Predict continuous laboratory metabolic values from wearable data.         │
 │ Targets:                                                                               │
 │   1. Continuous HOMA-IR (Primary: Insulin Resistance Severity)                         │
 │   2. Continuous HbA1c (%) (Secondary: Long-term Glycemic Burden)                       │
 │   3. Continuous Fasting Plasma Glucose (mg/dL)                                         │
 │ Predictors: 168-hour actigraphy sequence + Circadian parameters + Resting HR + Demographics │
 │ Models:                                                                                │
 │   • Regularized Baselines: Ridge, Lasso, ElasticNet                                    │
 │   • Non-Linear Ensembles: Random Forest, XGBoost, LightGBM                             │
 │   • Deep Learning Sequence Models: 1D-CNN, LSTM, GRU                                   │
 │ Evaluation: MAE, RMSE, Pearson r, R², Bland-Altman Agreement Plots                     │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 2: NON-INVASIVE METABOLIC RISK SCREENING (ZERO BLOOD ACCESS)                     │
 │ Objective: Screen individuals for metabolic risk using ONLY smartwatch-accessible data. │
 │ Constraints: Zero invasive blood tests or glucose history available at test time.     │
 │ Subtask A (Metabolic Trajectory / TyG Surrogate):                                      │
 │   • Classify elevated Triglyceride-Glucose Index (surrogate of hepatic lipogenesis).   │
 │ Subtask B (ADA 3-Class Clinical Risk Category):                                        │
 │   • Class 0: Low Risk / Insulin Sensitive (HOMA-IR < 2.0 & HbA1c < 5.7%)               │
 │   • Class 1: Moderate Risk / Prediabetes (HOMA-IR 2.0–2.9 OR HbA1c 5.7%–6.4%)         │
 │   • Class 2: High Risk / Insulin Resistant (HOMA-IR ≥ 3.0 OR HbA1c ≥ 6.5%)             │
 │ Models: Logistic Regression, Random Forest, LightGBM, XGBoost, Multi-Layer Perceptron   │
 │ Evaluation: Balanced Accuracy, Macro F1, AUROC, Confusion Matrix, Subgroup Fairness    │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Step-by-Step Implementation Plan of Action

Here is our precise execution roadmap:

```
                                  EXECUTION ROADMAP
  [Step 1: Feature Extraction] ──────────► [Step 2: Phase 1 Regression Benchmark]
  • 24-h Cosinor Model (Mesor, Amp)         • Ridge, Lasso, RF, XGBoost, LightGBM
  • NPCRA (IS, IV, M10, L5, RA)            • 1D-CNN & LSTM on 168-hour sequences
  • Sedentary bouts & sleep metrics         • Evaluate MAE, RMSE, R², Pearson r
             │                                              │
             ▼                                              ▼
  [Step 4: Thesis Documentation] ◄──────── [Step 3: Phase 2 Risk Screening]
  • Comprehensive reports & figures         • Zero blood access inference
  • Clinical interpretation & SHAP          • 3-Class ADA risk classification
  • Ready for publication & viva defense    • Balanced Acc, Macro F1, AUROC
```

* **Step 1 (Completed):** Circadian rhythm and wearable feature extraction from `PAXHR_G` and `PAXHR_H`. 12 circadian and activity parameters extracted for 3,296 subjects and validated.
* **Step 2 (Immediate Next Step):** Phase 1 Continuous Biomarker Regression Benchmark across tabular and deep learning sequence architectures.
* **Step 3:** Phase 2 Non-Invasive 3-Class Clinical Risk Classification Benchmark under strict stratified inter-subject cross-validation.
* **Step 4:** Model Explainability (SHAP feature attribution), clinical subgroup analysis (stratified by BMI and age tiers), and dissertation synthesis.

---

## 8. Expected Impact and How Results Will Be Used

1. **Defensibility in the Final Viva:**
   * Transitioning from $N = 9$ to $N = 3,745$ participants eliminates questions regarding statistical power and overfitting.
   * Predicting HOMA-IR directly replicates and extends Google’s research, proving that consumer-grade wearable actigraphy and resting pulse can serve as an early digital biomarker for prediabetes.
2. **Clinical Translation:**
   * 42.8% of individuals with prediabetes and diabetes globally remain undiagnosed because laboratory blood draws are invasive and infrequent. A non-invasive wearable algorithm running passively on a smartwatch can provide continuous risk stratification at zero marginal cost, flagging high-risk individuals for confirmatory lab tests before irreversible beta-cell failure occurs.
3. **Publication-Ready Foundation:**
   * The methodology, sample size, and clinical alignment make this work suitable for submission to digital health and biomedical informatics conferences/journals (e.g., IEEE JBHI, ACM CHIL, or Nature Digital Medicine).
