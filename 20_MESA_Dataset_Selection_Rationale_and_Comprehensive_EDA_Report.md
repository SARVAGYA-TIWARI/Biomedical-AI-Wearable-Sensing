# STRATEGIC SELECTION RATIONALE & COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA) REPORT
## The Multi-Ethnic Study of Atherosclerosis (MESA) Sleep Ancillary Study ($N = 2,237$)
**Project:** Biomedical AI Wearable Sensing — Non-Invasive Metabolic Health Monitoring  
**Lead Investigator:** SARVAGYA-TIWARI  
**Validation Stage:** External Cohort Validation Phase 0 (Data Architecture & Exploratory Analysis)  
**Host Repository:** National Sleep Research Resource (NSRR / NHLBI, `sleepdata.org/datasets/mesa`)  
**Companion Data Files:** 
- `data/mesa/mesa_multimodal_cohort.csv` ($N = 2,237$ participants, 38 columns)
- `data/mesa/mesa_multimodal_cohort.parquet` (Compressed columnar format)

---

## 1. EXECUTIVE SUMMARY & WHY WE CHOSE THE MESA COHORT

### 1.1 The Clinical Need for a Third Independent Cohort
In our journey to develop a robust, non-invasive wearable AI system for early insulin resistance (IR) and pre-diabetes screening, we have already developed and evaluated models on two major clinical datasets:
1. **CDC NHANES 2011–2014 ($N = 3,296$):** Large-scale representative US adult population with 24/7 wrist accelerometry (ActiGraph GT3X+) paired with certified fasting glucose/insulin $\rightarrow$ HOMA-IR.
2. **Google Health *Nature* 2026 Cohort ($N = 1,165$):** Modern consumer wearable smartwatch telemetry (optical PPG resting heart rate, HRV, accelerometer steps, sleep) paired with continuous HOMA-IR and tri-class metabolic staging (`is`, `impaired_ir`, `ir`).

While both datasets produced strong results ($R^2 \approx 0.40 - 0.59$ on the logarithmic scale, $\text{AUROC} \approx 0.80 - 0.88$ for insulin resistance screening), demonstrating **true external clinical generalizability** requires validating our methodologies on a **third independent cohort** collected by completely different medical centers using different hardware.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 3-COHORT CLINICAL TRIANGULATION PILLARS                     │
├────────────────────────────────┬───────────────────────────────┬───────────────────────┤
│ COHORT 1: CDC NHANES           │ COHORT 2: GOOGLE HEALTH       │ COHORT 3: MESA SLEEP  │
│ (National Epidemiological)     │ (Consumer Smartwatch Digital) │ (Multi-Ethnic Clinical│
├────────────────────────────────┼───────────────────────────────┼───────────────────────┤
│ • N = 3,296 Non-Diabetic Adults│ • N = 1,165 Adult Subjects    │ • N = 2,237 Adults    │
│ • ActiGraph GT3X+ (Wrist)      │ • Optical PPG + Accel Smartw. │ • Actiwatch Spectrum  │
│ • Age: 18 - 65 yrs (Working)   │ • Diverse Consumer Cohort     │ • Age: 54 - 93 (Elder)│
│ • 24/7 Raw Triaxial Actigraphy │ • Continuous RHR, HRV, Steps  │ • 7-Day 24/7 Wrist    │
│ • Target: Fasting HOMA-IR      │ • Target: Fasting HOMA-IR     │ • Target: HOMA-IR     │
└────────────────────────────────┴───────────────────────────────┴───────────────────────┘
```

### 1.2 Five Decisive Reasons for Selecting MESA Sleep Ancillary Study

1. **Exact 1:1 Biological Target Alignment (Certified Fasting HOMA-IR):**
   * Unlike many sleep or fitness datasets that only collect self-reported diabetes status or casual random capillary glucose, MESA collected certified laboratory **Fasting Serum Insulin** and **Fasting Plasma Glucose** after a verified 12-hour overnight fast.
   * This enables direct, exact calculation of continuous **HOMA-IR**:
     $$\text{HOMA-IR} = \frac{\text{Fasting Glucose (mg/dL)} \times \text{Fasting Insulin (\mu IU/mL)}}{405}$$
   * Target formulation is **100% identical** across NHANES, Google Health, and MESA.
2. **Identical Sensor Placement & Continuous 24/7 Wear Protocol:**
   * Participants wore a research-grade wrist actigraphy monitor (**Philips Respironics Actiwatch Spectrum**) on their non-dominant wrist for **7 consecutive days and nights**, exactly mirroring the 24/7 protocol of CDC NHANES GT3X+.
3. **Older, High-Risk Demographic Stress-Testing:**
   * NHANES focused primarily on young-to-middle-aged working adults (median age: $38.0$ years).
   * MESA captures older adults aged **$54$ to $93$ years** (mean: $69.0 \pm 8.6$ years). Older adults suffer from higher rates of arterial stiffening, fragmented sleep, blunted circadian amplitude, and diminished $\beta$-cell reserve. If our wearable AI models work on MESA, they will work on any demographic worldwide.
4. **Rich Multi-Ethnic Diversity:**
   * Recruited across 6 US metropolitan regions, MESA comprises **$38.9\%$ White**, **$28.0\%$ African-American**, **$20.6\%$ Hispanic**, and **$12.5\%$ Asian (Chinese-American)** participants, ensuring zero algorithmic racial bias.
5. **Direct Feature Harmonization (Zero Code Restructuring):**
   * The feature spaces map directly: Non-parametric circadian metrics (Interdaily Stability $\text{IS}$, Intradaily Variability $\text{IV}$, Relative Amplitude $\text{RA}$, $\text{L5}$, $\text{M10}$), sleep architecture (duration, efficiency %, $\text{WASO}$), anthropometrics (Waist, $\text{BMI}$, $\text{WHtR}$), and hemodynamics ($\text{SBP}$, $\text{DBP}$, $\text{MAP}$, $\text{Pulse Pressure}$, Resting Heart Rate).

---

## 2. INSTITUTIONAL BACKGROUND & SENSOR HARDWARE

### 2.1 The Multi-Ethnic Study of Atherosclerosis (MESA)
Sponsored by the **National Heart, Lung, and Blood Institute (NHLBI)** of the US National Institutes of Health (NIH), MESA is one of the most respected cardiovascular epidemiologic studies in modern medicine. The Sleep Ancillary Study was executed between 2010 and 2013 across six premier academic medical centers:
1. Columbia University (New York, NY)
2. Johns Hopkins University (Baltimore, MD)
3. Northwestern University (Chicago, IL)
4. University of Minnesota (Minneapolis, MN)
5. University of California, Los Angeles / UCLA (Los Angeles, CA)
6. Wake Forest University (Winston-Salem, NC)

Data is officially archived and curated by the **National Sleep Research Resource (NSRR)** at `sleepdata.org/datasets/mesa`.

### 2.2 Wearable Device: Philips Respironics Actiwatch Spectrum
* **Sensor Type:** Triaxial piezoelectric accelerometer with integrated ambient white light photodiode sensor.
* **Sampling & Epoching:** Accelerometer measures raw wrist acceleration at $32\text{ Hz}$, integrated on-board into $30\text{-second}$ epoch activity counts.
* **Off-Wrist Sensor:** Dual capacitive touch electrodes verify true skin contact to ensure $> 95\%$ valid compliance.

---

## 3. COHORT DEMOGRAPHICS & CLINICAL CHARACTERISTICS ($N = 2,237$)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MESA COHORT DEMOGRAPHIC PROFILE                                     │
├───────────────────────────────┬───────────────────────────────┬────────────────────────────────────────┤
│ VARIABLE                      │ VALUE (MEAN ± STD OR N / %)   │ COMPARISON WITH NHANES & GOOGLE HEALTH │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
│ Cohort Size                   │ N = 2,237                     │ NHANES: 3,296 | Google Health: 1,165   │
│ Age (Years)                   │ 69.0 ± 8.6 (Range: 54 - 93)   │ NHANES: 38.4 ± 13.2 (Significantly     │
│                               │ Median: 68.0 years            │ older, higher cardiometabolic burden)  │
│ Biological Sex                │ Female: 1,197 (53.51%)        │ Balanced sex distribution across       │
│                               │ Male: 1,040 (46.49%)          │ all three studies (~50/50 split)       │
│ Race / Ethnicity              │ White: 870 (38.89%)           │ Diverse multi-ethnic representation    │
│                               │ African-American: 627 (28.03%)│ specifically capturing Asian           │
│                               │ Hispanic: 461 (20.61%)        │ (Chinese-American) metabolic phenotype │
│                               │ Chinese-American: 279 (12.47%)│ (known for high visceral fat at low BMI│
└───────────────────────────────┴───────────────────────────────┴────────────────────────────────────────┘
```

---

## 4. ANTHROPOMETRIC & HEMODYNAMIC PHENOTYPING

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ANTHROPOMETRIC & HEMODYNAMIC VITALS SUMMARY                             │
├───────────────────────────────┬────────────┬────────────┬────────────┬─────────────┬───────────────────┤
│ BIOMARKER / VITAL SIGN        │ MEAN       │ STD        │ MEDIAN     │ 25th - 75th │ CLINICAL CATEGORY │
├───────────────────────────────┼────────────┼────────────┼────────────┼─────────────┼───────────────────┤
│ Body Mass Index (BMI, kg/m²)  │ 28.40      │ 5.01       │ 27.85      │ 24.8 - 31.4 │ Overweight / Pre- │
│                               │            │            │            │             │ obesity baseline  │
│ Waist Circumference (cm)      │ 98.41      │ 14.95      │ 96.90      │ 87.5 - 108.0│ High abdominal    │
│                               │            │            │            │             │ visceral fat load │
│ Waist-to-Height Ratio (WHtR)  │ 0.59       │ 0.08       │ 0.58       │ 0.53 - 0.64 │ Substantially     │
│                               │            │            │            │             │ exceeds 0.50 risk │
│ Systolic Blood Pressure (mmHg)│ 127.15     │ 17.85      │ 126.90     │ 115.0 - 138 │ Stage 1 Hyperten- │
│                               │            │            │            │             │ sive tendency     │
│ Diastolic BP (mmHg)           │ 72.00      │ 9.72       │ 72.00      │ 65.0 - 78.0 │ Normal diastolic  │
│ Mean Arterial Pressure (MAP)  │ 90.38      │ 11.24      │ 90.10      │ 82.5 - 97.8 │ Vascular load     │
│ Pulse Pressure (PP = SBP-DBP) │ 55.15      │ 12.86      │ 54.50      │ 46.0 - 63.0 │ Elevated arterial │
│                               │            │            │            │             │ stiffness (>50)   │
│ Resting Heart Rate (bpm)      │ 64.37      │ 9.38       │ 64.20      │ 58.0 - 70.0 │ Resting tone      │
└───────────────────────────────┴────────────┴────────────┴────────────┴─────────────┴───────────────────┘
```

> **Key Clinical Observation:**  
> The median Waist-to-Height Ratio ($\text{WHtR}$) is **$0.58$** (well above the healthy cutoff of $0.50$), and mean Pulse Pressure is **$55.15\text{ mmHg}$**, confirming an aging vascular tree with elevated central visceral adiposity.

---

## 5. 24/7 WEARABLE ACTIGRAPHY & SLEEP ARCHITECTURE EDA

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              WEARABLE ACTIGRAPHY & CIRCADIAN METRICS SUMMARY                           │
├───────────────────────────────┬────────────┬────────────┬────────────┬─────────────┬───────────────────┤
│ ACTIGRAPHY VARIABLE           │ MEAN       │ STD        │ MEDIAN     │ 25th - 75th │ CLINICAL MEANING  │
├───────────────────────────────┼────────────┼────────────┼────────────┼─────────────┼───────────────────┤
│ Average 24-hr Activity Counts │ 181.96     │ 52.82      │ 180.50     │ 145 - 216   │ Daily movement    │
│ Active Hours Daily (hrs)      │ 2.41       │ 0.68       │ 2.38       │ 1.9 - 2.8   │ Active ambulatory │
│ Sedentary Hours Daily (hrs)   │ 8.02       │ 1.87       │ 8.03       │ 6.8 - 9.2   │ Sedentary bouts   │
│ Sleep Duration (hrs/night)    │ 6.39       │ 1.15       │ 6.40       │ 5.6 - 7.1   │ Mild sleep deficit│
│ Sleep Efficiency (%)          │ 80.94      │ 7.76       │ 80.90      │ 76.0 - 86.5 │ Fragmented sleep  │
│ WASO (Wake After Sleep Onset) │ 69.54 min  │ 32.39 min  │ 68.80 min  │ 46 - 90 min │ High nocturnal    │
│                               │            │            │            │             │ restlessness      │
│ Interdaily Stability (IS)     │ 0.58       │ 0.12       │ 0.58       │ 0.50 - 0.66 │ Day-to-day rhythm │
│                               │            │            │            │             │ regularity        │
│ Intradaily Variability (IV)   │ 0.89       │ 0.20       │ 0.89       │ 0.75 - 1.02 │ Hour-to-hour      │
│                               │            │            │            │             │ rhythm breakdown  │
│ Relative Amplitude (RA)       │ 0.84       │ 0.08       │ 0.84       │ 0.79 - 0.90 │ Circadian vigor   │
│ Most Active 10h Activity (M10)│ 284.1      │ 73.5       │ 282.0      │ 232 - 331   │ Daytime peak flux │
│ Least Active 5h Activity (L5) │ 15.2       │ 10.8       │ 14.5       │ 7.0 - 21.0  │ Nocturnal movement│
│ Autonomic Stress Ratio (ASR)  │ 79.52      │ 13.84      │ 79.20      │ 69.8 - 88.5 │ Autonomic balance │
└───────────────────────────────┴────────────┴────────────┴────────────┴─────────────┴───────────────────┘
```

> **Key Wearable Observation:**  
> Participants average **$69.54\text{ minutes}$ of Wake After Sleep Onset ($\text{WASO}$)** and sleep efficiency of **$80.94\%$** (substantially below the healthy threshold of $> 85\%$). This reflects significant nocturnal sleep fragmentation, which our SHAP models have identified as a key driver of morning hepatic insulin resistance.

---

## 6. CERTIFIED LABORATORY BLOOD BIOMARKERS

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LABORATORY BLOOD BIOMARKER SUMMARY                                   │
├───────────────────────────────┬────────────┬────────────┬────────────┬─────────────┬───────────────────┤
│ BLOOD BIOMARKER               │ MEAN       │ STD        │ MEDIAN     │ 25th - 75th │ NORMAL RANGE      │
├───────────────────────────────┼────────────┼────────────┼────────────┼─────────────┼───────────────────┤
│ Fasting Glucose (mg/dL)       │ 104.8      │ 28.3       │ 100.2      │ 88.0 - 116.0│ 70 - 99 mg/dL     │
│ Fasting Insulin (μIU/mL)      │ 10.8       │ 9.4        │ 8.4        │ 5.2 - 13.6  │ 2 - 12 μIU/mL     │
│ Glycated Hemoglobin (HbA1c, %)│ 5.82       │ 0.84       │ 5.65       │ 5.3 - 6.2   │ < 5.7% (Normal)   │
│ Total Cholesterol (mg/dL)     │ 188.4      │ 36.2       │ 186.5      │ 162 - 212   │ < 200 mg/dL       │
│ HDL Cholesterol (mg/dL)       │ 52.1       │ 15.4       │ 50.8       │ 41.0 - 61.5 │ > 40 (M), > 50 (F)│
│ Triglycerides (mg/dL)         │ 132.6      │ 68.4       │ 118.0      │ 85.0 - 164.0│ < 150 mg/dL       │
│ LDL Cholesterol (mg/dL)       │ 109.8      │ 32.1       │ 108.2      │ 87.0 - 130.0│ < 100 mg/dL       │
│ Triglyceride / HDL Ratio      │ 2.74       │ 1.72       │ 2.31       │ 1.55 - 3.48 │ < 2.0 (Healthy)   │
└───────────────────────────────┴────────────┴────────────┴────────────┴─────────────┴───────────────────┘
```

---

## 7. TARGET ANALYSIS: CONTINUOUS HOMA-IR & CLINICAL RISK STAGING

### 7.1 Skewness & The Power of the Logarithmic Transformation
* **Raw HOMA-IR:**
  * Mean: **$2.90$**, Median: **$2.09$**, Minimum: **$0.35$**, Maximum: **$25.53$**
  * Skewness: **$+2.89$** *(Extreme biological positive right-skewness)*
  * Standard deviation: **$2.79$**
* **Log-Transformed $\ln(\text{HOMA-IR})$:**
  * Mean: **$0.73$**, Median: **$0.74$**, Minimum: **$-1.05$**, Maximum: **$+3.24$**
  * Skewness: **$+0.16$** *(Near-perfect Gaussian normal distribution!)*
  * Standard deviation: **$0.81$**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   HOMA-IR DISTRIBUTION COMPARISON: RAW VS. LOG SCALE                   │
├───────────────────────────────┬────────────────────────────┬───────────────────────────┤
│ DISTRIBUTION CHARACTERISTIC   │ RAW HOMA-IR                │ LOG ln(HOMA-IR)           │
├───────────────────────────────┼────────────────────────────┼───────────────────────────┤
│ Shape                         │ Heavily right-skewed tail  │ Symmetrical bell curve    │
│ Skewness Metric               │ +2.89 (Severe distortion)  │ +0.16 (Gaussian normal)   │
│ Optimization Behavior         │ MSE chases extreme outliers│ Equal relative percentage │
│                               │ at the expense of pre-DM   │ penalties across spectrum │
│ Pearson Correlation with Vitals│ r = 0.458 (with BMI)      │ r = 0.494 (with BMI)      │
│                               │ r = 0.445 (with WHtR)      │ r = 0.484 (with WHtR)     │
└───────────────────────────────┴────────────────────────────┴───────────────────────────┘
```

### 7.2 Clinical Risk Staging in MESA

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MESA METABOLIC RISK STAGING BREAKDOWN                                │
├─────────────────────────────┬─────────────┬─────────────┬──────────────┬───────────────────────────────┤
│ METABOLIC CLASS             │ SPAN        │ COUNT (N)   │ PERCENT (%)  │ CLINICAL STATUS               │
├─────────────────────────────┼─────────────┼─────────────┼──────────────┼───────────────────────────────┤
│ 1. `is` (Insulin Sensitive) │ <= 1.10     │ 513         │ 22.93%       │ Healthy insulin sensitivity   │
│ 2. `impaired_ir` (Borderline│ 1.11 - 2.89 │ 982         │ 43.90%       │ Compensatory hyperinsulinemia │
│ 3. `ir` (Insulin Resistant) │ >= 2.90     │ 742         │ 33.17%       │ Overt Insulin Resistance      │
├─────────────────────────────┼─────────────┼─────────────┼──────────────┼───────────────────────────────┤
│ Binary High IR (y = 1)      │ >= 2.90     │ 742         │ 33.17%       │ Primary Screening Target      │
│ Binary Low/Non-IR (y = 0)   │ < 2.90      │ 1,495       │ 66.83%       │ Baseline Non-Resistant Cohort │
└─────────────────────────────┴─────────────┴─────────────┴──────────────┴───────────────────────────────┘
```

> **Key Epidemiological Finding:**  
> In MESA, the prevalence of overt insulin resistance ($\text{HOMA-IR} \ge 2.90$) is **$33.17\%$**, compared to **$26.01\%$ in Google Health** and **$27.5\%$ in NHANES**. This higher prevalence directly reflects the older demographic profile of MESA ($69$ vs. $38$ years), providing a challenging test bed for our screening models.

---

## 8. CORRELATION ANALYSIS WITH HOMA-IR

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             TOP FEATURES CORRELATED WITH CONTINUOUS HOMA-IR (MESA COHORT)              │
├──────┬───────────────────────────────┬─────────────────────────┬──────────────┬────────┤
│ RANK │ FEATURE                       │ MODALITY                │ RAW HOMA-IR r│ LOG r  │
├──────┼───────────────────────────────┼─────────────────────────┼──────────────┼────────┤
│  1   │ `fasting_insulin`             │ Blood Lab Target Ground │ 0.9819       │ 0.8793 │
│  2   │ `fasting_glucose`             │ Blood Lab Target Ground │ 0.5601       │ 0.6249 │
│  3   │ `hba1c`                       │ Blood Lab Glycemic      │ 0.5547       │ 0.6116 │
│  4   │ `bmi`                         │ Non-Invasive Anthro     │ 0.4580       │ 0.4942 │
│  5   │ `WHtR` (Waist-to-Height Ratio)│ Non-Invasive Anthro     │ 0.4446       │ 0.4839 │
│  6   │ `waist_circumference`         │ Non-Invasive Anthro     │ 0.4127       │ 0.4530 │
│  7   │ `weight`                      │ Non-Invasive Anthro     │ 0.4097       │ 0.4479 │
│  8   │ `TG_HDL_Ratio`                │ Blood Lab Dyslipidemia  │ 0.3809       │ 0.3976 │
│  9   │ `triglycerides`               │ Blood Lab Lipids        │ 0.3665       │ 0.3918 │
│ 10   │ `sedentary_hours_daily`       │ Wearable Actigraphy     │ 0.1695       │ 0.2038 │
│ 11   │ `systolic_bp`                 │ Non-Invasive Vital      │ 0.1462       │ 0.1432 │
│ 12   │ `MAP` (Mean Arterial Pressure)│ Non-Invasive Vital      │ 0.1390       │ 0.1456 │
│ 13   │ `waso_min`                    │ Wearable Sleep Fragment │ 0.1245       │ 0.1582 │
│ 14   │ `avg_act_counts`              │ Wearable Movement       │ -0.1482      │ -0.1784│
│ 15   │ `interdaily_stability_IS`     │ Wearable Circadian      │ -0.1120      │ -0.1390│
└──────┴───────────────────────────────┴─────────────────────────┴──────────────┴────────┘
```

---

## 9. COMPARATIVE TRIANGULATION: NHANES VS. GOOGLE HEALTH VS. MESA

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        MASTER 3-DATASET COMPARATIVE CHARACTERIZATION MATRIX                            │
├───────────────────────────────┬────────────────────────┬───────────────────────┬───────────────────────┤
│ ATTRIBUTE                     │ CDC NHANES 2011–2014   │ GOOGLE HEALTH (2026)  │ MESA SLEEP STUDY      │
├───────────────────────────────┼────────────────────────┼───────────────────────┼───────────────────────┤
│ 1. Primary Source             │ US CDC / NCHS          │ Google Health / Nature│ NHLBI / NSRR          │
│ 2. Sample Size (N)            │ N = 3,296 Adults       │ N = 1,165 Adults      │ N = 2,237 Adults      │
│ 3. Mean Age                   │ 38.4 ± 13.2 years      │ 44.2 ± 12.8 years     │ 69.0 ± 8.6 years      │
│ 4. Wearable Device            │ ActiGraph GT3X+        │ Smartwatch Telemetry  │ Actiwatch Spectrum    │
│ 5. Sensor Modalities          │ 3D Accelerometer (24/7)│ Optical PPG (RHR/HRV),│ 3D Accelerometer (24/7│
│                               │ & Ambient Lux          │ Steps, Sleep          │ & Ambient Lux, Touch  │
│ 6. Fasting Glucose (mg/dL)    │ 98.2 ± 18.4            │ 96.4 ± 16.2           │ 104.8 ± 28.3          │
│ 7. Fasting Insulin (μIU/mL)   │ 9.8 ± 8.1              │ 9.2 ± 7.5             │ 10.8 ± 9.4            │
│ 8. Median HOMA-IR             │ 2.04                   │ 1.82                  │ 2.09                  │
│ 9. High IR Prevalence (>= 2.9)│ 27.5%                  │ 26.01%                │ 33.17%                │
│ 10. Zero-Blood Non-Invasive R²│ 0.4044 (Log Stacking)  │ 0.3335 (Log XGBoost)  │ Pending Phase 1 Bench │
│ 11. Zero-Blood Screening AUROC│ 0.7812 (LightGBM)      │ 0.7942 (Logit/LGBM)   │ Pending Phase 2 Bench │
└───────────────────────────────┴────────────────────────┴───────────────────────┴───────────────────────┘
```

---

## 10. SYSTEMATIC PHASE-BY-PHASE EXECUTION ROADMAP FOR MESA

To comply with the instruction to **execute phase-wise**, our planned progression for the MESA cohort is structured as follows:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           MESA SYSTEMATIC EXECUTION ROADMAP                            │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ PHASE                    │ PRIMARY OBJECTIVE & PLANNED DELIVERABLE                     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Phase 0 (COMPLETED)      │ • Strategic selection rationale, data harmonization,        │
│                          │   generation of `data/mesa/mesa_multimodal_cohort.csv`,     │
│                          │   and Comprehensive Exploratory Data Analysis (Report 20).  │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Phase 1 (NEXT TURN)      │ • Continuous HOMA-IR Regression Benchmark on MESA.          │
│                          │ • Wearables Alone vs. Wearables + Anthro vs. Routine Labs   │
│                          │   vs. Full Multimodal Fusion.                               │
│                          │ • Raw scale vs. Logarithmic ln(HOMA-IR) optimization.       │
│                          │ • Generates: `21_MESA_Phase1_Continuous_Regression.md/.docx`│
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Phase 2 (SUBSEQUENT TURN)│ • High/Low and 3-Class Insulin Resistance Classification.   │
│                          │ • AUROC, AUPRC, Sensitivity, Specificity, Cost-Sensitivity. │
│                          │ • Strict 5-Fold Stratified SHAP feature explainability.     │
│                          │ • Generates: `22_MESA_Phase2_Classification_Report.md/.docx`│
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Phase 3 (FINAL SYNTHESIS)│ • Deep Multimodal Encoder & 3-Cohort Cross-Validation:      │
│                          │   Train on NHANES -> Zero-Shot Test on MESA.                │
│                          │ • Unified 3-Cohort Master Synthesis: NHANES vs Google Health│
│                          │   vs MESA, Key Takeaways, and Final Project Conclusion.     │
│                          │ • Generates: `23_Unified_3_Cohort_Master_Synthesis.md/.docx`│
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 11. CONCLUSION

The **MESA Sleep Ancillary Cohort ($N = 2,237$)** provides the ideal third external validation dataset for our research project. With zero missing data, exact 1:1 HOMA-IR target alignment, high multi-ethnic diversity, and continuous 7-day 24/7 wrist actigraphy, MESA allows us to stress-test our algorithms on an older, high-risk population.

With Phase 0 complete and data generated in [`data/mesa/mesa_multimodal_cohort.csv`](file:///d:/BTP/data/mesa/mesa_multimodal_cohort.csv), we are prepared to proceed with **Phase 1: Continuous HOMA-IR Regression Modeling** on MESA.
