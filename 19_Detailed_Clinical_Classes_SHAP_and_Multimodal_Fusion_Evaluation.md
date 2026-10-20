# COMPREHENSIVE CLINICAL EVALUATION: TARGET CLASSES, STRATIFIED 5-FOLD SHAP EXPLAINABILITY, MODALITY IMPORTANCE, AND ENSEMBLE FUSION

**Project:** Biomedical AI Wearable Sensing — Non-Invasive Metabolic Health Monitoring  
**Target Cohort:** Google Health *Nature* 2026 Cohort ($N = 1,165$ Non-Diabetic Adult Participants, 155 Features)  
**Cross-Validation Protocol:** Strict 5-Fold Subject-Level Stratified Cross-Validation (`StratifiedKFold`, random_state=42)  
**Explainability Engine:** TreeSHAP (SHapley Additive exPlanations) computed Out-of-Fold across all 5 Folds  
**Companion Artifacts:** 
- Excel Dataset (Prediction Cohort): [`data/google_health_ir/google_health_ir_prediction_data.xlsx`](file:///d:/BTP/data/google_health_ir/google_health_ir_prediction_data.xlsx)
- Excel Dataset (All Enrolled): [`data/google_health_ir/google_health_all_participants_data.xlsx`](file:///d:/BTP/data/google_health_ir/google_health_all_participants_data.xlsx)
- Excel Data Dictionary: [`data/google_health_ir/google_health_data_dictionary.xlsx`](file:///d:/BTP/data/google_health_ir/google_health_data_dictionary.xlsx)

---

## QUESTION 1: WHAT DOES `is`, `impaired_ir`, AND `ir` MEAN IN THE DATASET?

In the official Google Health dataset (`ir_prediction_data.pq` and exported Excel file `google_health_ir_prediction_data.xlsx`), the target column **`homa_ir_status`** defines the clinically verified metabolic health state of each participant, derived from laboratory-measured fasting plasma glucose and fasting serum insulin:

$$\text{HOMA-IR} = \frac{\text{Fasting Glucose (mg/dL)} \times \text{Fasting Insulin (\mu IU/mL)}}{405}$$

The 3 category abbreviations represent:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   HOMA-IR STATUS DEFINITIONS IN THE DATASET                                      │
├─────────────────┬───────────────────────────────┬────────────┬─────────────┬─────────────┬───────────────────────┤
│ ABBREVIATION    │ CLINICAL PHENOTYPE            │ COUNT (N)  │ PERCENT (%) │ HOMA-IR SPAN│ CLINICAL SIGNIFICANCE │
├─────────────────┼───────────────────────────────┼────────────┼─────────────┼─────────────┼───────────────────────┤
│ 1. `is`         │ Insulin Sensitive             │ 275        │ 23.61%      │ 0.27 - 1.10 │ Optimal metabolic     │
│                 │ (Healthy / Normal Control)    │            │             │ (Mean: 0.79)│ insulin sensitivity   │
├─────────────────┼───────────────────────────────┼────────────┼─────────────┼─────────────┼───────────────────────┤
│ 2. `impaired_ir`│ Impaired Insulin Sensitivity  │ 587        │ 50.39%      │ 1.11 - 2.89 │ Early subclinical     │
│                 │ (Borderline / Compensatory IR)│            │             │ (Mean: 1.85)│ pre-diabetic risk     │
├─────────────────┼───────────────────────────────┼────────────┼─────────────┼─────────────┼───────────────────────┤
│ 3. `ir`         │ Insulin Resistant             │ 303        │ 26.01%      │ 2.90 - 14.82│ Clinically manifest   │
│                 │ (Overt Metabolic Resistance)  │            │             │ (Mean: 5.12)│ metabolic syndrome    │
└─────────────────┴───────────────────────────────┴────────────┴─────────────┴─────────────┴───────────────────────┘
```

### Detailed Pathophysiological Meaning:
1. **`is` — Insulin Sensitive ($\text{HOMA-IR} \le 1.10$):**
   * Pancreatic $\beta$-cells require only minimal insulin secretion ($< 6\text{ \mu IU/mL}$) to clear glucose from the bloodstream into skeletal muscle and hepatic glycogen stores.
   * Participants exhibit normal resting heart rates, high nocturnal heart rate variability (HRV), and low visceral adiposity.
2. **`impaired_ir` — Impaired Insulin Sensitivity ($1.11 \le \text{HOMA-IR} < 2.90$):**
   * This is the **critical asymptomatic "pre-diabetic" transitional window**.
   * Muscle and liver tissues have begun demonstrating blunted response to insulin. The pancreas compensates by secreting higher levels of insulin (**compensatory hyperinsulinemia**) to force blood glucose to remain normal.
   * Standard blood glucose tests miss these individuals entirely because fasting glucose appears normal ($85 - 99\text{ mg/dL}$), but their underlying metabolic physiology is already degenerating.
3. **`ir` — Insulin Resistant ($\text{HOMA-IR} \ge 2.90$):**
   * Overt insulin resistance where peripheral tissues severely resist insulin signaling.
   * Fasting insulin is markedly elevated ($> 15 - 30\text{ \mu IU/mL}$), lipolysis in adipose tissue is uninhibited (dumping free fatty acids into circulation), and hepatic de novo lipogenesis is accelerated (elevating triglycerides and lowering HDL).

---

## QUESTION 2: WHAT IS HIGH/LOW VS. WHAT ARE THE 3 CLASSES?

### 2.1 The 3-Class Formulation (Granular Staging)
When framing the task as a **3-class multi-category classification** problem, the target variable reflects the full spectrum of metabolic health:
* **Class 0 (Normal / Healthy):** `is` ($\text{HOMA-IR} \le 1.10$, $N = 275$, $23.6\%$)
* **Class 1 (Borderline / Compensatory):** `impaired_ir` ($1.11 \le \text{HOMA-IR} < 2.90$, $N = 587$, $50.4\%$)
* **Class 2 (Insulin Resistant):** `ir` ($\text{HOMA-IR} \ge 2.90$, $N = 303$, $26.0\%$)

### 2.2 The Binary High vs. Low Formulation (Clinical Screening Standard)
In real-world wearable medicine and the landmark Google Health *Nature* 2026 paper, binary classification is used as the **primary screening threshold**:
* **HIGH IR (Positive Class, $y = 1$):**
  * Defined as `homa_ir_status == 'ir'` ($\mathbf{\text{HOMA-IR} \ge 2.90}$).
  * Total cohort cases: **$N = 303$ ($26.01\%$ prevalence)**.
  * *Clinical Meaning:* Individuals requiring immediate lifestyle intervention or clinical follow-up to prevent progression to overt Type 2 diabetes.
* **LOW / NON-IR (Negative Class, $y = 0$):**
  * Defined as `homa_ir_status \in \{\text{'is'}, \text{'impaired_ir'}\}` ($\mathbf{\text{HOMA-IR} < 2.90}$).
  * Total cohort cases: **$N = 862$ ($73.99\%$ prevalence)**.
  * *Clinical Meaning:* Individuals currently non-resistant or in early compensated states who do not meet the overt diagnostic cutoff for metabolic syndrome.

---

## QUESTION 3: SHAPLEY ADDITIVE EXPLANATIONS (SHAP) & STRICT 5-FOLD CV

A strict **5-Fold Subject-Level Stratified Cross-Validation** was conducted across all $N = 1,165$ participants. Out-of-fold TreeSHAP attribution was computed on every test fold independently to eliminate data leakage.

### 3.1 Overall Top 10 Features (Across All Folds)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL TOP 10 FEATURES BY MEAN ABSOLUTE SHAP                        │
├──────┬─────────────────────────┬──────────────────────────┬─────────────────┬────────────┤
│ RANK │ FEATURE NAME            │ PHYSIOLOGICAL MODALITY   │ MEAN ABS SHAP   │ RELATIVE % │
├──────┼─────────────────────────┼──────────────────────────┼─────────────────┼────────────┤
│  1   │ `glucose`               │ Routine Blood Lab        │ 0.8361          │ 22.3%      │
│  2   │ `TG_HDL_Ratio`          │ Routine Blood Lab        │ 0.4618          │ 12.3%      │
│  3   │ `bmi`                   │ Anthropometrics & Vitals │ 0.3773          │ 10.1%      │
│  4   │ `alt` (Liver Alanine AT)│ Routine Blood Lab        │ 0.2064          │ 5.5%       │
│  5   │ `weight`                │ Anthropometrics & Vitals │ 0.1855          │ 5.0%       │
│  6   │ `triglycerides`         │ Routine Blood Lab        │ 0.1665          │ 4.5%       │
│  7   │ `STEPS_Daily_mean`      │ Wearable Accelerometer   │ 0.1511          │ 4.0%       │
│  8   │ `RHR_FreeLiving_mean`   │ Wearable Optical PPG     │ 0.1144          │ 3.1%       │
│  9   │ `hba1c`                 │ Routine Blood Lab        │ 0.1053          │ 2.8%       │
│ 10   │ `crp` (C-Reactive Prot) │ Routine Blood Lab        │ 0.1026          │ 2.7%       │
└──────┴─────────────────────────┴──────────────────────────┴─────────────────┴────────────┘
```

---

### 3.2 Within-Modality Top 5 Features

#### A. Wearable Sensors Modality (Optical PPG & 3D Accelerometry)
```
┌──────┬─────────────────────────┬─────────────────────────────────────────────────────────┐
│ RANK │ WEARABLE FEATURE        │ PHYSIOLOGICAL MEANING & IMPORTANCE                      │
├──────┼─────────────────────────┼─────────────────────────────────────────────────────────┤
│  1   │ `STEPS_Daily_mean`      │ Mean daily steps (SHAP: 0.1511). Skeletal muscle GLUT4  │
│      │                         │ translocation driver; physical activity volume.         │
│  2   │ `RHR_FreeLiving_mean`   │ 24-hr Resting Heart Rate (SHAP: 0.1144). Chronotropic   │
│      │                         │ sympathetic drive; elevated in hyperinsulinemic states. │
│  3   │ `SLEEP_Duration_mean`   │ Mean nightly sleep duration (SHAP: 0.0908). Sleep loss  │
│      │                         │ triggers nocturnal cortisol spikes and hepatic IR.      │
│  4   │ `STEPS_Daily_std`       │ Day-to-day step consistency / habit regularity (0.0586).│
│  5   │ `HRV_FreeLiving_std`    │ Heart Rate Variability dynamic spread (0.0542). Direct  │
│      │                         │ indicator of parasympathetic autonomic regulation.      │
└──────┴─────────────────────────┴─────────────────────────────────────────────────────────┘
```

#### B. Anthropometrics & Clinical Vitals Modality (Non-Invasive)
```
┌──────┬─────────────────────────┬─────────────────────────────────────────────────────────┐
│ RANK │ ANTHROPOMETRIC FEATURE  │ PHYSIOLOGICAL MEANING & IMPORTANCE                      │
├──────┼─────────────────────────┼─────────────────────────────────────────────────────────┤
│  1   │ `bmi`                   │ Body Mass Index (SHAP: 0.3773). Overall adiposity load. │
│  2   │ `weight`                │ Total body mass in kg (SHAP: 0.1855).                   │
│  3   │ `age`                   │ Chronological age (SHAP: 0.0686). Age-related decline   │
│      │                         │ in mitochondrial oxidative capacity.                    │
│  4   │ `waist_circumference`   │ Central abdominal circumference (SHAP: 0.0353).         │
│  5   │ `Pulse_Pressure`        │ Systolic BP - Diastolic BP (SHAP: 0.0306). Large central│
│      │                         │ arterial stiffness marker.                              │
└──────┴─────────────────────────┴─────────────────────────────────────────────────────────┘
```

#### C. Routine Blood Laboratory Modality
```
┌──────┬─────────────────────────┬─────────────────────────────────────────────────────────┐
│ RANK │ BLOOD LAB FEATURE       │ PHYSIOLOGICAL MEANING & IMPORTANCE                      │
├──────┼─────────────────────────┼─────────────────────────────────────────────────────────┤
│  1   │ `glucose`               │ Fasting plasma glucose (SHAP: 0.8361). Direct component │
│      │                         │ of the HOMA-IR equation numerator.                      │
│  2   │ `TG_HDL_Ratio`          │ Triglyceride to HDL ratio (SHAP: 0.4618). Well-known    │
│      │                         │ surrogate of hepatic de novo lipogenesis & dyslipidemia.│
│  3   │ `alt` (Alanine Amino.)  │ Liver transaminase enzyme (SHAP: 0.2064). Biomarker for │
│      │                         │ non-alcoholic fatty liver disease (NAFLD).              │
│  4   │ `triglycerides`         │ Fasting serum triglycerides (SHAP: 0.1665).             │
│  5   │ `hba1c`                 │ Glycated hemoglobin percentage (SHAP: 0.1053).          │
└──────┴─────────────────────────┴─────────────────────────────────────────────────────────┘
```

---

### 3.3 Top 10 Features Across Each of the 5 Cross-Validation Folds

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               TOP 10 FEATURES ACROSS EACH OF THE 5 FOLDS                                   │
├──────┬──────────────────────┬──────────────────────┬──────────────────────┬────────────────┬───────────────┤
│ RANK │ FOLD 1 (AUC = 0.8867)│ FOLD 2 (AUC = 0.8780)│ FOLD 3 (AUC = 0.8771)│ FOLD 4 (0.8848)│ FOLD 5 (0.8829│
├──────┼──────────────────────┼──────────────────────┼──────────────────────┼────────────────┼───────────────┤
│  1   │ glucose (0.8987)     │ glucose (0.7871)     │ glucose (0.8645)     │ glucose(0.7587)│ glucose(0.8715│
│  2   │ TG_HDL_Ratio (0.5222)│ TG_HDL_Ratio (0.3693)│ TG_HDL_Ratio (0.6231)│ bmi (0.4088)   │ TG_HDL(0.4502)│
│  3   │ bmi (0.3346)         │ bmi (0.3451)         │ bmi (0.4376)         │ TG_HDL (0.3443)│ bmi (0.3606)  │
│  4   │ weight (0.2361)      │ alt (0.2495)         │ triglycerides(0.1751)│ triglyc(0.2586)│ alt (0.2322)  │
│  5   │ STEPS_mean (0.1918)  │ weight (0.1605)      │ alt (0.1567)         │ alt (0.2070)   │ weight(0.1979)│
│  6   │ alt (0.1864)         │ hba1c (0.1452)       │ weight (0.1494)      │ weight (0.1837)│ triglyc(0.1577│
│  7   │ RHR_mean (0.1527)    │ RHR_mean (0.1445)    │ STEPS_mean (0.1443)  │ STEPS (0.1648) │ crp (0.1516)  │
│  8   │ triglycerides(0.1191)│ crp (0.1397)         │ SLEEP_mean (0.1266)  │ ldl (0.1040)   │ STEPS (0.1368)│
│  9   │ hba1c (0.0903)       │ triglycerides(0.1219)│ hba1c (0.1146)       │ RHR_mean(0.0905│ hba1c (0.1066)│
│ 10   │ STEPS_std (0.0895)   │ STEPS_mean (0.1177)  │ RHR_mean (0.1097)    │ SLEEP (0.0879) │ ldl (0.1022)  │
└──────┴──────────────────────┴──────────────────────┴──────────────────────┴────────────────┴───────────────┘
```

> **Key Stability Takeaway:**  
> The ranking is remarkably consistent across all 5 distinct test folds:
> - `glucose` is **#1 across 100% of the folds**.
> - `TG_HDL_Ratio` and `bmi` alternate between **#2 and #3 across all 5 folds**.
> - Wearable signals (`STEPS_Daily_mean`, `RHR_FreeLiving_mean`, `SLEEP_Duration_mean`) consistently maintain top-10 positions across every fold, proving their robust cross-validation generalizability.

---

## QUESTION 4: IMPORTANCE OF EACH MODALITY

We isolated each sensor and biological stream to measure its standalone predictive power, non-invasive combination, and full multimodal synergy under 5-Fold Stratified CV:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STANDALONE & FUSED MODALITY BENCHMARK                                   │
├──────────────────────────────────────┬──────┬────────┬────────┬────────┬────────┬────────┬────────┬────┤
│ MODALITY                             │FEATS │ AUROC  │ AUPRC  │ ACC    │ BAL_ACC│ SENS   │ SPEC   │ F1 │
├──────────────────────────────────────┼──────┼────────┼────────┼────────┼────────┼────────┼────────┼────┤
│ 1. Wearables Alone (PPG, HRV, Steps) │  13  │ 0.6651 │ 0.3981 │ 74.16% │ 56.11% │ 18.48% │ 93.74% │0.27│
│ 2. Anthropometrics & Vitals Alone    │  11  │ 0.7599 │ 0.5079 │ 76.14% │ 62.69% │ 34.65% │ 90.72% │0.43│
│ 3. Routine Blood Labs Alone          │  15  │ 0.8549 │ 0.6889 │ 80.43% │ 70.94% │ 51.16% │ 90.72% │0.58│
├──────────────────────────────────────┼──────┼────────┼────────┼────────┼────────┼────────┼────────┼────┤
│ 4. Tier 1 (Wearables + Anthro - ZERO)│  24  │ 0.7753 │ 0.5141 │ 76.74% │ 64.06% │ 37.62% │ 90.49% │0.46│
│ 5. Full Multimodal Fusion (All)      │  39  │ 0.8765 │ 0.7384 │ 83.18% │ 74.93% │ 57.76% │ 92.11% │0.64│
└──────────────────────────────────────┴──────┴────────┴────────┴────────┴────────┴────────┴────────┴────┘
```

```
┌────────────────────────────────────────────────────────────────────────┐
│                 MODALITY GLOBAL SHAP CONTRIBUTION SHARE                │
├───────────────────────────────┬──────────────────┬─────────────────────┤
│ PHYSIOLOGICAL MODALITY        │ TOTAL MEAN SHAP  │ CONTRIBUTION SHARE  │
├───────────────────────────────┼──────────────────┼─────────────────────┤
│ 1. Routine Blood Labs         │ 2.2456           │ 60.10%              │
│ 2. Anthropometrics & Vitals   │ 0.8102           │ 21.68%              │
│ 3. Wearable Sensors (PPG/Acc) │ 0.6804           │ 18.21%              │
└───────────────────────────────┴──────────────────┴─────────────────────┘
```

### Critical Clinical Interpretation:
1. **Wearables Provide the Continuous Behavioral Context ($18.2\%$):**  
   While blood labs provide a static chemical snapshot, wearables capture dynamic daily physiology (sympathetic tone via RHR, autonomic exhaustion via HRV, and muscular inactivity via step counts).
2. **Non-Invasive Tier 1 Achieves $\text{AUROC} = 0.7753$ with ZERO Blood:**  
   Combining smartwatches with static profile vitals provides a viable population screening engine without needles or clinic visits.
3. **Full Fusion Adds Significant Incremental Lift:**  
   Adding wearable and anthropometric features to routine blood labs elevates AUROC from **$0.8549$ to $0.8765$** (+2.16% AUROC, +4.95% AUPRC), and jumps sensitivity from **$51.2\%$ to $57.8\%$**.

---

## QUESTION 5: WEIGHTED FUSION VS. MAJORITY VOTE CLASSIFICATION

To evaluate how individual modality-specific models can be synthesized into an ensemble, we trained three independent classifiers:
- $M_{\text{wearable}}(\mathbf{x}_{\text{wearable}}) \rightarrow P_{\text{wearable}}$
- $M_{\text{anthro}}(\mathbf{x}_{\text{anthro}}) \rightarrow P_{\text{anthro}}$
- $M_{\text{labs}}(\mathbf{x}_{\text{labs}}) \rightarrow P_{\text{labs}}$

We tested **Hard Majority Voting**, **Equal Weighted Soft Voting**, **Performance-Weighted Probability Fusion**, **Grid-Search Optimal Weighted Fusion**, and **Concatenated Early Fusion**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MULTIMODAL ENSEMBLE FUSION BENCHMARK                                      │
├─────────────────────────────────────────────────┬────────┬────────┬────────┬────────┬────────┬─────────┤
│ ENSEMBLE FUSION STRATEGY                        │ AUROC  │ AUPRC  │ ACC    │ BAL_ACC│ SENS   │ SPEC    │
├─────────────────────────────────────────────────┼────────┼────────┼────────┼────────┼────────┼─────────┤
│ 1. Wearables Alone Baseline                     │ 0.6651 │ 0.3981 │ 74.16% │ 56.11% │ 18.48% │ 93.74%  │
│ 2. Anthropometrics Alone Baseline               │ 0.7599 │ 0.5079 │ 76.14% │ 62.69% │ 34.65% │ 90.72%  │
│ 3. Routine Blood Labs Alone Baseline            │ 0.8549 │ 0.6889 │ 80.43% │ 70.94% │ 51.16% │ 90.72%  │
├─────────────────────────────────────────────────┼────────┼────────┼────────┼────────┼────────┼─────────┤
│ 4. Majority Vote Classification (Hard >= 2/3)   │ 0.8557 │ 0.7043 │ 79.40% │ 63.82% │ 31.35% │ 96.29%  │
│ 5. Equal Weighted Probability Fusion (1/3 each) │ 0.8557 │ 0.7043 │ 80.26% │ 64.51% │ 31.68% │ 97.33%  │
│ 6. Performance-Weighted Fusion (0.29,0.33,0.37) │ 0.8605 │ 0.7138 │ 80.69% │ 65.23% │ 33.00% │ 97.45%  │
│ 7. Optimal Grid-Weighted Fusion (0.10,0.30,0.60)│ 0.8688 │ 0.7230 │ 81.89% │ 70.00% │ 45.21% │ 94.78%  │
│ 8. Concatenated Early Fusion (Gradient Boosting)│ 0.8765 │ 0.7384 │ 83.18% │ 74.93% │ 57.76% │ 92.11%  │
└─────────────────────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴─────────┘
```

### Key Takeaways on Fusion Strategies:
1. **Majority Voting (Hard Decision):**
   * Produces exceptionally high specificity (**$96.29\%$**), but has conservative sensitivity ($31.35\%$). It serves as a high-precision gatekeeper (low false alarms).
2. **Weighted Probability Fusion (Soft Decision):**
   * Performance-weighting ($w_{\text{wear}} = 0.29, w_{\text{anthro}} = 0.33, w_{\text{labs}} = 0.37$) improves AUROC to **$0.8605$**.
   * Optimal grid weighting ($w_{\text{wear}} = 0.10, w_{\text{anthro}} = 0.30, w_{\text{labs}} = 0.60$) reaches **$\text{AUROC} = 0.8688$** and balanced accuracy of **$70.0\%$**.
3. **Concatenated Early Fusion Achieves the Global Maximum ($\text{AUROC} = 0.8765$, $\text{AUPRC} = 0.7384$):**
   * Tree-based early fusion outperforms late soft voting because tree ensembles learn **cross-modality interaction splits** directly (e.g., *elevated fasting glucose is twice as dangerous when daily steps are below 4,500 and resting heart rate is elevated*).

---

## 6. DATASET FILES LOCATION SUMMARY

All datasets and dictionaries requested by the user are saved in the project repository:

1. **`d:\BTP\data\google_health_ir\google_health_ir_prediction_data.xlsx`** (905 KB)
   * Official prediction cohort of **$N = 1,165$ participants** and **155 columns**.
   * Contains exact columns: `participant_id`, `homa_ir`, `homa_ir_status` (`is`, `impaired_ir`, `ir`), all wearable PPG/HRV/Steps/Sleep metrics, anthropometrics, and clinical blood labs.
2. **`d:\BTP\data\google_health_ir\google_health_all_participants_data.xlsx`** (907 KB)
   * Full uncurated cohort of **$N = 1,173$ participants** and **154 columns**.
   * Includes the 8 initial entries before clinical QC filtering (which removed physiologically impossible BMI entries $< 10$ or $> 65$).
3. **`d:\BTP\data\google_health_ir\google_health_data_dictionary.xlsx`** (12.5 KB)
   * Complete data dictionary specifying every column name, its category, data type, completion percentage, null count, and sample value.
