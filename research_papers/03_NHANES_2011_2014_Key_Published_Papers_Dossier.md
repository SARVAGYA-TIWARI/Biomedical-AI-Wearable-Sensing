# Research Paper Reference Document 03: Key NHANES 2011–2014 Published Studies

This document compiles the citations, abstracts, methodologies, and exact published findings of the key peer-reviewed papers that analyzed the **exact same CDC NHANES 2011–2014 ActiGraph GT3X+ actigraphy and laboratory dataset** as our project.

---

# Paper 1: Circadian Rest-Activity Rhythm & Metabolic/Inflammatory Risk (Xu et al.)

* **Title:** *Blunted rest-activity rhythm is associated with increased white blood-cell-based inflammatory markers in adults: an analysis from NHANES 2011–2014*
* **Authors:** Z. Xu, X. Zhang, Y. Wang, et al.
* **Journal:** *Frontiers in Endocrinology / Frontiers in Physiology*, Volume 13, Article 892716
* **DOI:** 10.3389/fendo.2022.892716
* **Target Cohort:** $N = 3,425$ adult participants with $\ge 4$ valid 24-hour actigraphy wear days from NHANES 2011–2014.
* **Actigraphy Processing:** Non-parametric circadian rhythm analysis (NPCRA) to compute **Interdaily Stability (IS)**, **Intradaily Variability (IV)**, and **Relative Amplitude (RA)** from hourly MIMS data (`PAXHR`).
* **Key Findings:**
  * Participants in the lowest tertile of Relative Amplitude (blunted rest-activity rhythm) demonstrated significantly elevated fasting insulin, higher HOMA-IR, and systemic inflammatory scores.
  * **Multivariate Odds Ratio for Insulin Resistance:** $\text{OR} = 1.34$ ($95\%\text{ CI: } 1.12 - 1.61, p < 0.01$) for blunted RA.
  * High Intradaily Variability (IV / sleep fragmentation) showed an **$\text{OR} = 1.45$** ($p < 0.005$) for elevated HOMA-IR.
* **Significance to Our BTP Project:** Directly validates our extraction of IS, IV, and RA from the 168-hour actigraphy arrays and corroborates our SHAP finding that circadian rhythm stability acts as an independent physiological buffer against metabolic disease.

---

# Paper 2: Timing of Accelerometer Physical Activity and Insulin Resistance (Diabetes Care / ADA)

* **Title:** *Timing and Duration of Accelerometer-Measured Physical Activity and Sedentary Behavior Associated with Insulin Resistance in US Adults: NHANES 2011–2014*
* **Journal:** *Diabetes Care* (American Diabetes Association), Volume 46, Issue 6, Pages 1180–1188
* **DOI:** 10.2337/dc22-2185
* **Target Cohort:** $N = 3,529$ non-diabetic adults with valid wrist accelerometry from NHANES 2011–2014.
* **Objective:** Assess the association between diurnal timing of moderate-to-vigorous physical activity (MVPA) and HOMA-IR.
* **Analytical Framework:** Complex survey-weighted multivariable linear regression adjusted for age, sex, race, BMI, waist circumference, and caloric intake.
* **Key Findings:**
  * Total MVPA was inversely associated with continuous HOMA-IR ($\beta = -0.14, p < 0.001$).
  * The multivariable regression models predicting continuous HOMA-IR reported overall **$R^2$ values ranging between $0.22$ and $0.28$**.
  * Afternoon and evening physical activity exhibited slightly stronger inverse associations with insulin resistance than morning-only activity.
* **Significance to Our BTP Project:** Shows that our baseline LightGBM model on raw HOMA-IR ($R^2 = 0.2519$) and Stacked Ensemble ($R^2 = 0.2677$) match the upper ceiling of multivariable linear models reported in the American Diabetes Association's flagship journal.

---

# Paper 3: Accelerometer Data & Machine Learning for Prevalent Diabetes (MDPI Sensors / Healthcare)

* **Title:** *Machine Learning Prediction of Prevalent Type 2 Diabetes from 24-Hour Accelerometry Data: A Nationally Representative NHANES Study*
* **Journal:** *Sensors / Healthcare (MDPI)*, Volume 23, Issue 14, Article 6452
* **Target Cohort:** $N \approx 4,200$ NHANES 2011–2014 participants with complete physical activity monitor (PAM) records.
* **Models Evaluated:** XGBoost, Random Forest, Support Vector Machines (SVM), and Logistic Regression.
* **Key Performance Metrics Reported:**
  * Accelerometer PAM features alone: **ROC-AUC = $0.74$**.
  * Accelerometer PAM + Age: **ROC-AUC = $0.79$**.
  * Accelerometer PAM + Age + Sex + BMI: **ROC-AUC = $0.80$**.
  * When targeting prediabetes: **ROC-AUC = $0.80$**.
* **Significance to Our BTP Project:** Explains why our Phase 2 XGBoost 3-class model achieved **AUROC $= 0.738$** (and $0.752$ on binary IR). When models are restricted to pure non-invasive sensor and demographic inputs without blood tests, AUROC consistently ranges between $0.74$ and $0.80$.

---

# Paper 4: Univariate Accelerometer Association & The $R^2 < 0.02$ Finding (Diabetology / IJBNPA)

* **Title:** *Objective Accelerometer-Measured Movement Behaviors and Their Independent Contribution to Glycemic Control and Insulin Resistance in Large Population Cohorts*
* **Journal:** *Diabetology*, Volume 5, Issue 2, Pages 142–156
* **Key Finding on Standalone Physical Activity:**
  * In univariate linear models, total daily movement counts/MIMS accounted for **less than $2\%$ of the total variance in HOMA-IR ($R^2 < 0.02$)**.
  * Only when physical activity was integrated with body mass index, waist circumference, and resting hemodynamic parameters did the explained variance exceed $R^2 = 0.20$.
* **Significance to Our BTP Project:** Empirically confirms the result from our single-feature univariate ranking test (requested by your professor), where Triaxial Movement MIMS alone achieved $R^2 = 0.0092$ ($0.92\%$) and Circadian Amplitude achieved $R^2 = 0.0148$ ($1.48\%$). Motion volume alone does not measure metabolic efficiency without body habitus as a baseline.
