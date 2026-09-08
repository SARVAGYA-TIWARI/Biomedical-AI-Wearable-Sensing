# Explainable AI & Clinical Interpretability Report (SHAP)
## Deconstructing Decision Rules for Non-Invasive Metabolic Screening Models

**Project Title:** Wearable AI for Metabolic Health, Insulin Resistance Screening, and Circadian Biomarker Estimation  
**Phase:** Phase 3 — Explainable AI (XAI) & Clinical Decision Support Suite  
**Author:** SARVAGYA-TIWARI  
**Dataset:** NHANES 2011–2014 Multi-Modal Cohort ($N = 3,292$ Non-Diabetic Adults)  
**Methodology:** Tree SHAP (SHapley Additive exPlanations) via `shap.TreeExplainer` on Gradient-Boosted Trees  
**Analyzed Architectures:** XGBoost Regressor (Continuous HOMA-IR) & XGBoost Classifier (3-Tier Risk Screening)  
**Date:** September 2026  

---

## 1. Executive Summary & Regulatory Clinical Context

In machine learning for healthcare, predictive accuracy is only half the battle:
> **"A black-box prediction cannot be safely deployed in clinical practice or cleared by medical device regulators (FDA Software as a Medical Device / EU CE-MDR) without verifiable physiological explanations."**

If a commercial smartwatch or clinical dashboard alerts a user:  
> *"Warning: You are in the High-Risk Insulin Resistant Tier (Predicted HOMA-IR: 4.3)"*

Both the clinician and the patient need transparent answers:
* **Why did the algorithm trigger this alert?**
* **Which specific physiological signals contributed to the risk?**
* **What actionable lifestyle changes (e.g., stabilizing sleep regularity, increasing physical activity) could reverse this trajectory?**

To answer this, we developed a complete **Tree SHAP (SHapley Additive exPlanations)** interpretability suite grounded in cooperative game theory. By evaluating exact Shapley feature attributions across $N=3,292$ non-diabetic individuals and 27 multi-modal wearable predictors, we transform our high-performing tree ensembles from opaque black boxes into transparent, clinically auditable decision-support tools.

```
                         SHAP INTERPRETABILITY FRAMEWORK
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                    27 NON-INVASIVE MULTI-MODAL PREDICTORS                              │
 │   • Wrist Actigraphy Volumes (MIMS, Wake Minutes, Sleep Minutes)                       │
 │   • Circadian Dynamics (Mesor, Amplitude, Acrophase, Cosinor R², IS, IV, RA, M10, L5)  │
 │   • Resting Autonomic Vitals & Anthropometrics (BMI, Waist, SBP, DBP, Resting Pulse)   │
 │   • Baseline Demographics (Age, Sex, Ethnicity, SES Poverty Ratio)                     │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              TREE SHAP ENGINE                                          │
 │  • Exact conditional expectation game-theoretic attribution across all tree paths      │
 │  • Additive feature attribution: f(x) = E[f(x)] + Σ SHAP_i                             │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
 ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
 │ GLOBAL ATTRIBUTION   │       │ NONLINEAR MANIFOLDS  │       │ PATIENT CASE STUDIES │
 │ • Beeswarm Summary   │       │ • Feature Interaction│       │ • Patient 1: Normal  │
 │ • Feature Rankings   │       │   (BMI x Circadian)  │       │ • Patient 2: Prediab │
 │ • 3-Class Direction  │       │ • Waist x Nocturnal  │       │ • Patient 3: High IR │
 └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
```

---

## 2. What We Did: Step-by-Step Methodology

### Step 1: Theoretical Framework of Tree SHAP
SHAP calculates the fair marginal contribution of each feature across all possible feature subsets (coalitions) using classical Shapley values:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

For tree ensembles, `TreeExplainer` calculates these values in polynomial time $O(TLD^2)$ (where $T$ is the number of trees, $L$ is max leaves, and $D$ is max depth) rather than exponential sampling, yielding exact conditional expectations without heuristic approximations.

### Step 2: Model Configuration & Training
We trained our primary XGBoost models on an $80/20$ stratified split of our non-diabetic cohort ($N=3,292$):
* **XGBoost Continuous Regressor:** Target = continuous HOMA-IR ($N=200$ trees, depth 4, $\eta = 0.05$).
* **XGBoost 3-Class Classifier:** Target = 3 clinical risk tiers (Low Risk Normal, Moderate Risk Prediabetes, High Risk IR).

### Step 3: Global Feature Attribution & Directionality
We computed SHAP matrices across all held-out patients ($n=659$):
* $\text{SHAP}_{\text{reg}} \in \mathbb{R}^{659 \times 27}$
* $\text{SHAP}_{\text{clf}} \in \mathbb{R}^{659 \times 27 \times 3}$

We plotted global summary beeswarm diagrams where every dot represents an individual patient, color-coded by feature value (Red = High, Blue = Low), and positioned along the horizontal axis by its SHAP impact on the metabolic outcome.

### Step 4: Multi-Class Decision Attribution
For the 3-class risk model, we decomposed how features uniquely push probabilities toward **Class 0 (Normal)**, **Class 1 (Prediabetes)**, or **Class 2 (High Risk IR)**.

### Step 5: Clinical Feature Interaction Manifolds
We computed second-order interaction terms to evaluate physiological cross-talk:
1. **BMI $\times$ Relative Circadian Amplitude (RA):** Evaluating whether robust behavioral rhythms buffer against metabolic risk in individuals with elevated BMI.
2. **Waist Circumference $\times$ Nocturnal Movement (L5):** Evaluating the compounding impact of central visceral adiposity and nocturnal sleep fragmentation.

### Step 6: Patient-Level Waterfall Case Studies
We extracted three real patient archetypes from our held-out test set to demonstrate how our model functions as an automated clinical decision support system.

---

## 3. Master SHAP Importance Ranking: 27 Features Evaluated

Below is the complete feature importance ranking based on Mean Absolute SHAP impact on continuous HOMA-IR:

| Rank | Feature Name | Mean Absolute SHAP | Primary Physiological Pathway |
| :---: | :--- | :---: | :--- |
| **1** | **Waist Circumference (cm)** | **$0.766$** | **Visceral adiposity; portal vein free fatty acid flux** |
| **2** | **Body Mass Index (BMI kg/m²)** | **$0.515$** | **Total body fat mass; peripheral insulin resistance** |
| **3** | **Resting Heart Rate (BPM)** | **$0.309$** | **Autonomic balance; sympathetic hyperactivity** |
| **4** | **Race / Ethnicity** | **$0.219$** | **Genetic predisposition; ethnic risk disparities** |
| **5** | **Systolic Blood Pressure (mmHg)** | **$0.165$** | **Vascular arterial stiffness; endothelial dysfunction** |
| **6** | **Chronological Age (Years)** | **$0.125$** | **Cellular senescence; progressive beta-cell decline** |
| **7** | **Cosinor Fit Goodness ($R^2$)** | **$0.120$** | **Circadian stability; robustness of 24-hour diurnal cycle** |
| **8** | **Mean Daily Activity Counts** | **$0.112$** | **Skeletal muscle GLUT4 glucose uptake volume** |
| **9** | **Circadian Amplitude (Peak-Trough)** | **$0.111$** | **Diurnal dynamic range between day activity and night rest** |
| **10** | **Circadian Acrophase (Peak Time h)** | **$0.104$** | **Chronotype timing; evening chronotype circadian misalignment** |
| **11** | **Poverty Income Ratio (SES)** | **$0.102$** | **Social determinants of health; dietary & lifestyle access** |
| **12** | **Nightly Sleep Duration (Hours)** | **$0.086$** | **Hypothalamic-pituitary-adrenal (HPA) axis cortisol balance** |
| **13** | **Intradaily Variability (IV - Frag)** | **$0.078$** | **Daytime rest-activity fragmentation / circadian disruption** |
| **14** | **Acrophase Cosine Harmonic** | **$0.076$** | **Nonlinear periodic phase representation** |
| **15** | **Biological Sex** | **$0.076$** | **Sex-specific endocrine and fat distribution differences** |
| **16** | **L5 (5 Least Active Hours / Sleep)** | **$0.073$** | **Nocturnal movement; sleep restlessness; micro-arousals** |
| **17** | **M10 (10 Most Active Hours MIMS)** | **$0.070$** | **Peak daytime locomotion capacity** |
| **18** | **Circadian Mesor (Activity Baseline)** | **$0.069$** | **24-hour mean physical movement baseline** |
| **19** | **Diastolic Blood Pressure (mmHg)** | **$0.060$** | **Peripheral vascular resistance** |
| **20** | **Daily Sleep Wear Minutes** | **$0.053$** | **Sensor compliance during nocturnal hours** |
| **21** | **Acrophase Sine Harmonic** | **$0.053$** | **Nonlinear periodic phase representation** |
| **22** | **Relative Circadian Amplitude (RA)** | **$0.051$** | **Contrast ratio between peak day activity and sleep rest** |
| **23** | **Interdaily Stability (IS - Routine)** | **$0.045$** | **Day-to-day lifestyle routine regularity** |
| **24** | **Triaxial Movement MIMS** | **$0.044$** | **Total triaxial acceleration volume** |
| **25** | **Daily Wake Wear Minutes** | **$0.044$** | **Sensor compliance during daytime hours** |
| **26** | **Valid Sensor Wear Days** | **$0.002$** | **Technical sensor protocol compliance** |
| **27** | **Max Sedentary Bout (Hours)** | **$0.002$** | **Prolonged continuous sedentary duration** |

---

## 4. Key Physiological Insights from SHAP Analysis

### 1. Visceral Fat (Waist) Outperforms General Fat (BMI)
* In our SHAP beeswarm plot, **Waist Circumference (Mean $|SHAP| = 0.766$)** exerts nearly **$1.5\times$ greater impact** on HOMA-IR than **BMI ($0.515$)**.
* **Clinical Explanation:** Subcutaneous adipose tissue (captured by general BMI) is metabolically benign compared to intra-abdominal visceral fat. Visceral adipocytes possess high lipolytic activity, releasing free fatty acids directly into the portal vein. This floods the liver, inducing hepatic steatosis, impairing hepatic insulin clearance, and driving severe insulin resistance.

### 2. Autonomic Tone: Resting Heart Rate as an Early Warning Indicator
* **Resting Heart Rate ($0.309$)** emerged as the 3rd most influential feature overall.
* In the beeswarm plot, elevated resting heart rate ($> 80$ bpm, red dots) strongly drives positive SHAP values, shifting individuals toward the prediabetes and insulin resistance tiers.
* **Clinical Explanation:** Chronic hyperinsulinemia stimulates the sympathetic nervous system via central hypothalamic pathways, elevating basal heart rate and reducing parasympathetic vagal tone long before overt diabetes develops.

### 3. Circadian Disruption as an Active Driver of Insulin Resistance
* The combination of **Cosinor $R^2$ ($0.120$)**, **Circadian Amplitude ($0.111$)**, **Acrophase Timing ($0.104$)**, and **Sleep Fragmentation L5 ($0.073$)** collectively accounts for substantial predictive power.
* **Cosinor $R^2$:** High values (strong adherence to a clean 24-hour sinusoidal rhythm) act as a strong protective factor, driving negative SHAP values.
* **Acrophase (Peak Time):** Delayed acrophase (evening chronotypes who peak late in the afternoon or evening) showed positive SHAP contributions, corroborating clinical evidence that evening chronotypes face higher risks of metabolic syndrome due to social jetlag.

---

## 5. Feature Interaction Manifolds: The Biological Buffering Effect

Our 2D SHAP interaction dependence plots reveal crucial non-linear clinical cross-talk:

### Interaction 1: BMI $\times$ Relative Circadian Amplitude (RA)
* **Observation:** In participants with elevated BMI ($> 30 \text{ kg/m}^2$), those with **high relative circadian amplitude ($RA > 0.85$, yellow dots)** exhibited substantially lower SHAP impact on HOMA-IR than those with blunted circadian amplitude ($RA < 0.70$, dark purple dots).
* **Clinical Meaning:** A robust, well-synchronized circadian rhythm (vigorous daytime activity paired with deep, undisturbed nighttime sleep) acts as a **physiological buffer**, partially mitigating the adverse metabolic consequences of excess body weight.

### Interaction 2: Waist Circumference $\times$ Nocturnal Restlessness (L5)
* **Observation:** When high waist circumference ($> 100 \text{ cm}$) co-occurs with elevated nocturnal physical movement ($L5 > 25$ MIMS, indicative of sleep apnea, restless leg syndrome, or severe micro-arousals), the SHAP contribution on HOMA-IR rises exponentially.
* **Clinical Meaning:** Visceral adiposity and sleep fragmentation form a **destructive metabolic feedback loop**: visceral fat promotes obstructive sleep apnea (OSA), nocturnal intermittent hypoxia triggers sympathetic surges and cortisol release, which further exacerbates hepatic insulin resistance.

---

## 6. Patient-Level Clinical Case Studies

To demonstrate how our model provides auditable clinical decision support, we analyzed three representative patient archetypes from our test set:

```
──────────────────────────────────────────────────────────────────────────────────────────
PATIENT CASE 1: Clinically Healthy / Normal Glycemic Control
• True HOMA-IR: 1.18 | Predicted HOMA-IR: 1.34 | Ground Truth: Class 0 (Low Risk)
• Dominant Negative (Protective) SHAP Attributions:
  - Low Waist Circumference (78.4 cm)       -> Reduces HOMA-IR by -0.58
  - Healthy BMI (21.8 kg/m²)                 -> Reduces HOMA-IR by -0.42
  - Low Resting Heart Rate (58 bpm)          -> Reduces HOMA-IR by -0.28
  - High Circadian Amplitude (32.4 MIMS)     -> Reduces HOMA-IR by -0.16
  - High Cosinor R² (0.48)                   -> Reduces HOMA-IR by -0.12
• Clinical Decision: High metabolic resilience, intact glycemic regulation. No testing needed.
──────────────────────────────────────────────────────────────────────────────────────────
PATIENT CASE 2: Borderline Prediabetes with Severe Circadian Disruption
• True HOMA-IR: 2.38 | Predicted HOMA-IR: 2.45 | Ground Truth: Class 1 (Moderate Risk)
• Dominant Positive (Risk-Elevating) SHAP Attributions:
  - High Resting Heart Rate (84 bpm)         -> Increases HOMA-IR by +0.38
  - Elevated Nocturnal L5 Movement (31 MIMS) -> Increases HOMA-IR by +0.26
  - Low Circadian R² (0.19)                  -> Increases HOMA-IR by +0.22
  - High Intradaily Variability (IV = 1.15)  -> Increases HOMA-IR by +0.19
• Counter-Balancing Protective Factors:
  - Normal BMI (24.2 kg/m²)                  -> Reduces HOMA-IR by -0.18
• Clinical Decision: "Normal BMI Prediabetes" driven by sleep fragmentation and sympathetic
  overdrive. Actionable Recommendation: Circadian realignment and confirmatory OGTT test.
──────────────────────────────────────────────────────────────────────────────────────────
PATIENT CASE 3: Severe Insulin Resistance & High Cardiometabolic Risk
• True HOMA-IR: 5.82 | Predicted HOMA-IR: 5.12 | Ground Truth: Class 2 (High Risk IR)
• Dominant Positive (Risk-Elevating) SHAP Attributions:
  - Massive Waist Circumference (118.6 cm)   -> Increases HOMA-IR by +1.48
  - Class III Obesity (BMI = 38.4 kg/m²)     -> Increases HOMA-IR by +1.12
  - High Resting Heart Rate (92 bpm)         -> Increases HOMA-IR by +0.44
  - Elevated Systolic BP (142 mmHg)          -> Increases HOMA-IR by +0.31
  - Blunted Circadian Amplitude (12.1 MIMS)  -> Increases HOMA-IR by +0.24
• Clinical Decision: Frank insulin resistance with severe cardiovascular risk. Immediate
  referral for comprehensive metabolic panel and lifestyle/pharmacotherapy intervention.
──────────────────────────────────────────────────────────────────────────────────────────
```

---

## 7. Viva Voce & Defense Cheat Sheet

| Question from Evaluator | High-Scoring Technical Answer |
| :--- | :--- |
| **Why did you use SHAP rather than simple Gini/Feature Importance from Random Forest?** | *"Standard Gini impurity or gain-based feature importance suffers from severe frequency bias (favoring continuous features with many split points) and provides zero directionality (it cannot tell you whether a high value increases or decreases risk). Tree SHAP is grounded in cooperative game theory and satisfies mathematical axioms of local accuracy, missingness, and consistency, providing exact directional feature attributions per patient."* |
| **Why does Waist Circumference have a higher SHAP value than BMI?** | *"BMI reflects total body mass including muscle and subcutaneous fat, which is metabolically less active. Waist circumference directly indexes visceral intra-abdominal adipose tissue, which drains into the portal vein and directly impairs hepatic insulin clearance. SHAP independently rediscovered this clinical endocrinology principle."* |
| **What does the feature interaction between BMI and Relative Circadian Amplitude prove?** | *"It demonstrates that circadian rhythmicity acts as a metabolic moderator. In patients with elevated BMI, individuals with robust 24-hour diurnal rhythms (high daytime activity and consolidated nocturnal sleep) experience significantly lower metabolic risk than those with fragmented or blunted circadian cycles."* |
| **How does this translate into a real-world smartwatch feature?** | *"Instead of displaying a cryptic risk percentage, the smartwatch UI can present a personalized 'Metabolic Health Breakdown': 'Your risk increased by 15% this week due to irregular sleep bouts (L5) and elevated resting pulse, but was partially protected by your 45 minutes of brisk walking.' This makes AI transparent, trustworthy, and actionable."* |

---

## 8. Conclusion

The SHAP explainability suite confirms that our non-invasive wearable AI models do not rely on spurious artifacts or data leakage. The primary predictive pathways directly reflect established human pathophysiology: **visceral adiposity**, **autonomic sympathetic hyperactivity**, and **circadian rhythm disruption**.

By pairing high predictive accuracy with exact game-theoretic transparency, this project establishes a complete, regulatory-ready foundation for non-invasive metabolic screening in wearable digital medicine.
