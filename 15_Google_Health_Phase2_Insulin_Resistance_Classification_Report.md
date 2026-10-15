# Google Health Benchmark — Phase 2: Insulin Resistance Risk Stratification & Classification
## Triage Screening at the Nature 2026 Primary Threshold (HOMA-IR ≥ 2.9) & High-Risk Subgroup Analysis

**Project Title:** Wearable AI for Early Metabolic Risk Screening & Continuous HOMA-IR Estimation  
**Dataset:** Google Health WEAR-ME Cohort ($N = 1,165$ Multi-Ethnic Adults)  
**Author:** SARVAGYA-TIWARI  
**Date:** October 2026  

---

# 1. Executive Summary & Core Results

In Phase 2, we implemented the discrete clinical screening and risk stratification suite evaluated in the *Nature* 2026 paper. We evaluated whether smartwatches and clinical vitals can accurately flag individuals with clinically significant Insulin Resistance (**$\text{HOMA-IR} \ge 2.9$**), the primary cutoff established in endocrinology for initiating preventative lifestyle or pharmacological intervention.

Across strict 5-Fold Stratified Cross-Validation on all $N = 1,165$ participants:
* **Wearables + Anthropometrics & Vitals (Zero Blood Access):**
  * **Logistic Regression:** Achieved **$\text{AUROC} = \mathbf{0.7942}$**, $\text{AUPRC} = \mathbf{0.5841}$, **Sensitivity $= 70.6\%$**, **Specificity $= 75.1\%$**, and **Balanced Accuracy $= 72.8\%$**!
  * **XGBoost:** Achieved **$\text{AUROC} = \mathbf{0.7781}$**, $\text{AUPRC} = \mathbf{0.5254}$, **Sensitivity $= 66.0\%$**, and **Specificity $= 78.5\%$**.
  * *Clinical Takeaway:* Matches published literature in *Nature Digital Medicine* showing that smartwatches + non-invasive body vitals reach $\text{AUROC} \approx 0.78 - 0.80$ without requiring blood draws!
* **Full Multimodal Fusion (Wearables + Anthropometrics + Routine Blood Labs):**
  * **LightGBM:** Achieved **$\text{AUROC} = \mathbf{0.8780}$**, $\text{AUPRC} = \mathbf{0.7398}$, **Sensitivity $= 73.9\%$**, **Specificity $= 84.6\%$**, and **Balanced Accuracy $= 79.2\%$**!
  * **XGBoost:** Achieved **$\text{AUROC} = \mathbf{0.8760}$**, $\text{AUPRC} = \mathbf{0.7396}$, **Sensitivity $= 74.3\%$**, **Specificity $= 85.0\%$**, and **Balanced Accuracy $= 79.6\%$**!
  * **Logistic Regression:** Achieved **$\text{AUROC} = \mathbf{0.8783}$**, $\text{AUPRC} = \mathbf{0.7451}$, **Sensitivity $= 76.6\%$**, and **Specificity $= 82.4\%$**!
* **High-Risk Subgroup Performance (Obese $\text{BMI} \ge 30$ & Sedentary, $N = 531$):**
  * When evaluated in the high-risk cohort (matching the Nature paper's subgroup analysis), XGBoost sensitivity jumped to **$89.0\%$** with an **$\text{AUROC} = \mathbf{0.8429}$**!

---

# 2. What We Did: Clinical Problem Formulation

```
                        PHASE 2 CLASSIFICATION PIPELINE
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ CLINICAL TARGET: BINARY INSULIN RESISTANCE (HOMA-IR >= 2.9)                            │
 │ • Negative Class (Low / Normal IR): HOMA-IR < 2.9  (N = 862, 74.0%)                    │
 │ • Positive Class (High Insulin Resistance): HOMA-IR >= 2.9  (N = 303, 26.0%)           │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 5-FOLD STRATIFIED CROSS-VALIDATION                                                     │
 │ • Exact class proportions preserved across all folds                                   │
 │ • Class Imbalance Management via scale_pos_weight = 2.85 & Balanced Cost-Sensitivity   │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 4 COMPETING ARCHITECTURES                                                              │
 │ • L2-Regularized Logistic Regression | Balanced Random Forest | LightGBM | XGBoost     │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ CLINICAL EVALUATION SUITE                                                              │
 │ • AUROC | AUPRC (Average Precision) | Sensitivity | Specificity | Balanced Accuracy    │
 │ • Confusion Matrices | ROC Curves | Precision-Recall Curves | High-Risk Subgroup Test  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Comprehensive Phase 2 Benchmark Results

The complete classification benchmark results evaluated across all 5 folds are shown below:

| Input Modality | Classifier | AUROC | AUPRC | Sensitivity | Specificity | Balanced Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wearables Alone** | Logistic Regression | $0.6965$ | $0.4382$ | $66.0\%$ | $63.6\%$ | $64.8\%$ | $0.6271$ |
| (PPG RHR, HRV, Steps, Sleep) | Random Forest | $0.6743$ | $0.3949$ | $46.9\%$ | $77.0\%$ | $61.9\%$ | $0.6053$ |
| | LightGBM | $0.6638$ | $0.3899$ | $52.8\%$ | $70.1\%$ | $61.4\%$ | $0.5996$ |
| | XGBoost | $0.6637$ | $0.3936$ | $53.1\%$ | $71.7\%$ | $62.4\%$ | $0.6084$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wearables + Anthro & Vitals** | **Logistic Regression** | **$\mathbf{0.7942}$** | **$\mathbf{0.5841}$** | **$70.6\%$** | **$75.1\%$** | **$72.8\%$** | **$0.7099$** |
| **(Strict Zero Blood Access)** | Random Forest | $0.7790$ | $0.5280$ | $60.4\%$ | $78.8\%$ | $69.6\%$ | $0.6865$ |
| | LightGBM | $0.7807$ | $0.5164$ | $67.3\%$ | $77.0\%$ | $72.2\%$ | $0.7011$ |
| | XGBoost | $0.7781$ | $0.5254$ | $66.0\%$ | $78.5\%$ | $72.3\%$ | $0.7030$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Routine Blood Labs Alone** | Logistic Regression | $0.8642$ | $0.7020$ | $77.6\%$ | $79.6\%$ | $78.6\%$ | $0.7622$ |
| (Lipids, Glucose, HbA1c) | Random Forest | $0.8523$ | $0.6790$ | $64.0\%$ | $85.4\%$ | $74.7\%$ | $0.7412$ |
| | LightGBM | $0.8529$ | $0.6887$ | $70.3\%$ | $80.9\%$ | $75.6\%$ | $0.7431$ |
| | XGBoost | $0.8516$ | $0.6745$ | $70.3\%$ | $81.7\%$ | $76.0\%$ | $0.7411$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Multimodal Fusion** | **Logistic Regression** | **$\mathbf{0.8783}$** | **$\mathbf{0.7451}$** | **$76.6\%$** | **$82.4\%$** | **$79.5\%$** | **$0.7758$** |
| **(Wearables + Anthro + Labs)** | Random Forest | $0.8715$ | $0.7364$ | $67.0\%$ | $88.6\%$ | $77.8\%$ | $0.7709$ |
| | **LightGBM** | **$\mathbf{0.8780}$** | **$\mathbf{0.7398}$** | **$73.9\%$** | **$84.6\%$** | **$79.2\%$** | **$0.7800$** |
| | **XGBoost** | **$\mathbf{0.8760}$** | **$\mathbf{0.7396}$** | **$74.3\%$** | **$85.0\%$** | **$79.6\%$** | **$0.7806$** |

---

# 4. High-Risk Subgroup Validation (Obese & Sedentary Cohort)

A major clinical contribution of the *Nature* paper is demonstrating that wearable AI achieves its highest sensitivity in the sub-population that needs it most: individuals who are both **obese ($\text{BMI} \ge 30$) and sedentary ($\le 5,000$ steps/day)**.

In our replication on the $N = 531$ high-risk participants:
* **AUROC:** **$0.8429$**
* **Sensitivity (Detection Rate of High IR):** **$89.0\%$** (identifies $\sim 9$ out of every 10 patients with severe metabolic dysfunction!).
* **Specificity:** **$60.7\%$**
* **Clinical Significance:** Confirms that in individuals with elevated metabolic risk, combining passive wrist telemetry with clinical vitals provides an effective early-warning triage system.

---

# 5. Key Takeaways & Conclusions for Your Professor

1. **Successful Replication of Nature AUROC Benchmarks:** The *Nature* paper reported AUROC values between $0.80$ and $0.88$ for multimodal models. Our models independently achieve **$\text{AUROC} = 0.876 - 0.878$**, replicating their published classification performance.
2. **True Smartwatch Feasibility (AUROC $= 0.794$):** When laboratory blood tests are excluded (evaluating only smartwatches and body measures), the model achieves **$\text{AUROC} = 0.794$ with $70.6\%$ sensitivity and $75.1\%$ specificity**, matching our NHANES Phase 2 findings.
3. **High-Risk Sensitivity ($89.0\%$):** The algorithm successfully detects nearly $90\%$ of high-risk insulin-resistant individuals in obese and sedentary cohorts, demonstrating high clinical utility for diabetes prevention programs.

---

# Generated Artifacts & Figures
* **Benchmark Results CSV:** [results/google_health_phase2_classification_benchmark.csv](file:///d:/BTP/results/google_health_phase2_classification_benchmark.csv)
* **Figure 1 (ROC and Precision-Recall Curves):** [figures/google_health_phase2_roc_pr_curves.png](file:///d:/BTP/figures/google_health_phase2_roc_pr_curves.png)
* **Figure 2 (Normalized Confusion Matrix):** [figures/google_health_phase2_confusion_matrix.png](file:///d:/BTP/figures/google_health_phase2_confusion_matrix.png)
