# Research Paper Reference Document 02: Princeton University & NeuTigers (2025/2026)

## SweetDeep: A Wearable AI Solution for Real-Time Non-Invasive Diabetes Screening

* **Preprint Identification:** arXiv:2512.03471 [cs.LG / cs.AI / q-bio.QM]
* **Publication Date:** December 2025 / 2026
* **Academic Institution:** Princeton University, Department of Electrical and Computer Engineering, Princeton, NJ, USA
* **Industry Partner:** NeuTigers, Inc., Princeton, NJ, USA
* **Document Characteristics:** 12 Pages, 6 Figures, 4 Tables, 48 References
* **arXiv URL:** https://arxiv.org/abs/2512.03471

---

### Authors and Institutional Affiliations

* **Lead Academic Advisor:** **Prof. Niraj K. Jha** — Professor of Electrical and Computer Engineering, Princeton University (IEEE/ACM Fellow, Pioneer in Edge-AI and Biomedical Machine Learning)
* **Co-Authors & NeuTigers Research Team:**
  * Pierluigi Nuzzo (Princeton / NeuTigers)
  * Adel Laoui (CEO & Founder, NeuTigers, Inc.)
  * Additional clinical contributors from EU & MENA regional clinical trial sites.

---

### Abstract (Verbatim Academic Summary)

> **Abstract:**  
> Type 2 Diabetes (T2D) is a global epidemic affecting over 537 million adults, with nearly half remaining undiagnosed due to the invasive, cost-prohibitive, and episodic nature of current diagnostic methods (e.g., fasting plasma glucose, oral glucose tolerance tests, and glycated hemoglobin HbA1c assays). Non-invasive, continuous screening using consumer wearables represents a transformative paradigm for early intervention. However, previous AI approaches predominantly rely on controlled laboratory protocols, proprietary clinical devices, or computationally heavy deep neural networks that cannot be deployed locally on resource-constrained consumer smartwatches.  
>  
> In this paper, we introduce **SweetDeep**, an ultra-lightweight, edge-optimized wearable AI framework designed for real-time non-invasive diabetes screening using off-the-shelf consumer smartwatches (Samsung Galaxy Watch 7). Leveraging a decentralized clinical trial (DCT) methodology across diverse geographic cohorts in the European Union (EU) and Middle East & North Africa (MENA) regions ($N = 285$ participants: 162 non-diabetic and 123 with confirmed T2D), we collect multi-modal physiological sensor streams—including bioelectrical impedance analysis (BIA), photoplethysmography (PPG), electrocardiography (ECG), skin temperature, and triaxial accelerometry—over six days of free-living, naturalistic wear.  
>  
> The SweetDeep architecture comprises a compact neural network of **fewer than 3,000 parameters**, optimized for on-device inference with milliwatt-level energy consumption and zero cloud latency. Under rigorous 3-fold cross-validation:  
> 1. SweetDeep achieves **82.5% patient-level screening accuracy** with an **82.1% Macro F1-score**, **79.7% sensitivity**, and **84.6% specificity**.  
> 2. By introducing a post-training abstention ("Don't Know") adapter that withholds high-uncertainty predictions (<10% of cases), SweetDeep's effective accuracy increases to **84.5%**.  
> 3. SweetDeep statistically outperforms standard non-invasive clinical risk assessment tools, including the Finnish Diabetes Risk Score (FINDRISC), particularly in diagnostic specificity and false positive reduction.  
>  
> Our results demonstrate the feasibility of privacy-preserving, edge-native metabolic screening directly on consumer smartwatches without continuous cloud data transmission or invasive blood draws.

---

### Key Methodological Details

* **Sample Size ($N$):** $N = 285$ adult participants (162 Non-Diabetic controls, 123 Diagnosed T2D patients).
* **Wearable Device:** **Samsung Galaxy Watch 7** worn continuously for 6 days in naturalistic, free-living conditions.
* **Sensor Modalities:**
  * Bioelectrical Impedance Analysis (BIA): Body composition, fat percentage, skeletal muscle mass, total body water.
  * Photoplethysmography (PPG) & ECG: Resting pulse, heart rate variability (HRV), pulse wave velocity.
  * Triaxial Accelerometer: Physical activity epochs, sedentary intervals, sleep fragmentation.
  * Demographics & History: Age, biological sex, family history questionnaire.
* **Architecture:** Ultra-compact Deep Neural Network (<3,000 parameters) with quantization for edge execution on wearable microcontrollers.
* **Post-Processing Innovation:** Confidence-calibrated abstention mechanism to direct ambiguous cases for confirmatory medical follow-up.

---

### Comparison with Our BTP Capstone Project

| Dimension | Princeton SweetDeep (arXiv 2025/2026) | Our BTP Project (SARVAGYA-TIWARI) |
| :--- | :--- | :--- |
| **Cohort Size** | $N = 285$ participants | **$N = 3,296$ participants** (Over $11\times$ larger sample size!) |
| **Diagnostic Target** | Binary Classification (Non-Diabetic vs T2D) | **Continuous HOMA-IR Regression ($R^2 = 0.4044$)** AND **3-Class ADA Risk Screening (Balanced Acc = $54.85\%$, AUROC $= 0.738$)** |
| **Wearable Device** | Samsung Galaxy Watch 7 | CDC ActiGraph GT3X+ (Gold-standard triaxial accelerometer) |
| **Focus** | Edge deployment (<3,000 parameter NN) | **Circadian Rest-Activity Modeling, Single-Feature Ablation, 168h Deep Sequence Models, Tree SHAP Interpretability** |
| **Blood Access** | Zero blood access | **Zero blood access at inference time** |
