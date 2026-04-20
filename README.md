# 13 — Employee Turnover Prediction · Self-Training (Semi-Supervised)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/SelfT_HR_Turnover_Prediction/blob/main/13_SelfT_HR_Turnover_Prediction.ipynb)

> *"Inference is the act of drawing conclusions from incomplete evidence. That is not a workaround. That is what reasoning is."*

---

## 🎯 Business Problem

Here is the question HR asks every quarter: *who is about to leave?*

Here is the data they have to answer it: 4,500 employee profiles — age, salary, satisfaction scores, workload, overtime hours. Solid foundation. Except that only **500 of those 4,500 people have a confirmed outcome**. The other 4,000 are still employed. We know their conditions, but not their future.

This is not a data quality problem. This is the **structural reality of any live HR dataset**. Resignation records only exist for people who have already left. Everyone still working is, by definition, unlabeled — and will remain so until they either stay long enough to be surveyed or walk out the door. Organizations that wait until they have "enough labeled data" to build a model will always be predicting the past.

> **Ethical note:** Turnover prediction models inform retention priorities, not performance reviews or termination decisions. A probability score indicates risk exposure for the organization — it does not define an employee's value. Treat model output as a signal for proactive HR intervention, not as a judgment.

Self-Training is the algorithm family designed for exactly this condition. It begins with what we know (500 labeled records), builds an initial model, scores the unlabeled pool with that model, and **selectively promotes the most confident predictions into the training set** — creating pseudo-labels that expand the learning signal without inventing information. The key design choice is the confidence threshold: in this project, only employees with P(resign) ≥ 0.90 or ≤ 0.10 receive a pseudo-label. Everyone in the uncertain middle gets discarded.

The result is a model that learned from 606 examples — 400 labeled, 206 pseudo-labeled — while being evaluated exclusively on real, held-out labels. That boundary is what keeps the validation honest.

---

## 📊 Dataset

**4,500 employee records · 7 features · Target: `resigned` (0 / 1 / NaN)**

The defining characteristic of this dataset is its label structure: 500 records carry a confirmed outcome (`resigned = 0` or `1`); 4,000 carry `NaN` — not missing data, but genuinely unknown future. The labeled subset is near-balanced: 278 stayed (55.6%), 222 resigned (44.4%).

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `age` | int | 20–60 | Employee age in years |
| `years_at_company` | int | 0–20 | Organizational tenure |
| `job_level` | int | 1–4 | Role hierarchy: 1=Operative, 2=Analyst, 3=Coordinator, 4=Manager |
| `monthly_salary` | float | 8,000–70,000 | Gross monthly salary (MXN) |
| `job_satisfaction` | int | 1–5 | Self-reported satisfaction score |
| `workload` | int | 1–5 | Reported workload intensity |
| `avg_overtime_hours` | int | 0–20 | Average overtime hours per month |
| `resigned` | float | 0 / 1 / NaN | Target — NaN = actively employed (unlabeled) |

**Data Origin — where these variables live in a real organization:**

| Feature | Source System | Department Owner |
|---------|--------------|-----------------|
| `age`, `years_at_company` | HRIS (SAP SuccessFactors / Workday) | HR Operations |
| `job_level` | Org Chart / HRIS | Talent Management |
| `monthly_salary` | Payroll System (ADP / SAP HR) | Compensation & Benefits |
| `job_satisfaction` | Employee Engagement Platform (Glint / Qualtrics) | People Analytics |
| `workload` | Manager Assessment / Project Tracking | Direct Management |
| `avg_overtime_hours` | Time & Attendance System | HR Operations |
| `resigned` | Resignation Tracking / Exit Survey | HR Business Partners |

**Three EDA findings that shaped the model:**

**1. Low satisfaction + high workload is a crisis combination.** In the labeled subset, the (satisfaction=1, workload=5) cell has near-100% resignation rate. The model learned this interaction; it appears clearly in the response surface from Section 9.

**2. Resigned employees average significantly more overtime.** Among labeled records, the resigned group averages roughly 2–4 more overtime hours per month than stayers. Not a dramatic gap — but consistent enough to carry predictive weight.

**3. Labeled and unlabeled employees share similar distributions.** The 500 labeled records are not a biased slice of the population — their feature distributions closely mirror the 4,000 unlabeled. This matters for semi-supervised methods: if the labeled pool were demographically skewed, the pseudo-labels would inherit that bias.

---

## 🤖 Model — Self-Training with Random Forest

Self-Training is a meta-algorithm, not a model family. It wraps around a supervised base learner and teaches that learner to expand its own training set. The choice of base learner matters: it needs to produce calibrated probability estimates, handle mixed numeric features without heavy preprocessing, and be reasonably robust to noisy pseudo-labels.

Random Forest satisfies all three conditions. It generates probability scores through majority vote across 200 trees, is scale-invariant (no normalization needed), and its ensemble structure means individual noisy observations are unlikely to dominate the decision boundary.

**Self-Training loop — what actually happened:**

| Step | What occurred |
|------|--------------|
| Initial training | Base RF trained on 400 labeled records |
| Unlabeled scoring | 4,000 employee profiles scored for P(resign) |
| Confidence filter | Threshold: P ≥ 0.90 → pseudo-resigned · P ≤ 0.10 → pseudo-stayed |
| Pseudo-labels generated | 206 employees (60 pseudo-resigned, 146 pseudo-stayed) |
| Discarded (uncertain zone) | 3,794 employees — their future stays unknown |
| Expanded training set | 606 records (400 real + 206 pseudo) |
| Final model | New RF retrained on expanded set |
| Evaluation | 100 held-out real-labeled records — never touched during training |

The 90%/10% threshold is conservative by design. A lower bar (e.g., 70%/30%) would include more pseudo-labels but introduce more noise. The right threshold depends on the base model's calibration and the acceptable level of label uncertainty — which is itself a business decision, not a pure ML one.

---

## 📈 Key Results

Evaluated on 100 held-out real-labeled records — test set separated before any pseudo-labeling occurred.

| Metric | Value |
|--------|-------|
| Accuracy | 71.0% |
| AUC-ROC | 0.7940 |
| F1 Score | 0.6420 |
| Precision | 70.3% |
| Recall | 59.1% |

**Confusion Matrix (n = 100 test records):**

| | Predicted: Stay | Predicted: Resign |
|--|--|--|
| **Actual: Stay** | 45 ✓ | 11 ✗ |
| **Actual: Resign** | 18 ✗ | 26 ✓ |

**Operational interpretation:** The model correctly flags 26 of 44 at-risk employees (Recall 59.1%) with high precision — 70% of its resignation alerts are real. The 18 missed cases (false negatives) represent employees who resigned without the model catching the signal. In HR terms, false negatives are the more costly error: each is a departure the organization had no chance to prevent.

An AUC-ROC of 0.794 means the model meaningfully ranks high-risk employees above low-risk ones — which is what matters for prioritized retention action. At 71% accuracy on a label-scarce problem where supervised models would train on 400 examples, self-training earns its complexity.

---

## 🔍 Top Drivers — Feature Importance

Feature importances from the final self-training model (Mean Decrease Impurity across 200 trees):

| Feature | Importance | Operational Meaning |
|---------|-----------|---------------------|
| `job_satisfaction` | 0.218 | Highest single signal — disengaged employees leave |
| `monthly_salary` | 0.168 | Compensation anchors retention across all levels |
| `workload` | 0.167 | Intensity without recognition compounds dissatisfaction |
| `avg_overtime_hours` | 0.154 | Chronic overtime reveals workload-contract misalignment |
| `years_at_company` | 0.130 | Tenure shapes departure probability non-linearly |
| `age` | 0.124 | Career stage affects risk tolerance and mobility |
| `job_level` | 0.040 | Weakest driver — level alone explains little without context |

The four dominant features — satisfaction, salary, workload, overtime — form a coherent narrative: **people leave when the psychological and economic contract with the organization breaks down**. Job level, by itself, carries almost no predictive weight.

---

## 🗂️ Repository Structure

```
SelfT_HR_Turnover_Prediction/
│
├── 13_SelfT_HR_Turnover_Prediction.ipynb   # Notebook (no outputs)
├── hr_data.csv                          # 750-row sample (500 labeled + 250 unlabeled)
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (4,500 rows with 4,000 unlabeled employees),
> notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` simulator
> available on [Gumroad](https://lozanolsa.gumroad.com).
>
> The GitHub CSV is intentionally structured as a learning artifact: 500 labeled rows
> + 250 unlabeled rows (NaN target) so you can run and understand the semi-supervised
> pipeline without the full dataset. The complete unlabeled pool is what makes
> self-training operationally meaningful — that lives in the full pack.

---

## 🚀 How to Run

**Google Colab (recommended):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/SelfT_HR_Turnover_Prediction/blob/main/13_SelfT_HR_Turnover_Prediction.ipynb)

The notebook loads the CSV automatically from GitHub if not found locally.

**Local:**

```bash
git clone https://github.com/LozanoLsa/SelfT_HR_Turnover_Prediction.git
cd SelfT_HR_Turnover_Prediction
pip install -r requirements.txt
jupyter notebook 13_SelfT_HR_Turnover_Prediction.ipynb
```

**requirements.txt:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 💡 Key Learnings

**1. Labeled data is a bottleneck, not a given.**
Most organizations treat the absence of labeled data as a reason to delay modeling. Self-training treats it as the starting condition. The question shifts from "when will we have enough labels?" to "how much can we extract from what we already know?"

**2. The confidence threshold is a design choice, not a hyperparameter to optimize blindly.**
Setting 90%/10% instead of 70%/30% means discarding 3,794 of 4,000 unlabeled records. That is not waste — it is selectivity. Pseudo-labels that are wrong systematically corrupt the training distribution in ways that are hard to detect after the fact.

**3. Evaluation must stay on real labels only.**
The value of pseudo-labels is in expanding the training signal. The test set must never contain them. Mixing pseudo-labels into evaluation produces inflated metrics that reflect the model's own confidence, not its actual accuracy on unknown outcomes.

**4. Satisfaction and salary carry more signal than job level.**
A manager with low satisfaction and high workload carries more resignation risk than an operative who is content. Hierarchical position, alone, explains almost nothing. This has direct implications for where HR should focus retention investment — across levels, not just for senior roles.

**5. Semi-supervised is a bridge, not a destination.**
Self-training works when labeled data is scarce and the base model produces calibrated probabilities. As more labeled records accumulate (through exit surveys, annual reviews), the unlabeled pool shrinks and supervised methods become viable again. The right approach evolves with the data.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
