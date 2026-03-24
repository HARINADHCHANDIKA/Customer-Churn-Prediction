# 📉 Customer Churn Prediction — Telecom Dataset

> **Internship ML Project** | End-to-end machine learning pipeline predicting which telecom customers are likely to churn, using Logistic Regression, Random Forest, and Gradient Boosting with full EDA, evaluation, and a production-ready inference engine.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=flat&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Seaborn-11557c?style=flat)

---

## 🎯 Objective

Predict which customers are likely to stop using a telecom service (churn), enabling targeted retention campaigns before customers leave. Every retained customer saves an estimated **$300+ in acquisition costs**.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Synthetic telecom dataset (IBM Telco schema) |
| Rows | 7,043 customers |
| Features | 20 raw → 44 engineered (after OHE) |
| Target | `Churn` (binary: 0 = Retained, 1 = Churned) |
| Churn rate | **20.4%** (class imbalance handled with `class_weight='balanced'`) |
| Train / Test | 80% / 20% — stratified split |

### Key Features
- **Contract type** — strongest churn predictor (month-to-month: 28.7% churn vs 9.4% for two-year)
- **Internet service** — Fiber optic customers churn 2.3× more than DSL
- **Tenure** — Customers with < 6 months tenure churn at 35%+; loyalty grows with time
- **Online Security / Tech Support** — Protective services reduce churn by ~40%
- **Payment method** — Electronic check correlates with highest churn (34%)

---

## 🗂️ Project Structure

```
churn_project/
│
├── data/
│   ├── generate_dataset.py      # Realistic telecom data generator
│   └── telecom_churn.csv        # Generated dataset (7,043 rows)
│
├── notebooks/
│   ├── eda.py                   # EDA — 8 charts (distributions, correlations, cohorts)
│   ├── train_models.py          # Full ML pipeline — 3 models, 7 evaluation charts
│   └── final_report.py          # Publication-quality summary dashboard
│
├── models/
│   ├── logistic_regression.pkl  # Best AUC model (0.8155)
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── outputs/                     # All 15 generated charts + JSON report
│   ├── eda_01_overview_dashboard.png
│   ├── eda_02_correlation_heatmap.png
│   ├── eda_03_categorical_churn_rates.png
│   ├── eda_04_tenure_charges_scatter.png
│   ├── eda_05_tenure_cohort_churn.png
│   ├── eda_06_service_churn_impact.png
│   ├── eda_07_financial_distributions.png
│   ├── eda_08_churn_risk_matrix.png
│   ├── model_09_metrics_comparison.png
│   ├── model_10_roc_pr_curves.png
│   ├── model_11_confusion_matrices.png
│   ├── model_12_feature_importance.png
│   ├── model_13_cv_calibration.png
│   ├── model_14_threshold_tuning.png
│   ├── model_15_score_distributions.png
│   ├── FINAL_churn_prediction_report.png
│   └── evaluation_report.json
│
├── predict.py                   # Production inference script
└── README.md
```

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | CV AUC (5-fold) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7360 | 0.4175 | **0.7491** | 0.5362 | **0.8155 🏆** | 0.8253 ± 0.0121 |
| Random Forest | 0.7807 | 0.4689 | 0.5784 | 0.5179 | 0.8013 | 0.8130 ± 0.0068 |
| Gradient Boosting | **0.8055** | **0.5359** | 0.3380 | 0.4145 | 0.7833 | 0.7999 ± 0.0050 |

### Key Findings
- **Best ROC-AUC**: Logistic Regression (0.8155) — best discriminative power overall
- **Best Recall**: Logistic Regression (0.7491) — catches 74.9% of actual churners
- **Best Accuracy**: Gradient Boosting (0.8055) — but biased toward majority class
- **Threshold tuning**: Default 0.5 → tuned **0.40** for higher recall; min-cost business threshold at **0.246**
- All models cross-validate stably (low std), confirming no overfitting

---

## 🔍 EDA Highlights

| Finding | Detail |
|---|---|
| Contract is #1 churn driver | Month-to-month: 28.7% churn vs 9.4% (two-year) |
| Fiber optic doubles churn risk | 30.5% vs 13.1% (DSL) — price sensitivity |
| New customers are highest risk | Tenure < 6 months: ~35% churn rate |
| Security add-ons are protective | OnlineSecurity: 14% churn vs 34% without |
| Electronic check payment flag | 34% churn — friction / dissatisfaction signal |
| Senior citizens churn more | +40% relative churn vs non-seniors |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### 2. Generate dataset
```bash
python3 data/generate_dataset.py
```

### 3. Run EDA (generates 8 charts)
```bash
python3 notebooks/eda.py
```

### 4. Train models (generates 7 evaluation charts + models)
```bash
python3 notebooks/train_models.py
```

### 5. Generate final report poster
```bash
python3 notebooks/final_report.py
```

### 6. Run inference on new customers
```bash
# Demo mode (3 test customers)
python3 predict.py

# Score your own CSV
python3 predict.py --csv new_customers.csv --threshold 0.40
```

---

## 🔮 Inference Example

```
══════════════════════════════════════════════════════
  CUSTOMER CHURN PREDICTION — INFERENCE RESULTS
  Model: Logistic Regression  |  Threshold: 0.40
══════════════════════════════════════════════════════

  Customer : DEMO-001 (High Risk)
  Churn Prob: 97.8%   🔴 CRITICAL
  Predicted : ⚠ CHURN
  Action    : Immediate outreach — offer retention discount

  Customer : DEMO-002 (Low Risk)
  Churn Prob: 0.7%    🟢 LOW
  Predicted : ✓ RETAIN
  Action    : Standard engagement — monitor quarterly

  Customer : DEMO-003 (Medium Risk)
  Churn Prob: 63.9%   🟠 HIGH
  Predicted : ⚠ CHURN
  Action    : Proactive call — upsell TechSupport bundle
```

---

## 💡 Business Impact

Assuming 10,000 monthly customers and $300 average revenue per churner:

| Scenario | Monthly Churn | Revenue at Risk | Recoverable (at 30% save rate) |
|---|---|---|---|
| No model (baseline) | 2,040 | $612,000 | — |
| With model (recall=75%) | 1,530 caught | $459,000 | **$137,700/month** |

Using the **business cost-optimized threshold (0.246)**, total intervention cost is minimized by balancing the $300 FN cost (missed churner) against the $30 FP cost (unnecessary outreach).

---

## 📈 Visualizations

15 charts covering:
- **EDA** (8): churn overview, correlations, all categorical features, tenure scatter/hexbin, cohort analysis, service impact, financial distributions, risk matrix
- **Models** (7): metrics comparison, ROC + PR curves, confusion matrices, feature importance, CV + calibration, threshold tuning, score distributions
- **Final poster**: publication-ready summary dashboard

---

## 🛠️ Technical Stack

| Component | Tool |
|---|---|
| Data manipulation | pandas 3.0, numpy |
| ML models | scikit-learn 1.8 |
| Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Validation | StratifiedKFold (5-fold), stratified train/test split |
| Visualisation | matplotlib, seaborn, scipy (KDE) |
| Serialisation | pickle |

---

## 👤 Author

Built as an internship machine learning project demonstrating:
- End-to-end ML pipeline design
- Exploratory data analysis with actionable insights
- Multi-model comparison with proper evaluation methodology
- Threshold optimization for business cost minimization
- Production-ready inference engine

---

## 📄 License
MIT
