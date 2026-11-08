# 🔍 Fraud Detection — IEEE-CIS Dataset

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.91-brightgreen)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.53-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **End-to-end fraud detection pipeline** on 590,540 real transactions from 
> Vesta Corporation, achieving ROC-AUC 0.91 and PR-AUC 0.53 (15x above 
> random baseline). Includes business-driven threshold optimization, 
> SHAP explainability, and production-ready model monitoring.

---

## 💼 Business Problem

Payment fraud costs the global economy over $32 billion annually. 
This project builds a production-ready fraud detection system that:

- Detects **80% of fraudulent transactions** at the optimal operating threshold
- Generates **$134,120 net profit** on the test set vs. **-$20,200** with the default threshold
- Provides **transaction-level explanations** for every decision (GDPR-compliant)
- Monitors **model drift** across production time windows

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9103** |
| PR-AUC | **0.5303** (15x above random baseline of 0.035) |
| Recall at optimal threshold | **80.0%** |
| Optimal threshold | **0.13** (vs. default 0.5) |
| Net profit improvement | **+$154,320** vs. default threshold |

### Business Impact

| Metric | Default (0.50) | Optimized (0.13) |
|--------|---------------|-----------------|
| Fraud detected | 49.1% (1,516 cases) | **80.0% (2,469 cases)** |
| False alerts | 1,264 | 11,563 |
| Revenue protected | $204,660 | **$333,315** |
| Net profit | -$20,200 | **+$134,120** |

> Cost assumptions: missed fraud = $135 (avg transaction), false alert = $10 (analyst review)

---

## 🗂️ Project Structure

fraud-detection/

├── raw_data/                  # Original IEEE-CIS files (not tracked)

├── processed/                 # Engineered features, model artifacts

├── notebooks/

│   ├── 01_data_ingestion.ipynb       # Merge + missing data analysis

│   ├── 02_eda.ipynb                  # Exploratory analysis

│   ├── 03_feature_engineering.ipynb  # Imputation + feature creation

│   ├── 04_modeling.ipynb             # XGBoost + Optuna + SHAP CV selection

│   ├── 05_threshold_optimization.ipynb # Profit curve + business impact

│   ├── 06_explainability.ipynb       # SHAP global + local explanations

│   └── 07_model_monitoring.ipynb     # Evidently drift detection

├── models/                    # Trained model artifacts

├── figures/                   # Visualizations

├── reports/                   # Evidently HTML drift reports

├── app/                       # Streamlit dashboard

├── model_card.md              # Model documentation

└── README.md

---

## 🔬 Methodology

### 1. Data & EDA
- Merged `train_transaction` (590,540 rows) with `train_identity` (144,233 rows)
- 414 of 434 columns contain missing values — missingness is **not random**:
  - `id_*` columns: missing = no identity verification data
  - `D*` columns: missing = no prior event to measure
  - `V*` columns: missing in conditional blocks by transaction type
- Key EDA finding: fraud rate drops during high-volume spike (day ~20-25) 
  due to product mix shift toward lower-risk categories — not a genuine 
  fraud reduction

### 2. Feature Engineering
- Differentiated imputation strategy preserving missingness as signal 
  (sentinel -999 + binary flags)
- Engineered features: `amt_zscore_card`, `amt_vs_card_mean`, `card_mean`, 
  `card_std`, `hour`, `day_of_week`
- Final dataset: **0 nulls, 475 columns**

### 3. Feature Selection
- **SHAP CV**: computed SHAP importance across 3 temporal folds 
  (TimeSeriesSplit) to reduce selection bias
- **Correlation filter**: removed one of each pair with correlation > 0.95
- Result: **66 features** selected from 474

### 4. Modeling
- Algorithm: **XGBoost** with Optuna hyperparameter tuning (30 trials)
- Validation: **TimeSeriesSplit** (3 folds) — respects temporal order
- Class imbalance: `scale_pos_weight = sqrt(neg/pos) ≈ 5.2x`
- Temporal split: 70% train / 15% val / 15% test

### 5. Threshold Optimization
- Default threshold (0.5) generates **-$20,200 loss** on test set
- Profit curve optimization finds threshold **0.13** → **+$134,120 profit**
- Cost ratio FN/FP = 13.5x justifies lower threshold

### 6. Explainability (SHAP)
- Global: C13, C1, card_mean, V258, day are top fraud signals
- Local: waterfall plots for individual transaction explanations
- GDPR-compliant: every automated decision can be explained

### 7. Model Monitoring
- Evidently AI drift reports across 3 temporal windows
- ROC-AUC degrades from 0.982 (training) → 0.910 (month 2)
- Retraining trigger: ROC-AUC < 0.88 or Recall drop > 15%

---

## 📈 Key Visualizations

### Class Distribution & Fraud Rate by Product
![EDA](figures/03_fraud_by_product.png)

### Profit Curve — Threshold Optimization
![Profit Curve](figures/10_profit_curve.png)

### SHAP Global Feature Importance
![SHAP](figures/12_shap_beeswarm.png)

### Model Performance Over Time
![Monitoring](figures/15_performance_over_time.png)

---

## ⚙️ Setup

```bash
# Create environment
conda create -n fraud_env python=3.11 -y
conda activate fraud_env

# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna shap evidently \
            streamlit plotly matplotlib seaborn pyarrow jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name fraud_env --display-name "Python (fraud_env)"
```

### Data
Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) 
and place `train_transaction.csv` + `train_identity.csv` in `raw_data/`.

---

## 🚀 Run the Dashboard

```bash
cd app
streamlit run dashboard.py
```

Or visit the live demo: **[🔗 Live Dashboard](#)** ← *(link after Streamlit Cloud deploy)*

---

## 📋 Limitations

- V features are anonymized — their business meaning cannot be verified
- No velocity features (real-time aggregations) — main gap vs. production systems
- Predicted probabilities are inflated due to `scale_pos_weight` — 
  threshold optimization compensates for this
- Concept drift detected after ~2 months — monthly retraining recommended

See full documentation in [`model_card.md`](model_card.md).

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600?style=flat)
![SHAP](https://img.shields.io/badge/-SHAP-FF0000?style=flat)
![Evidently](https://img.shields.io/badge/-Evidently_AI-6C4EF5?style=flat)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Optuna](https://img.shields.io/badge/-Optuna-3A8EBA?style=flat)

---

## 👩‍💻 Author

**Olga Bencomo** 
[LinkedIn](#) · [GitHub](#)