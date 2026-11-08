# Model Card — IEEE-CIS Fraud Detection

## Model Overview

| Field | Details |
|---|---|
| **Model type** | XGBoost Classifier |
| **Task** | Binary classification (fraud vs. legitimate transaction) |
| **Version** | 1.0 |
| **Last updated** | July 2026 |
| **Owner** | Olga Bencomo |

---

## Problem Statement

Payment fraud causes significant financial losses to merchants and financial 
institutions. This model identifies fraudulent transactions in real time, 
enabling automated intervention before settlement. The primary business goal 
is to maximize revenue protected from fraud while keeping manual review costs 
operationally manageable.

---

## Dataset

| Field | Details |
|---|---|
| **Source** | IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation) |
| **Size** | 590,540 transactions |
| **Period** | ~6 months (exact dates anonymized by Vesta) |
| **Fraud rate** | 3.50% (class imbalance handled via scale_pos_weight) |
| **Features** | 434 raw features → 66 selected via SHAP CV + correlation filter |
| **Split** | Temporal 70/15/15% (train/val/test) — no data leakage |

### Feature families
- **C features**: transaction count aggregations by card/address
- **V features**: Vesta proprietary features (anonymized, 159 conditional blocks)
- **D features**: time-delta features (days since prior event)
- **M features**: match flags between card and transaction metadata
- **Engineered**: `card_mean`, `card_std`, `amt_zscore_card`, `amt_vs_card_mean`, 
  `hour`, `day_of_week`

---

## Training

| Field | Details |
|---|---|
| **Algorithm** | XGBoost (gradient boosted trees) |
| **Hyperparameter tuning** | Optuna (30 trials, TimeSeriesSplit CV, optimizing PR-AUC) |
| **Class imbalance** | `scale_pos_weight = sqrt(neg/pos) ≈ 5.2x` |
| **Feature selection** | SHAP CV (3 temporal folds) + correlation filter (>0.95) |
| **Final features** | 66 |

---

## Performance

### Test set metrics (temporal holdout, last 15% of data)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9103 |
| PR-AUC | 0.5303 |
| Recall (fraud, optimal threshold) | 80.0% |
| Precision (fraud, optimal threshold) | 17.6% |

### Business impact (test set, 88,581 transactions)

| Metric | Default (0.50) | Optimized (0.13) |
|---|---|---|
| Fraud detected | 49.1% (1,516 cases) | 80.0% (2,469 cases) |
| False alerts | 1,264 | 11,563
| Revenue protected | $204,660 | $333,315 |
| Cost of false alerts | $12,640 | $115,630 |
| Cost of missed fraud | $212,220 | $83,565 |
| Net profit | -$20,200 | +$134,120 |

> Cost assumptions: FN = $135 (avg transaction amount), FP = $10 (analyst review)

---

## Threshold

The optimal classification threshold is **0.13** (vs. default 0.5), derived 
from a profit curve that maximizes expected net profit given the FN/FP cost 
ratio of 13.5x. The threshold should be recalibrated when:
- Business costs change (chargeback rates, analyst salaries)
- Fraud rate shifts more than 50% from baseline
- Recall drops more than 15% in production monitoring

---

## Production Monitoring

Model performance was evaluated across 3 temporal windows using Evidently AI:

| Window | ROC-AUC | Recall | Precision |
|---|---|---|---|
| Reference (training) | 0.982 | 0.945 | 0.262 |
| Window 1 (month 1) | 0.923 | 0.817 | 0.190 |
| Window 2 (month 2) | 0.910 | 0.800 | 0.176 |

**Recommended retraining triggers:**
- ROC-AUC drops below 0.88 vs reference
- Recall drops more than 15% vs reference
- Detection rate increases more than 50% vs reference

---

## Limitations

- **Synthetic/anonymized features**: V features are proprietary to Vesta 
  Corporation and anonymized — their business meaning cannot be verified. 
  This limits domain-driven feature engineering.
- **No velocity features**: real production fraud models use real-time 
  aggregations (transactions per card in last 1h/24h/7d) that are not 
  available in this static dataset. This is the main gap vs. state-of-the-art.
- **Probability calibration**: predicted probabilities are inflated 
  (mean 0.19 vs actual fraud rate 0.035) due to `scale_pos_weight`. 
  The model ranks correctly (ROC-AUC 0.91) but absolute probabilities 
  should not be interpreted as true fraud likelihood.
- **Concept drift**: ROC-AUC degrades from 0.982 (training) to 0.910 
  (2 months later), consistent with known fraud pattern evolution. 
  Monthly retraining is recommended in production.
- **Selection bias in feature selection**: SHAP CV reduces but does not 
  eliminate selection bias — features were selected based on models trained 
  on overlapping data.

---

## Ethical Considerations

- **Automated decisions**: this model makes automated fraud decisions that 
  directly affect customers. Under GDPR Article 22, customers have the right 
  to explanation — SHAP waterfall plots provide transaction-level explanations.
- **False positives**: legitimate transactions flagged as fraud cause customer 
  friction. The chosen threshold (0.13) generates ~11,500 false alerts per 
  88k transactions — operational processes should include a fast dispute 
  resolution channel.
- **Demographic bias**: the dataset does not include demographic variables, 
  so disparate impact analysis by protected characteristics (age, gender, 
  nationality) was not possible. This should be evaluated before production 
  deployment with real customer data.
- **Data privacy**: all training data is anonymized by Vesta Corporation. 
  No PII was used in model development.

---

## Intended Use

| ✅ Intended | ❌ Not intended |
|---|---|
| Real-time transaction scoring | Credit scoring or loan decisions |
| Fraud analyst queue prioritization | Identity verification |
| Threshold-based automated blocking | Any decision affecting protected classes without bias audit |

---

## Next Iterations

1. Add velocity features (transaction count per card in 1h/24h/7d windows)
2. Add email domain risk features (historical fraud rate by P_emaildomain)
3. Implement proper probability calibration compatible with XGBoost + scale_pos_weight
2. Explore LightGBM as alternative — faster training, better calibration
3. Nested cross-validation to fully eliminate selection bias

---

## References

- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- SHAP: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- Model Cards: Mitchell et al. (2019), "Model Cards for Model Reporting"
- Evidently AI: [evidentlyai.com](https://www.evidentlyai.com)