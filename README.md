#  Telecom Churn Prediction

> Machine Learning pipeline to predict customer churn on 100,000 telecom customers — with hyperparameter tuning, SHAP explainability, and an interactive Dash dashboard.

---

##  Results at a Glance

| Metric | Value |
|---|---|
| Best Model | XGBoost (tuned) |
| ROC-AUC | **0.6975** |
| PR-AUC | 0.6870 |
| F1 | 0.6790 |
| Recall | **79.76%** |
| Precision | 59.11% |
| Decision Threshold | 0.426 |

> At threshold 0.426 on 20,000 test customers: **7,906 churners correctly flagged**, 2,006 missed, 5,470 false alarms.

---

##  Project Structure

```
churn prediction/
├── 01_eda.py                   Step 1 — EDA & cleaning
├── preprocessing_pipeine.py    Feature engineering & encoding
├── 02_preprocessing.py         Step 2 — Call preprocessing_pipeine.py and save 
├── 03_baseline_cv.py           Step 3 — Baseline cross-validation (6 models)
├── 04_tuning.py                Step 4 — Hyperparameter tuning (XGBoost & LightGBM)
├── 05_evaluation.py            Step 5 — Test evaluation, SHAP, business summary
├── run_pipeline.py             Orchestrator — runs all steps in order
├── dashboard_v2.py             Standalone Dash dashboard (no retraining)
├── Telecom_customer churn.csv  Original dataset
│
├── data/                       Auto-created by pipeline
│   ├── df_clean.parquet
│   ├── X_train.parquet / X_test.parquet
│   ├── y_train.parquet / y_test.parquet
│   ├── best_lgb.pkl / best_xgb.pkl
│   ├── cv_results.json
│   └── tuning_summary.json
│
└── outputs/
    ├── eda/                    EDA charts
    └── eval/                   Confusion matrices, ROC/PR curves, SHAP plots
```

---

## ⚙️ Installation

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
pip install shap statsmodels scipy seaborn matplotlib pyarrow
pip install dash dash-bootstrap-components plotly
```

>  **Use Python 3.11.** Python 3.14 has a known `scipy` import crash that prevents `sklearn` and all ML libraries from loading.

---

##  Quick Start

```bash
# 1. Run the full pipeline (all 5 steps)
python run_pipeline.py

# 2. Resume from a specific step (skips earlier steps)
python run_pipeline.py --from 4

# 3. Run steps individually
python 01_eda.py
python preprocessing_pipeline.py
python 02_preprocessing.py
python 03_baseline_cv.py
python 04_tuning.py
python 05_evaluation.py

# 4. Launch the dashboard
python dashboard_v2.py
# Open http://127.0.0.1:8050
```

---

##  Pipeline Steps

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `01_eda.py` | Load CSV, clean nulls, EDA plots (heatmap, boxplots, histograms, correlations), t-tests, chi2 tests, drop low-signal columns | `data/df_clean.parquet` |
| 2 |` preprocessing_ pipeline.py`/`02_preprocessing.py` | Train/test split, median imputation, +20 engineered features, target encoding, OrdinalEncoder from preprocessing_pipeline.py | `X/y train & test parquets` |
| 3 | `03_baseline_cv.py` | 3-fold stratified CV across 6 models with F1-optimal thresholding | `data/cv_results.json` |
| 4 | `04_tuning.py` | RandomizedSearchCV (30 iters, 5-fold) for XGBoost & LightGBM | `best_lgb.pkl` / `best_xgb.pkl` |
| 5 | `05_evaluation.py` | Test metrics, confusion matrices, ROC/PR/calibration dashboard, SHAP global + local, business summary | `outputs/eval/*.png` |

---

##  Dataset

| Property | Value |
|---|---|
| Source | `Telecom_customer churn.csv` |
| Rows | 100,000 customers |
| Raw columns | 100 |
| Target | `churn` (0 = stayed, 1 = churned) |
| Churn rate | ~49.6% (near-balanced) |
| Train / Test | 80,000 / 20,000 (stratified) |

### Preprocessing Steps (`preprocessing.py`/`02_preprocessing.py`)

- Modular sklearn preprocessing pipeline with leak-free transformations  
- Median imputation for numerics, mode imputation for categoricals  
- 20+ engineered features: call completion rate, drop rate, revenue per minute, overage ratio, customer LTV, MOU/revenue momentum, revenue stability, active sub ratio, device health metrics, and risk flags  
- Percentile clipping at 1st–99th for ratio and momentum features (train-set bounds applied to test)  
- Binary mapping for `asl_flag`, `creditcd`, `dualband`, `ownrent`  
- Cross-fold target encoding (`k=20`, 5-fold OOF) for 7 high-cardinality categoricals — no target leakage  
- `OrdinalEncoder` for remaining categorical columns with unseen-category handling  
- Optional VIF-based multicollinearity removal for linear models  (skipped for tree models)
- Pipeline saved as `preprocessor.pkl` for reproducible inference and deployment  
---

##  Models

### Baseline Cross-Validation (`03_baseline_cv.py`)

6 models evaluated with 3-fold stratified CV. Threshold optimised per fold to maximise F1.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| XGBoost | 0.6934 ± 0.0029 | 0.6786 ± 0.0046 | 0.6892 ± 0.0017 |
| LightGBM | 0.6925 ± 0.0028 | 0.6782 ± 0.0041 | 0.6882 ± 0.0016 |
| Gradient Boosting | — | — | — |
| Random Forest | — | — | — |
| Logistic Regression | — | — | — |
| KNN | 0.5925 ± 0.0007 | 0.5824 ± 0.0010 | 0.6638 ± 0.0003 |

### Hyperparameter Tuning (`04_tuning.py`)

RandomizedSearchCV with 30 iterations and 5-fold CV, optimising ROC-AUC.

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | [200, 300, 500] |
| `learning_rate` | LogUniform(0.01, 0.20) |
| `max_depth` | randint(3, 8) |
| `subsample` | Uniform(0.6, 0.4) |
| `colsample_bytree` | Uniform(0.6, 0.4) |
| `gamma` / `reg_alpha` / `reg_lambda` | LogUniform(1e-4, 5–10) |
| `min_child_weight` | randint(1, 10) |

### Final Test Set Results (`05_evaluation.py`)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Threshold |
|---|---|---|---|---|---|---|
| **XGBoost (tuned)** | **0.6975** | **0.6870** | **0.6790** | **79.76%** | 59.11% | 0.426 |
| LightGBM (tuned) | 0.6949 | 0.6832 | 0.6798 | 80.65% | 58.75% | 0.424 |

> Composite score weights: ROC-AUC 35% · PR-AUC 25% · F1 20% · Recall 15% · Brier 5%

### Business Impact (XGBoost, threshold = 0.426)

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | TN = 4,618  | FP = 5,470 : wasted spend |
| **Actual Churn** | FN = 2,006 : revenue at risk | TP = 7,906  |

---

##  SHAP Explainability

SHAP values are computed with `TreeExplainer` on the best XGBoost model.

| Output file | Description |
|---|---|
| `shap_global.png` | Mean \|SHAP\| bar chart + beeswarm for top 20 features |
| `shap_dependence_top3.png` | Scatter plots for the 3 highest-importance features |
| `shap_local_churner.png` | Waterfall chart for the highest-risk customer |
| `shap_local_safe.png` | Waterfall chart for the lowest-risk customer |

**Top SHAP drivers (typical):**

- `rev_Mean` — monthly revenue (lower → higher churn risk)
- `mou_Mean` — minutes of use (declining usage is a strong signal)
- `change_mou` — month-over-month usage change
- `months` — tenure (shorter → higher risk)
- `eqpdays` — equipment age in days
- `customer_ltv` — `totmrc_Mean × months`

---

##  Dashboard (`dashboard_v2.py`)

Single-file Dash app. Reads all pipeline outputs — **no retraining on launch**.

```bash
python dashboard_v2.py
# Open http://127.0.0.1:8050
```

### Required files

| File | Purpose |
|---|---|
| `Telecom_customer churn.csv` | EDA tab dropdowns |
| `data/best_lgb.pkl` | Saved model (or `best_xgb.pkl`) |
| `data/X_test.parquet` + `y_test.parquet` | Predictions & metrics |
| `data/X_train.parquet` + `y_train.parquet` | Optimal threshold computation |
| `data/cv_results.json` | CV tab bar charts |
| `data/tuning_summary.json` | Tuning summary bar chart |

### Tabs

| Tab | Content |
|---|---|
| Overview | KPI strip, churn donut, risk segmentation, probability histogram, correlation heatmap |
| EDA | Interactive log-histograms, density plots, boxplots, categorical counts by churn |
| CV Results | Bar charts from `cv_results.json`, tuning summary, scores table |
| Performance | Live threshold slider → confusion matrix + 7 metrics update instantly; ROC & PR curves |
| Feature Importance | Top 20 built-in importance from saved model |
| Customer Explorer | Filterable DataTable with risk/prediction filters and CSV export |

---

##  Speed Optimisations

| Technique | Benefit |
|---|---|
| Parquet I/O between steps | 5–10× faster than CSV; ~50% smaller on disk |
| `float32` model inputs | Half the memory of float64 during training |
| `matplotlib.use("Agg")` | No GUI overhead; plots saved directly |
| VIF sampled to 5,000 rows | Avoids O(n²) regression on 80,000 rows |
| `--from N` resume flag | Skip completed steps; re-run only what changed |
| `n_jobs=1` in tuning | Prevents memory spikes from parallel forked processes |
| Module-level dict cache in dashboard | Predictions computed once at startup, never again |

---

##  All Output Files

| File | Description |
|---|---|
| `data/df_clean.parquet` | Cleaned dataset after dropping low-signal columns |
| `data/X_train.parquet` | Preprocessed training features |
| `data/X_test.parquet` | Preprocessed test features |
| `data/y_train.parquet` | Training labels |
| `data/y_test.parquet` | Test labels |
| `data/cv_results.json` | 3-fold CV scores for all 6 baseline models |
| `data/tuning_summary.json` | Tuning CV ROC-AUC for XGBoost and LightGBM |
| `data/best_xgb.pkl` | Best tuned XGBoost model |
| `data/best_lgb.pkl` | Best tuned LightGBM model |
| `outputs/eda/missing_heatmap.png` | Missing value heatmap |
| `outputs/eda/outlier_boxplots.png` | Boxplots by churn for key features |
| `outputs/eda/log_histograms.png` | Log-scale histograms by churn label |
| `outputs/eda/correlation_matrix.png` | Feature correlation heatmap |
| `outputs/eval/confusion_matrices.png` | Confusion matrices for both tuned models |
| `outputs/eval/evaluation_dashboard.png` | ROC, PR, calibration, threshold-sensitivity |
| `outputs/eval/shap_global.png` | SHAP bar + beeswarm for top 20 features |
| `outputs/eval/shap_dependence_top3.png` | SHAP dependence plots for top 3 features |
| `outputs/eval/shap_local_churner.png` | SHAP waterfall for highest-risk customer |
| `outputs/eval/shap_local_safe.png` | SHAP waterfall for lowest-risk customer |
