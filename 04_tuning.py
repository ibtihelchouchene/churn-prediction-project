"""
04_tuning.py — Hyperparameter Tuning (XGBoost & LightGBM, RandomizedSearchCV)
Run: python 04_tuning.py
Inputs:  data/X_train.parquet, data/y_train.parquet, data/config.json
Outputs: data/best_xgb.pkl, data/best_lgb.pkl, data/tuning_summary.json
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
X_tr = pd.read_parquet("data/X_train.parquet").astype(np.float32)
y_tr = pd.read_parquet("data/y_train.parquet").squeeze().to_numpy()
config = pd.read_json("data/config.json", typ="series")
SCALE_POS_WEIGHT = float(config["SCALE_POS_WEIGHT"])
RANDOM_SEED = 42
print(f"  X_tr: {X_tr.shape}")

# ── CV + scorer ───────────────────────────────────────────────────────────────
cv_strat   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")

# ── XGBoost tuning ────────────────────────────────────────────────────────────
xgb_param_dist = {
    "n_estimators":     [200, 300, 500],
    "learning_rate":    stats.loguniform(0.01, 0.2),
    "max_depth":        stats.randint(3, 8),
    "subsample":        stats.uniform(0.6, 0.4),
    "colsample_bytree": stats.uniform(0.6, 0.4),
    "gamma":            stats.loguniform(1e-4, 5),
    "reg_alpha":        stats.loguniform(1e-4, 10),
    "reg_lambda":       stats.loguniform(1e-4, 10),
    "min_child_weight": stats.randint(1, 10),
}

xgb_base = XGBClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT,
    eval_metric="auc",
    random_state=RANDOM_SEED,
    n_jobs=1,
    verbosity=0,
)

xgb_search = RandomizedSearchCV(
    xgb_base, param_distributions=xgb_param_dist,
    n_iter=30, scoring=auc_scorer, cv=cv_strat,
    refit=True, random_state=RANDOM_SEED,
    n_jobs=1, verbose=1, error_score="raise",
)

print("\n[TUNE] Tuning XGBoost...")
xgb_search.fit(X_tr, y_tr)
print(f"[OK] XGBoost best CV ROC-AUC : {xgb_search.best_score_:.4f}")
print(f"   Best params: {xgb_search.best_params_}")

# ── LightGBM tuning ───────────────────────────────────────────────────────────
lgb_param_dist = {
    "n_estimators":      [200, 300, 500],
    "learning_rate":     stats.loguniform(0.01, 0.2),
    "num_leaves":        stats.randint(20, 50),
    "subsample":         stats.uniform(0.6, 0.4),
    "colsample_bytree":  stats.uniform(0.6, 0.4),
    "reg_alpha":         stats.loguniform(1e-4, 10),
    "reg_lambda":        stats.loguniform(1e-4, 10),
    "min_child_samples": stats.randint(5, 30),
}

lgb_base = lgb.LGBMClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT,
    random_state=RANDOM_SEED,
    n_jobs=1,
    verbose=-1,
)

lgb_search = RandomizedSearchCV(
    lgb_base, param_distributions=lgb_param_dist,
    n_iter=30, scoring=auc_scorer, cv=cv_strat,
    refit=True, random_state=RANDOM_SEED,
    n_jobs=1, verbose=1, error_score="raise",
)

print("\n[TUNE] Tuning LightGBM...")
lgb_search.fit(X_tr, y_tr)
print(f"[OK] LightGBM best CV ROC-AUC : {lgb_search.best_score_:.4f}")
print(f"   Best params: {lgb_search.best_params_}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"Model": "LightGBM", "CV ROC-AUC (tuned)": lgb_search.best_score_},
    {"Model": "XGBoost",  "CV ROC-AUC (tuned)": xgb_search.best_score_},
]).sort_values("CV ROC-AUC (tuned)", ascending=False).reset_index(drop=True)
print("\n[DATA] Tuning Summary:")
print(summary.to_string(index=False))

# ── Save models + summary ─────────────────────────────────────────────────────
with open("data/best_xgb.pkl", "wb") as f:
    pickle.dump(xgb_search.best_estimator_, f)
with open("data/best_lgb.pkl", "wb") as f:
    pickle.dump(lgb_search.best_estimator_, f)

summary.to_json("data/tuning_summary.json", orient="records")
print("\n[OK] Saved: data/best_xgb.pkl, data/best_lgb.pkl, data/tuning_summary.json")
print("   Run next: python 05_evaluation.py")
