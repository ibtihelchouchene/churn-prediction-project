"""
03_baseline_cv.py — Baseline Cross-Validation (all models, 3-fold)
Run: python 03_baseline_cv.py
Inputs:  data/X_train.parquet, data/y_train.parquet, data/config.json
Outputs: data/cv_results.json  (best model name + scores)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_parquet("data/X_train.parquet").astype(np.float32)
y_train = pd.read_parquet("data/y_train.parquet").squeeze()
config  = pd.read_json("data/config.json", typ="series")
SCALE_POS_WEIGHT = float(config["SCALE_POS_WEIGHT"])
RANDOM_SEED = 42
print(f"  X_train: {X_train.shape}  |  Churn rate: {y_train.mean():.2%}")

# ── Helper ─────────────────────────────────────────────────────────────────────
def best_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    idx = np.argmax(f1s)
    return 0.0 if idx == 0 else thresholds[idx - 1]

# ── Model zoo ─────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                                   solver="saga", random_state=RANDOM_SEED)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=30,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=RANDOM_SEED,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="auc",
        early_stopping_rounds=20, random_state=RANDOM_SEED,
        n_jobs=-1, verbosity=0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT, random_state=RANDOM_SEED,
        n_jobs=-1, verbose=-1,
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
    ]),
}

EARLY_STOP = {"XGBoost", "LightGBM"}

# ── 3-fold stratified CV ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "=" * 65)
print("  CROSS-VALIDATION  (3-fold, train set only)")
print("=" * 65)

cv_log = {}
X_arr  = X_train.values
y_arr  = y_train.values

for name, model in MODELS.items():
    auc_scores, prauc_scores, f1_scores = [], [], []

    for tr_idx, val_idx in cv.split(X_arr, y_arr):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        if name in EARLY_STOP:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_val)[:, 1]
        thr    = best_threshold(y_val, y_prob)
        y_pred = (y_prob >= thr).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_prob))
        prauc_scores.append(average_precision_score(y_val, y_prob))
        f1_scores.append(f1_score(y_val, y_pred))

    cv_log[name] = {
        "ROC-AUC": float(np.mean(auc_scores)),
        "PR-AUC":  float(np.mean(prauc_scores)),
        "F1":      float(np.mean(f1_scores)),
    }
    print(f"\n  {name}")
    print(f"    ROC-AUC : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"    PR-AUC  : {np.mean(prauc_scores):.4f} ± {np.std(prauc_scores):.4f}")
    print(f"    F1      : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# ── Save CV summary ───────────────────────────────────────────────────────────
pd.DataFrame(cv_log).T.to_json("data/cv_results.json")
print("\n[OK] Saved: data/cv_results.json")
print("   Run next: python 04_tuning.py")
