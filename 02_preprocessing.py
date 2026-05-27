"""
02_preprocessing.py — Imputation, Feature Engineering, Encoding, VIF
Run: python 02_preprocessing.py
Inputs:  data/df_clean.parquet
Outputs: data/X_train.parquet, data/X_test.parquet,
         data/y_train.parquet, data/y_test.parquet,
         data/preprocessor.pkl
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing_pipeline import build_preprocessing_pipeline

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET      = "churn"
RANDOM_SEED = 42

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_parquet("data/df_clean.parquet")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET], errors="ignore").copy()

SCALE_POS_WEIGHT = (1 - y.mean()) / y.mean()
print(f"  Dataset : {X.shape}  |  Churn rate: {y.mean():.2%}")
print(f"  Scale pos weight: {SCALE_POS_WEIGHT:.2f}")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── Build & fit pipeline ───────────────────────────────────────────────────────
print("\nBuilding preprocessing pipeline ...")
preprocessor = build_preprocessing_pipeline(vif_threshold=None)  # Set VIF threshold to None to skip VIF-based feature dropping

print("Fitting on train set (this includes OOF target encoding + VIF) ...")
X_train_proc = preprocessor.fit_transform(X_train, y_train)
# fit_transform returns a DataFrame; reset index for clean parquet output
X_train_proc = pd.DataFrame(X_train_proc).reset_index(drop=True)

print("Transforming test set ...")
X_test_proc = preprocessor.transform(X_test)
X_test_proc = pd.DataFrame(X_test_proc).reset_index(drop=True)

# Sanity checks
assert X_train_proc.isnull().sum().sum() == 0, "NaNs remain in X_train after preprocessing!"
assert X_test_proc.isnull().sum().sum()  == 0, "NaNs remain in X_test after preprocessing!"
assert list(X_train_proc.columns) == list(X_test_proc.columns), "Train / test column mismatch!"

print(f"\n[OK] Train shape after preprocessing : {X_train_proc.shape}")
print(f"[OK] Test  shape after preprocessing : {X_test_proc.shape}")

# ── Save ───────────────────────────────────────────────────────────────────────
X_train_proc.to_parquet("data/X_train.parquet", index=False)
X_test_proc.to_parquet("data/X_test.parquet",   index=False)
y_train.to_frame().reset_index(drop=True).to_parquet("data/y_train.parquet", index=False)
y_test.to_frame().reset_index(drop=True).to_parquet("data/y_test.parquet",   index=False)

# Save fitted preprocessor so downstream scripts / inference can reuse it
joblib.dump(preprocessor, "data/preprocessor.pkl")
pd.Series({"SCALE_POS_WEIGHT": SCALE_POS_WEIGHT}).to_json("data/config.json")

print("\n[OK] Saved:")
print("     data/X_train.parquet")
print("     data/X_test.parquet")
print("     data/y_train.parquet")
print("     data/y_test.parquet")
print("     data/preprocessor.pkl")
print("\n   Run next: python 03_baseline_cv.py")