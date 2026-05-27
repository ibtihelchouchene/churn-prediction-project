"""
01_eda.py — Data Loading, Cleaning & Exploratory Analysis
Run: python 01_eda.py
Outputs: cleaned data saved to data/df_clean.parquet
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")          # headless — no GUI window needed
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv("Telecom_customer churn.csv", sep=",")
df = data.copy()
print(f"  Shape: {df.shape}")

# ── 2. Basic checks ────────────────────────────────────────────────────────────
duplicates = df.duplicated().sum()
print(f"  Duplicates: {duplicates}")

df.replace(["NULL", "null", "NaN", "missing", "-"], np.nan, inplace=True)

# ── 3. Missing-value heatmap ───────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap="viridis", yticklabels=False)
plt.title("Heatmap of Missing Values")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.savefig("outputs/eda/missing_heatmap.png", dpi=100)
plt.close()
print("  Saved: outputs/eda/missing_heatmap.png")

# ── 4. Target distribution ─────────────────────────────────────────────────────
print("\nChurn distribution:")
print(df["churn"].value_counts(normalize=True))

# ── 5. Column type split ───────────────────────────────────────────────────────
TARGET   = "churn"
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET)
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

# ── 6. Outlier boxplots (key numerics only) ────────────────────────────────────
num_features = [
    "rev_Mean", "mou_Mean", "totmrc_Mean", "da_Mean",
    "ovrmou_Mean", "ovrrev_Mean", "vceovr_Mean", "roam_Mean",
    "change_mou", "months", "totcalls", "eqpdays",
]
n_cols = 3
n_rows = math.ceil(len(num_features) / n_cols)
plt.figure(figsize=(18, n_rows * 5))
for i, col in enumerate(num_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(data=df, x="churn", y=col, palette="Set2")
    plt.title(f"Boxplot of {col} by Churn", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/eda/outlier_boxplots.png", dpi=100)
plt.close()
print("  Saved: outputs/eda/outlier_boxplots.png")

# ── 7. Log-distribution histograms ────────────────────────────────────────────
print("  Creating log histograms... (this may take a moment)")
# Limit to key numeric columns for speed
key_cols = [col for col in num_cols if col in [
    "rev_Mean", "mou_Mean", "totmrc_Mean", "da_Mean", "ovrmou_Mean", "ovrrev_Mean",
    "vceovr_Mean", "roam_Mean", "change_mou", "months", "eqpdays", "recchu_Mean",
    "adjrev", "adjmou", "monthsit_Mean", "drop_dat_Mean"
]]
grid_cols = 4
grid_rows = math.ceil(len(key_cols) / grid_cols)
print(f"    Grid size: {grid_rows} rows x {grid_cols} cols ({len(key_cols)} key columns)")
fig, axes = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(14, 3 * grid_rows))
axes = axes.flatten()
for i, col in enumerate(key_cols):
    print(f"    Processing column {i+1}/{len(key_cols)}: {col}", end="\r")
    sns.histplot(data=df, x=np.log1p(df[col]), hue=TARGET, bins=20, ax=axes[i])
    axes[i].set_title(f"log({col}) by churn", fontsize=9)
for j in range(len(key_cols), len(axes)):
    fig.delaxes(axes[j])
print("  Saving log_histograms.png...")
plt.tight_layout()
plt.savefig("outputs/eda/log_histograms.png", dpi=100)
plt.close()
print("  Saved: outputs/eda/log_histograms.png")

# ── 8. Correlation heatmap ─────────────────────────────────────────────────────
print("  Creating correlation heatmap...")
# Limit to key columns for readability
heatmap_cols = [col for col in num_cols if col in [
    "rev_Mean", "mou_Mean", "totmrc_Mean", "da_Mean", "ovrmou_Mean", "ovrrev_Mean",
    "vceovr_Mean", "roam_Mean", "change_mou", "months", "eqpdays", "recchu_Mean",
    "adjrev", "adjmou", "monthsit_Mean", "drop_dat_Mean"
]]
plt.figure(figsize=(10, 8))
sns.heatmap(df[heatmap_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix (Key Features)")
plt.tight_layout()
print("  Saving correlation matrix...")
plt.savefig("outputs/eda/correlation_matrix.png", dpi=100)
plt.close()
print("  Saved: outputs/eda/correlation_matrix.png")

# ── 9. Statistical tests ───────────────────────────────────────────────────────
# ── 9. Statistical tests ───────────────────────────────────────────────────────
print("\nRunning t-tests (key numeric features only)...")
print("T-test p-values (numeric vs churn):")
test_cols = [col for col in num_cols if col in [
    "rev_Mean", "mou_Mean", "totmrc_Mean", "da_Mean", "ovrmou_Mean", "ovrrev_Mean",
    "vceovr_Mean", "roam_Mean", "change_mou", "months", "eqpdays"
]]
for col in test_cols:
    g1 = df[df["churn"] == 0][col].dropna()
    g2 = df[df["churn"] == 1][col].dropna()
    _, p = ttest_ind(g1, g2, equal_var=False)
    print(f"  {col}: {p:.4f}")

print("\nRunning chi2 tests (categorical vs churn)...")
print("Chi2 p-values (categorical vs churn):")
# Limit to main categorical columns
cat_test_cols = [col for col in cat_cols if col in df.columns][:10]  # Top 10 categorical cols
for col in cat_test_cols:
    try:
        contingency = pd.crosstab(df[col], df["churn"])
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"  {col}: {p:.4f}")
    except Exception as e:
        print(f"  {col}: Error - {str(e)[:30]}")

# ── 10. Drop low-signal columns ────────────────────────────────────────────────
DROP_COLS = {
    "datovr_Mean", "recv_sms_Mean", "unan_dat_Mean", "callfwdv_Mean",
    "mou_pead_Mean", "drop_dat_Mean", "totmou", "totcalls", "mou_opkd_Mean",
    "avg6qty", "drop_blk_Mean", "attempt_Mean", "complete_Mean",
    "ccrndmou_Mean", "mouowylisv_Mean", "mouiwylisv_Mean", "mou_rvce_Mean",
    "threeway_Mean", "da_Mean", "adjmou", "adjqty", "blck_dat_Mean",
    "totrev", "adjrev", "truck", "rv", "income", "numbcars", "forgntvl",
    "Customer_ID", "new_cell", "dwllsize", "kid3_5", "kid6_10",
    "kid11_15", "kid16_17",
}
df_1 = df.drop(columns=DROP_COLS & set(df.columns))
print(f"\nShape after dropping low-signal columns: {df_1.shape}")

# ── 11. Save ───────────────────────────────────────────────────────────────────
df_1.to_parquet("data/df_clean.parquet", index=False)
print("\n[OK] Saved: data/df_clean.parquet")
print("   Run next: python 02_preprocessing.py")
