"""
05_evaluation.py - Test Evaluation, SHAP, Business Summary
Run: python 05_evaluation.py
Inputs:  data/X_train.parquet, data/X_test.parquet,
         data/y_train.parquet, data/y_test.parquet,
         data/best_xgb.pkl, data/best_lgb.pkl
Outputs: outputs/eval/*.png  (confusion matrices, dashboard, SHAP plots)
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    classification_report, roc_curve, precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs/eval", exist_ok=True)
RANDOM_SEED = 42

# Load data
print("Loading data and tuned models...")
X_tr   = pd.read_parquet("data/X_train.parquet").astype(np.float32)
X_te   = pd.read_parquet("data/X_test.parquet").astype(np.float32)
y_train = pd.read_parquet("data/y_train.parquet").squeeze()
y_test  = pd.read_parquet("data/y_test.parquet").squeeze()

with open("data/best_xgb.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("data/best_lgb.pkl", "rb") as f:
    lgb_model = pickle.load(f)

TUNED_MODELS = {"XGBoost (tuned)": xgb_model, "LightGBM (tuned)": lgb_model}

# Helper function
def best_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    idx = np.argmax(f1s)
    return 0.0 if idx == 0 else thresholds[idx - 1]

# Compute metrics
records, thresholds, probas = {}, {}, {}
print("=" * 65)
print("  TEST SET EVALUATION — TUNED MODELS")
print("=" * 65)

for name, model in TUNED_MODELS.items():
    y_prob       = model.predict_proba(X_te)[:, 1]
    probas[name] = y_prob
    thr          = best_threshold(y_train, model.predict_proba(X_tr)[:, 1])
    thresholds[name] = thr
    y_pred       = (y_prob >= thr).astype(int)

    records[name] = {
        "Model":       name,
        "ROC-AUC":     roc_auc_score(y_test, y_prob),
        "PR-AUC":      average_precision_score(y_test, y_prob),
        "F1":          f1_score(y_test, y_pred),
        "Precision":   precision_score(y_test, y_pred, zero_division=0),
        "Recall":      recall_score(y_test, y_pred),
        "Brier Score": brier_score_loss(y_test, y_prob),
        "Threshold":   round(thr, 3),
    }
    print(f"\n  -- {name}  (threshold = {thr:.3f}) --")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

results = (
    pd.DataFrame(records.values())
    .sort_values("ROC-AUC", ascending=False)
    .reset_index(drop=True)
)
print("\n[DATA] SUMMARY TABLE")
print(results.to_string(index=False))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices - Tuned Models", fontsize=13, fontweight="bold")
for ax, (name, model) in zip(axes, TUNED_MODELS.items()):
    thr    = thresholds[name]
    y_pred = (probas[name] >= thr).astype(int)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], cbar=False)
    ax.set_title(f"{name}\nRecall={tp/(tp+fn):.2%}  Precision={tp/(tp+fp):.2%}", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/eval/confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n  Saved: outputs/eval/confusion_matrices.png")

# ROC + PR + calibration + threshold sensitivity dashboard
COLORS = ["#2196F3", "#FF5722"]
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
for (name, y_prob), color in zip(probas.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color, lw=2)
ax_roc.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
ax_roc.legend(fontsize=8)

ax_pr = fig.add_subplot(gs[0, 1])
base_pr = y_test.mean()
ax_pr.axhline(base_pr, color="k", ls="--", lw=0.8, label=f"Random ({base_pr:.2f})")
for (name, y_prob), color in zip(probas.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax_pr.plot(rec, prec, label=f"{name} ({ap:.3f})", color=color, lw=2)
ax_pr.set(title="Precision-Recall Curves", xlabel="Recall", ylabel="Precision")
ax_pr.legend(fontsize=8)

ax_bar = fig.add_subplot(gs[0, 2])
bar_data = results.set_index("Model")[["ROC-AUC", "PR-AUC", "F1", "Recall"]]
bar_data.plot(kind="barh", ax=ax_bar, colormap="Set2")
ax_bar.set(title="Test Metrics Comparison", xlabel="Score")
ax_bar.axvline(0.5, color="grey", ls="--", lw=0.7)
ax_bar.legend(loc="lower right", fontsize=8)

ax_cal = fig.add_subplot(gs[1, 0])
ax_cal.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
for (name, y_prob), color in zip(probas.items(), COLORS):
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax_cal.plot(mean_pred, frac_pos, "s-", label=name, color=color, ms=5)
ax_cal.set(title="Calibration Curves", xlabel="Mean Predicted Prob",
           ylabel="Fraction of Positives")
ax_cal.legend(fontsize=8)

ax_brier = fig.add_subplot(gs[1, 1])
brier_data = results.set_index("Model")["Brier Score"].sort_values()
brier_data.plot(kind="barh", ax=ax_brier, color="steelblue")
ax_brier.axvline(y_test.mean() * (1 - y_test.mean()), color="red",
                 ls="--", lw=0.8, label="Naive baseline")
ax_brier.set(title="Brier Score  (↓ better)", xlabel="Brier Score")
ax_brier.legend(fontsize=8)

ax_thr = fig.add_subplot(gs[1, 2])
for (name, y_prob), color in zip(probas.items(), COLORS):
    thrs = np.linspace(0.01, 0.99, 100)
    f1s  = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thrs]
    ax_thr.plot(thrs, f1s, label=name, color=color, lw=2)
    ax_thr.axvline(thresholds[name], color=color, ls="--", lw=0.8)
ax_thr.set(title="F1 vs Decision Threshold", xlabel="Threshold", ylabel="F1 Score")
ax_thr.legend(fontsize=8)

plt.suptitle("Tuned Models - Full Evaluation Dashboard", fontsize=14, y=1.01)
plt.savefig("outputs/eval/evaluation_dashboard.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: outputs/eval/evaluation_dashboard.png")

# Best model selection
weights = {"ROC-AUC": 0.35, "PR-AUC": 0.25, "F1": 0.20, "Recall": 0.15, "Brier Score": 0.05}
results["Composite Score"] = (
    results["ROC-AUC"]   * weights["ROC-AUC"]  +
    results["PR-AUC"]    * weights["PR-AUC"]   +
    results["F1"]        * weights["F1"]        +
    results["Recall"]    * weights["Recall"]    -
    results["Brier Score"] * weights["Brier Score"]
)
results_final = results.sort_values("Composite Score", ascending=False).reset_index(drop=True)
best_name  = results_final.iloc[0]["Model"]
best_model = TUNED_MODELS[best_name]
best_thr   = thresholds[best_name]

print("\n" + "=" * 65)
print(f"  🏆  Best Model: {best_name}")
print(f"      Threshold : {best_thr:.3f}")
print(f"      ROC-AUC   : {results_final.iloc[0]['ROC-AUC']:.4f}")
print(f"      Recall    : {results_final.iloc[0]['Recall']:.4f}")
print(f"      F1        : {results_final.iloc[0]['F1']:.4f}")

# SHAP global feature importance
print(f"\n[CONFIG] Computing SHAP values for {best_name}...")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer(X_te, check_additivity=False)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
feature_names = X_te.columns.tolist()

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
plt.sca(axes[0])
shap.plots.bar(sv, max_display=20, show=False)
axes[0].set_title(f"Top 20 Features — Mean |SHAP|\n{best_name}", fontsize=12, fontweight="bold")
plt.sca(axes[1])
shap.plots.beeswarm(sv, max_display=20, show=False)
axes[1].set_title(f"SHAP Beeswarm — Impact Direction\n{best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eval/shap_global.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: outputs/eval/shap_global.png")

# SHAP dependence (top 3 features)
mean_shap  = np.abs(sv.values).mean(axis=0)
top3_idx   = np.argsort(mean_shap)[::-1][:3]
top3_feats = [feature_names[i] for i in top3_idx]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f"SHAP Dependence Plots - Top 3 Features ({best_name})",
             fontsize=13, fontweight="bold")
for ax, feat in zip(axes, top3_feats):
    shap.plots.scatter(sv[:, feat], ax=ax, show=False)
    ax.set_title(feat, fontsize=10)
plt.tight_layout()
plt.savefig("outputs/eval/shap_dependence_top3.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: outputs/eval/shap_dependence_top3.png")

# SHAP local waterfall
y_prob_best = probas[best_name]
for idx, label in [(int(np.argmax(y_prob_best)), "High-Risk Churner"),
                   (int(np.argmin(y_prob_best)), "Safe Customer")]:
    plt.figure(figsize=(14, 5))
    shap.plots.waterfall(sv[idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall - {label}  (prob={y_prob_best[idx]:.3f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = f"outputs/eval/shap_local_{'churner' if 'Risk' in label else 'safe'}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# Business impact summary
y_pred_best = (y_prob_best >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
print("\n" + "=" * 65)
print("  BUSINESS IMPACT SUMMARY")
print("=" * 65)
print(f"\n  Model          : {best_name}")
print(f"  Test set size  : {len(y_test):,} customers")
print(f"\n  ✅ Churners correctly flagged (TP) : {tp:,}")
print(f"  ❌ Churners missed           (FN) : {fn:,}  ← revenue at risk")
print(f"  [!] Non-churners flagged      (FP) : {fp:,}  <- wasted retention spend")
print(f"  ✅ Non-churners ignored       (TN) : {tn:,}")
print(f"\n  Churn Recall   : {tp/(tp+fn):.2%}")
print(f"  Churn Precision: {tp/(tp+fp):.2%}")
print(f"\n  👉 At threshold {best_thr:.3f}, model catches {tp/(tp+fn):.0%} of churners")
print(f"     with {fp/(fp+tn):.0%} false-alarm rate on safe customers.")
print("\n[OK] All evaluation outputs saved to outputs/eval/")
