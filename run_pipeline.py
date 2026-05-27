"""
run_pipeline.py — Execute all pipeline steps in order
Usage:
    python run_pipeline.py             # run all 5 steps
    python run_pipeline.py --from 3   # resume from step 3
"""


import subprocess
import sys
import time

STEPS = [
    ("01_eda.py",          "EDA & Cleaning"),
    ("02_preprocessing.py","Imputation, Engineering, Encoding, VIF"),
    ("03_baseline_cv.py",  "Baseline Cross-Validation"),
    ("04_tuning.py",       "Hyperparameter Tuning"),
    ("05_evaluation.py",   "Evaluation & SHAP"),
]

start_from = 1
if "--from" in sys.argv:
    idx = sys.argv.index("--from")
    start_from = int(sys.argv[idx + 1])

# ── Python executable — override with --python if needed ──────────────────────
python_exe = sys.executable   # default: whatever ran this script
if "--python" in sys.argv:
    idx = sys.argv.index("--python")
    python_exe = sys.argv[idx + 1]
print(f"Using Python: {python_exe}")

print("=" * 65)
print("  CHURN PREDICTION PIPELINE")
print("=" * 65)

for i, (script, desc) in enumerate(STEPS, start=1):
    if i < start_from:
        print(f"\n  [SKIP] Step {i} skipped  ({desc})")
        continue
    print(f"\n{'=' * 65}")
    print(f"  STEP {i}/{len(STEPS)} — {desc}")
    print(f"  Script: {script}")
    print("=" * 65)
    t0 = time.time()
    result = subprocess.run([python_exe, script], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Step {i} failed (exit code {result.returncode}). Pipeline halted.")
        sys.exit(result.returncode)
    print(f"\n  [OK] Step {i} done in {elapsed:.1f}s")

print("\n" + "=" * 65)
print("  [DONE] PIPELINE COMPLETE")
print("  Check outputs/ for all plots and data/ for saved artefacts.")
print("=" * 65)