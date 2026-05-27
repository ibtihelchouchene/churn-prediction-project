"""utils/shap_utils.py — SHAP computations with caching."""

import numpy as np
import pandas as pd
from functools import lru_cache

_shap_cache = {}   # keyed by model id

def get_shap_explainer(model):
    mid = id(model)
    if mid not in _shap_cache:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            _shap_cache[mid] = explainer
        except Exception as e:
            print(f"[shap] TreeExplainer failed: {e}. Trying LinearExplainer.")
            try:
                import shap
                explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(data=None))
                _shap_cache[mid] = explainer
            except Exception as e2:
                print(f"[shap] All explainers failed: {e2}")
                _shap_cache[mid] = None
    return _shap_cache[mid]


def compute_global_shap(model, X_test, max_rows=500):
    """Compute global SHAP values (sampled for speed)."""
    try:
        import shap
        explainer = get_shap_explainer(model)
        if explainer is None:
            return None, None
        X_sample = X_test.iloc[:max_rows] if len(X_test) > max_rows else X_test
        sv = explainer(X_sample, check_additivity=False)
        # Handle list output (older shap)
        if isinstance(sv, list):
            sv = sv[1]
        mean_shap = np.abs(sv.values).mean(axis=0)
        df = pd.DataFrame({
            "feature":    X_test.columns,
            "mean_shap":  mean_shap,
        }).sort_values("mean_shap", ascending=False).head(20)
        return df, sv
    except Exception as e:
        print(f"[shap] global compute failed: {e}")
        return None, None


def compute_local_shap(model, X_row, feature_names):
    """Compute SHAP values for a single customer row."""
    try:
        import shap
        explainer = get_shap_explainer(model)
        if explainer is None:
            return None
        sv = explainer(X_row, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        values = sv.values[0] if sv.values.ndim > 1 else sv.values
        df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": values,
        }).sort_values("shap_value", key=abs, ascending=False).head(15)
        return df
    except Exception as e:
        print(f"[shap] local compute failed: {e}")
        return None
