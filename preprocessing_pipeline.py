"""
preprocessing_pipeline.py — Sklearn-compatible preprocessing pipeline
for the telecom churn project.

Exports
-------
build_preprocessing_pipeline(high_card_cats, binary_maps)
    Returns a fitted-ready sklearn Pipeline whose steps are:
      imputer → engineer → clipper → binary_maps →
      target_enc → ordinal_enc → vif_dropper

Usage
-----
    preprocessor = build_preprocessing_pipeline()

    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc  = preprocessor.transform(X_test)

    import joblib
    joblib.dump(preprocessor, "preprocessor.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONSTANTS  (aligned with the actual telecom dataset)
# =============================================================================

RANDOM_SEED = 42
EPS = 1e-6          # match script (1e-6, not 1e-9) to keep feature values identical

RATIO_KEYWORDS = (
    "_rate", "_ratio", "_share", "_momentum", "_stability",
    "per_minute", "per_sub", "per_call", "ltv",
)

# --- Corrected to match the script's dataset columns ---
BINARY_MAPS: dict[str, dict] = {
    "asl_flag": {"Y": 1, "N": 0},
    "creditcd": {"Y": 1, "N": 0},
    "ownrent":  {"R": 0, "U": 0, "O": 1},
    "dualband": {"Y": 1, "N": 0, " ": 0},
}

HIGH_CARD_CATS: list[str] = [
    "prizm_social_one", "area", "crclscod", "marital",
    "dwlltype", "refurb_new", "hnd_webcap",
]


# =============================================================================
# UTILITIES
# =============================================================================

def _to_frame(X, columns=None) -> pd.DataFrame:
    """Coerce ndarray / Series to DataFrame, preserving column names."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=columns)
    return pd.DataFrame(X)


# =============================================================================
# 1. MIXED IMPUTER
# =============================================================================

class MixedImputer(BaseEstimator, TransformerMixin):
    """Median imputation for numerics, most-frequent for categoricals."""

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if self.num_cols_:
            self.num_imp_ = SimpleImputer(strategy="median").fit(X[self.num_cols_])
        if self.cat_cols_:
            self.cat_imp_ = SimpleImputer(strategy="most_frequent").fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X = _to_frame(X).copy()
        if self.num_cols_:
            X[self.num_cols_] = self.num_imp_.transform(X[self.num_cols_])
        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_imp_.transform(X[self.cat_cols_])
        return X


# =============================================================================
# 2. FEATURE ENGINEER
# =============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer — adds all derived ratio/flag features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_frame(X).copy()

        # ── Service-quality rates ──────────────────────────────────────────────
        X["call_completion_rate"]    = X["comp_vce_Mean"]  / (X["plcd_vce_Mean"] + EPS)
        X["drop_rate"]               = X["drop_vce_Mean"]   / (X["plcd_vce_Mean"] + EPS)
        X["block_rate"]              = X["blck_vce_Mean"]   / (X["plcd_vce_Mean"] + EPS)
        X["short_call_ratio"]        = X["inonemin_Mean"]   / (X["plcd_vce_Mean"] + EPS)
        X["dropped_or_blocked_rate"] = (X["drop_vce_Mean"] + X["blck_vce_Mean"]) / (X["plcd_vce_Mean"] + EPS)
        X["unanswered_rate"]         = X["unan_vce_Mean"]   / (X["plcd_vce_Mean"] + EPS)
        X["data_to_voice_ratio"]     = X["plcd_dat_Mean"]   / (X["plcd_vce_Mean"] + EPS)
        X["voice_mou_per_call"]      = X["mou_cvce_Mean"]   / (X["comp_vce_Mean"] + EPS)

        # ── Usage & value ratios ───────────────────────────────────────────────
        X["revenue_per_minute"]      = X["rev_Mean"]        / (X["mou_Mean"]      + EPS)
        X["overage_ratio"]           = X["ovrmou_Mean"]     / (X["mou_Mean"]      + EPS)
        X["custcare_rate"]           = X["custcare_Mean"]   / (X["months"]        + EPS)
        X["peak_usage_share"]        = X["peak_vce_Mean"]   / (X["peak_vce_Mean"] + X["opk_vce_Mean"] + EPS)
        X["roam_to_total_ratio"]     = X["roam_Mean"]       / (X["plcd_vce_Mean"] + EPS)
        X["revenue_per_subscriber"]  = X["rev_Mean"]        / (X["actvsubs"]      + EPS)
        X["customer_ltv"]            = X["totmrc_Mean"]     * X["months"]

        # ── Trend & trajectory ─────────────────────────────────────────────────
        X["mou_momentum"]            = X["avg3mou"]         / (X["avg6mou"]       + EPS)
        X["rev_momentum"]            = X["avg3rev"]         / (X["avg6rev"]       + EPS)
        X["lifetime_vs_recent_mou"]  = X["avg3mou"]         / (X["avgmou"]        + EPS)
        X["revenue_stability"]       = X["avg3rev"]         / (X["avgrev"]        + EPS)

        # ── Subscription & device ──────────────────────────────────────────────
        X["active_sub_ratio"]        = X["actvsubs"]        / (X["uniqsubs"]      + EPS)
        X["models_per_phone"]        = X["models"]          / (X["phones"]        + EPS)

        # Device age tier — stored as plain int to avoid nullable-Int64 issues
        X["device_age_tier"] = pd.cut(
            X["eqpdays"],
            bins=[-np.inf, 180, 365, 730, np.inf],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(float)       # float avoids pd.NA propagation downstream

        # ── Binary trend flags ────────────────────────────────────────────────
        X["mou_declining_flag"]     = (X["change_mou"] < -5).astype(int)
        X["revenue_declining_flag"] = (X["change_rev"] < -5).astype(int)

        return X


# =============================================================================
# 3. PERCENTILE CLIPPER
# =============================================================================

class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip ratio/rate features at train-set 1st–99th percentiles."""

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        keywords: tuple = RATIO_KEYWORDS,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.keywords = keywords

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.ratio_cols_ = [
            c for c in X.columns if any(kw in c for kw in self.keywords)
        ]
        self.bounds_ = {
            col: (
                float(X[col].quantile(self.lower_quantile)),
                float(X[col].quantile(self.upper_quantile)),
            )
            for col in self.ratio_cols_
        }
        return self

    def transform(self, X):
        X = _to_frame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X


# =============================================================================
# 4. BINARY MAP ENCODER
# =============================================================================

class BinaryMapEncoder(BaseEstimator, TransformerMixin):
    """Map known binary/ordinal categoricals to integers."""

    def __init__(self, maps: dict | None = None):
        self.maps = maps or BINARY_MAPS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_frame(X).copy()
        for col, mapping in self.maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0).astype(int)
        return X


# =============================================================================
# 5. TARGET ENCODER (OOF on train, smoothed map on test)
# =============================================================================

class TargetEncoderOOF(BaseEstimator, TransformerMixin):
    """
    Leak-free target encoding.

    • fit_transform  → out-of-fold smoothed encoding (used on train set)
    • transform      → full-training smoothed map     (used on test set)
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        k: int = 20,
        n_folds: int = 5,
        random_state: int = RANDOM_SEED,
    ):
        self.cols = cols or HIGH_CARD_CATS
        self.k = k
        self.n_folds = n_folds
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _build_smooth_map(self, col_series: pd.Series, y: pd.Series, global_mean: float) -> pd.Series:
        df = pd.concat([col_series.rename("col"), y.rename("y")], axis=1)
        agg = df.groupby("col")["y"].agg(["mean", "count"])
        return (agg["count"] * agg["mean"] + self.k * global_mean) / (agg["count"] + self.k)

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoderOOF requires y during fit.")
        X = _to_frame(X).reset_index(drop=True)
        y = pd.Series(y, name="y").reset_index(drop=True)

        self.global_mean_ = float(y.mean())
        self.smooth_maps_ = {
            col: self._build_smooth_map(X[col], y, self.global_mean_)
            for col in self.cols if col in X.columns
        }
        self.encoded_cols_ = list(self.smooth_maps_.keys())   # cols actually present
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        """Test-set path: use full-training smoothed maps."""
        check_is_fitted(self, "smooth_maps_")
        X = _to_frame(X).copy()
        for col, smap in self.smooth_maps_.items():
            if col not in X.columns:
                continue
            X[f"{col}_enc"] = X[col].map(smap).fillna(self.global_mean_)
            X = X.drop(columns=[col])
        return X

    # ------------------------------------------------------------------
    def fit_transform(self, X, y=None, **fit_params):
        """
        Train-set path: out-of-fold encoding to prevent target leakage.
        sklearn Pipeline calls this on every intermediate step.
        """
        if y is None:
            raise ValueError("TargetEncoderOOF.fit_transform() requires y.")
        X = _to_frame(X).reset_index(drop=True).copy()
        y = pd.Series(y, name="y").reset_index(drop=True)

        self.fit(X, y)          # populate smooth_maps_ on full training data

        # Initialise OOF arrays with global mean
        oof: dict[str, np.ndarray] = {
            col: np.full(len(X), self.global_mean_, dtype=float)
            for col in self.encoded_cols_
        }

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for fit_idx, val_idx in skf.split(X, y):
            y_fold     = y.iloc[fit_idx]
            gm_fold    = float(y_fold.mean())
            for col in oof:
                smap_fold       = self._build_smooth_map(X[col].iloc[fit_idx], y_fold, gm_fold)
                oof[col][val_idx] = (
                    X[col].iloc[val_idx].map(smap_fold).fillna(gm_fold).values
                )

        for col, arr in oof.items():
            X[f"{col}_enc"] = arr
            X = X.drop(columns=[col])

        return X


# =============================================================================
# 6. ORDINAL ENCODER (remaining object / category columns)
# =============================================================================

class OrdinalCategoricalEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.cat_cols_:
            self.enc_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ).fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X = _to_frame(X).copy()
        if self.cat_cols_:
            X[self.cat_cols_] = self.enc_.transform(X[self.cat_cols_])
        return X


# =============================================================================
# 7. VIF DROPPER  (optional; fits on train, applies same column mask to test)
# =============================================================================

class VIFDropper(BaseEstimator, TransformerMixin):
    """
    Iteratively drops the column with the highest VIF until all VIFs
    are below `threshold`.  Sampling is used for speed.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        sample_rows: int = 5_000,
        random_state: int = RANDOM_SEED,
    ):
        self.threshold = threshold
        self.sample_rows = sample_rows
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_vif(df: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = df.select_dtypes(include=[np.number]).astype(np.float64).dropna(axis=1)
        if X.empty or X.shape[1] < 2:
            return pd.DataFrame(columns=["feature", "VIF"])
        return (
            pd.DataFrame(
                {"feature": X.columns,
                 "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]}
            )
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = _to_frame(X).copy()
        if X.shape[0] > self.sample_rows:
            X = X.sample(n=self.sample_rows, random_state=self.random_state)

        self.dropped_cols_: list[str] = []
        while True:
            vif = self._compute_vif(X)
            if vif.empty:
                break
            worst = vif.iloc[0]
            if worst["VIF"] > self.threshold:
                col = worst["feature"]
                X = X.drop(columns=[col])
                self.dropped_cols_.append(col)
                print(f"  [VIF] Dropped '{col}'  (VIF={worst['VIF']:.1f})")
            else:
                break

        self.keep_cols_ = X.columns.tolist()
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        check_is_fitted(self, "keep_cols_")
        X = _to_frame(X)
        # Only keep columns that exist in both
        cols = [c for c in self.keep_cols_ if c in X.columns]
        return X[cols]


# =============================================================================
# PIPELINE FACTORY
# =============================================================================

def build_preprocessing_pipeline(
    high_card_cats: list[str] | None = None,
    binary_maps: dict | None = None,
    vif_threshold: float | None = 10.0,
) -> Pipeline:
    """
    Build the full preprocessing pipeline.

    Parameters
    ----------
    high_card_cats : list[str], optional
        High-cardinality categorical columns for target encoding.
        Defaults to ``HIGH_CARD_CATS``.
    binary_maps : dict, optional
        Column → {category: int} mapping for binary encoding.
        Defaults to ``BINARY_MAPS``.
    vif_threshold : float or None
        VIF threshold for the VIFDropper step.
        Pass ``None`` to skip VIF dropping entirely.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = [
        ("imputer",     MixedImputer()),
        ("engineer",    FeatureEngineer()),
        ("clipper",     PercentileClipper()),
        ("binary_maps", BinaryMapEncoder(maps=binary_maps or BINARY_MAPS)),
        ("target_enc",  TargetEncoderOOF(cols=high_card_cats or HIGH_CARD_CATS)),
        ("ordinal_enc", OrdinalCategoricalEncoder()),
    ]

    if vif_threshold is not None:
        steps.append(("vif_dropper", VIFDropper(threshold=vif_threshold)))

    return Pipeline(steps)