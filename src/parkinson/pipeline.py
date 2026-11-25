"""Learner declarations for the Parkinson's MDS-UPDRS motor score task.

Owns the functions that build and return unfit learners, plus the
``PatientFeatures`` transformer that derives per-patient longitudinal
features before the final estimator, ``EnvelopeFeatures`` which
uses the upper-envelope signal to derive normalised visit scores, and
``StackedEnvelopeRegressor`` which stacks a trajectory-level stage-1
model with a visit-deviation stage-2 model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline


# Columns excluded from model features: row identifier and patient
# grouping key (patient_id leaks patient identity to the model).
_DROP_COLS = ["Index", "patient_id"]

# Columns for which a 0/1 missing flag is added.
_MISSING_FLAG_COLS = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]

# Columns over which per-patient aggregates (mean/median/min/max/count) are computed.
_AGG_COLS = ["off", "on", "ledd"]

# Categorical columns whose levels are learnt in fit().
_CAT_COLS = ["cohort", "gene", "rater_id"]


def _fit_patient_line(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute per-patient OLS line of ``col`` vs ``age`` within ``df``.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id``, ``age``, and ``col``.
    col :
        Name of the y-series column.

    Returns
    -------
    slope : pd.Series
        Per-row OLS slope (0 when fewer than 2 valid observations).
    intercept : pd.Series
        Per-row OLS intercept.
    line_val : pd.Series
        Predicted value at this visit's age (= intercept + slope * age).
    """
    col_mean_series = df.get(f"{col}_mean")

    valid = df[["patient_id", "age", col]].copy()
    valid = valid.rename(columns={col: "y_val"})
    valid["y_val_valid"] = valid["y_val"].notna() & valid["age"].notna()

    valid["x_v"] = valid["age"].where(valid["y_val_valid"])
    valid["y_v"] = valid["y_val"].where(valid["y_val_valid"])
    valid["x2_v"] = (valid["x_v"] ** 2).where(valid["y_val_valid"])
    valid["xy_v"] = (valid["x_v"] * valid["y_v"]).where(valid["y_val_valid"])
    valid["n_v"] = valid["y_val_valid"].astype(float)

    grp = valid.groupby("patient_id", sort=False)
    n_v = grp["n_v"].transform("sum")
    sx = grp["x_v"].transform("sum")
    sy = grp["y_v"].transform("sum")
    sx2 = grp["x2_v"].transform("sum")
    sxy = grp["xy_v"].transform("sum")

    denom = n_v * sx2 - sx ** 2
    slope = (n_v * sxy - sx * sy) / denom.replace(0, np.nan)
    intercept = (sy - slope * sx) / n_v

    fewer_than_2 = n_v < 2
    slope = slope.where(~fewer_than_2, 0.0)
    if col_mean_series is not None:
        intercept = intercept.where(~fewer_than_2, col_mean_series)
    else:
        intercept = intercept.where(~fewer_than_2, np.nan)

    line_val = intercept + slope * df["age"]
    if col_mean_series is not None:
        line_val = line_val.fillna(col_mean_series)

    return slope, intercept, line_val


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop identifier and grouping columns before fitting.

    ``Index`` is the row identifier required only for the submission
    CSV. ``patient_id`` is the grouping key for ``GroupKFold`` and
    must not be exposed to the model because it would leak patient
    identity, causing overfitting to seen patients.

    Parameters
    ----------
    columns :
        List of column names to drop. Defaults to ``["Index",
        "patient_id"]``.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns if columns is not None else _DROP_COLS

    def fit(self, X: pd.DataFrame, y=None):
        """No-op fit: record that fit was called so ``check_is_fitted`` passes."""
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the configured columns from ``X``."""
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class PatientFeatures(BaseEstimator, TransformerMixin):
    """Add longitudinal per-patient features derived from the visit history.

    All features are computed purely from X columns (never from the
    target), so the transformer runs identically on X_test. The whole
    table is always transformed in one call because the features are
    per-patient (e.g. visit_rank, n_visits, previous/next off).

    ``fit`` learns only the category levels of ``cohort``, ``gene``,
    and ``rater_id`` so that ``transform`` can safely re-encode them
    as ordered pandas Categoricals on unseen slices.

    When ``rater_adjust=True``, ``fit`` also learns a per-rater bias
    for off and on (``rater_bias_``), and ``transform`` adds rater-bias
    features, adjusted series, and their per-patient OLS lines.

    Added features (per-patient, sorted by age within patient):
    tsd              : age - age_at_diagnosis
    visit_rank       : 0-based rank of this visit within the patient
    n_visits         : total number of visits for this patient
    age_since_first  : age minus the patient's first observed age
    {col}_missing    : 0/1 flag for NaN in off/on/ledd/time_since_intake_on/off
    {col}_mean, _median, _min, _max, _count : per-patient aggregates for off/on/ledd
    off_slope, off_line_val : least-squares slope of off vs age; predicted value
                              at this visit's age (patient mean when < 2 obs)
    on_slope, on_line_val   : same for on vs age
    prev_off, next_off      : previous and next observed off within the patient
    prev_on, next_on        : previous and next observed on within the patient
    age_gap_prev, age_gap_next : age difference to adjacent visits

    When ``rater_adjust=True``, additionally:
    off_rater_bias, on_rater_bias : per-rater mean deviation from patient OLS line
    off_adj, on_adj               : off/on minus their rater bias
    off_adj_mean, on_adj_mean     : per-patient mean of the adjusted series
    off_adj_slope, off_adj_line_val : OLS line of off_adj vs age
    on_adj_slope, on_adj_line_val   : OLS line of on_adj vs age
    off_dev                       : off - off_line_val
    on_dev                        : on - on_line_val

    ``Index`` and ``patient_id`` are dropped at the end of transform.

    Parameters
    ----------
    rater_adjust : bool, default False
        When True, learn and apply per-rater bias correction.
    """

    def __init__(self, rater_adjust: bool = False) -> None:
        self.rater_adjust = rater_adjust

    def fit(self, X: pd.DataFrame, y=None):
        """Learn category levels (and optionally per-rater bias) from training data.

        Parameters
        ----------
        X :
            Training feature matrix (must contain cohort, gene, rater_id).
        y :
            Ignored.

        Returns
        -------
        self
        """
        self.cat_levels_: dict[str, pd.Index] = {}
        for col in _CAT_COLS:
            if col in X.columns:
                self.cat_levels_[col] = X[col].astype("category").cat.categories

        if self.rater_adjust and "rater_id" in X.columns:
            self.rater_bias_: dict[str, dict] = {}
            df = X.copy()
            df = df.sort_values(["patient_id", "age"])

            # Compute per-patient aggregates for off/on so _fit_patient_line works.
            for col in ["off", "on"]:
                if col in df.columns:
                    df[f"{col}_mean"] = df.groupby("patient_id", sort=False)[col].transform("mean")

            for col in ["off", "on"]:
                if col not in df.columns:
                    continue
                _, _, line_val = _fit_patient_line(df, col)
                # deviation = actual - line_val (NaN where col is NaN)
                deviation = df[col] - line_val.values
                tmp = pd.DataFrame({"rater_id": df["rater_id"], "dev": deviation})
                self.rater_bias_[col] = tmp.groupby("rater_id")["dev"].mean().to_dict()
        else:
            # Always set a trailing-underscore attribute in fit.
            self.rater_bias_: dict[str, dict] = {}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive longitudinal patient features and return the enriched table.

        Parameters
        ----------
        X :
            Feature matrix with at least ``patient_id`` and ``age``.

        Returns
        -------
        pd.DataFrame
            One row per input row, same order as input, with
            ``Index`` and ``patient_id`` dropped.
        """
        df = X.copy()

        # Re-encode categoricals with the levels seen in fit so that
        # pipeline slices (used by skore's checks) stay consistent.
        for col, levels in self.cat_levels_.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=levels)

        # Sort within each patient by age; keep original index to restore row order.
        orig_index = df.index
        df = df.sort_values(["patient_id", "age"])

        # --- simple scalar features per visit ---
        df["tsd"] = df["age"] - df["age_at_diagnosis"]

        # --- visit_rank and n_visits (vectorised groupby) ---
        df["visit_rank"] = df.groupby("patient_id", sort=False).cumcount()
        df["n_visits"] = df.groupby("patient_id", sort=False)["age"].transform("count")

        # --- age since first visit ---
        df["age_since_first"] = df["age"] - df.groupby("patient_id", sort=False)["age"].transform("min")

        # --- missing flags ---
        for col in _MISSING_FLAG_COLS:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(np.int8)

        # --- per-patient aggregates for off / on / ledd ---
        for col in _AGG_COLS:
            if col not in df.columns:
                continue
            grp = df.groupby("patient_id", sort=False)[col]
            df[f"{col}_mean"] = grp.transform("mean")
            df[f"{col}_median"] = grp.transform("median")
            df[f"{col}_min"] = grp.transform("min")
            df[f"{col}_max"] = grp.transform("max")
            df[f"{col}_count"] = grp.transform("count")

        # --- least-squares line of off/on vs age within patient ---
        for col in ["off", "on"]:
            if col not in df.columns:
                continue
            slope, _, line_val = _fit_patient_line(df, col)
            df[f"{col}_slope"] = slope.values
            df[f"{col}_line_val"] = line_val.values

        # --- rater-bias adjustment (only when rater_adjust=True) ---
        if self.rater_adjust and self.rater_bias_:
            for col in ["off", "on"]:
                if col not in df.columns:
                    continue
                bias_map = self.rater_bias_.get(col, {})
                rater_bias = df["rater_id"].astype(object).map(bias_map).fillna(0.0)
                df[f"{col}_rater_bias"] = rater_bias.values

                adj_col = f"{col}_adj"
                df[adj_col] = df[col].astype(float) - rater_bias.values

                # Per-patient mean of the adjusted series.
                df[f"{adj_col}_mean"] = df.groupby("patient_id", sort=False)[adj_col].transform("mean")

                # OLS line of the adjusted series vs age.
                slope_adj, _, line_val_adj = _fit_patient_line(df, adj_col)
                df[f"{adj_col}_slope"] = slope_adj.values
                df[f"{adj_col}_line_val"] = line_val_adj.values

            # --- deviation from the OLS line for raw off/on ---
            for col in ["off", "on"]:
                if col in df.columns and f"{col}_line_val" in df.columns:
                    df[f"{col}_dev"] = df[col].astype(float) - df[f"{col}_line_val"]

        # --- previous/next off and on within patient, and age gaps ---
        for col in ["off", "on"]:
            if col in df.columns:
                grp = df.groupby("patient_id", sort=False)[col]
                df[f"prev_{col}"] = grp.shift(1)
                df[f"next_{col}"] = grp.shift(-1)

        age_grp = df.groupby("patient_id", sort=False)["age"]
        df["age_gap_prev"] = df["age"] - age_grp.shift(1)
        df["age_gap_next"] = age_grp.shift(-1) - df["age"]

        # Restore original row order, then drop identifiers.
        df = df.loc[orig_index]
        df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

        return df


class PatientSmoothedRegressor(BaseEstimator, RegressorMixin):
    """Regressor that post-processes predictions with a per-patient polynomial smoother.

    ``fit`` clones and fits ``base_estimator``. ``predict`` generates raw
    predictions from ``base_estimator_``, then fits a polynomial of
    ``degree`` in (age − patient_mean_age) to those predictions for each
    patient and returns the fitted polynomial values in the original row order.

    The polynomial degree is automatically reduced to
    ``n_distinct_ages − 1`` when a patient has fewer distinct ages than
    ``degree + 1``, so single-visit patients are handled gracefully (degree 0,
    i.e. a constant equal to the raw prediction).

    Parameters
    ----------
    base_estimator :
        Any sklearn-compatible regressor.
    degree : int, default 2
        Degree of the smoothing polynomial.
    """

    def __init__(self, base_estimator, degree: int = 2) -> None:
        self.base_estimator = base_estimator
        self.degree = degree

    def fit(self, X: pd.DataFrame, y):
        """Fit the base estimator.

        Parameters
        ----------
        X :
            Feature matrix (must contain ``patient_id`` and ``age`` before
            any internal transformers strip them).
        y :
            Target vector.

        Returns
        -------
        self
        """
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate smoothed predictions.

        Parameters
        ----------
        X :
            Feature matrix with ``patient_id`` and ``age`` present.

        Returns
        -------
        np.ndarray
            Smoothed predictions in the same row order as ``X``.
        """
        raw_pred = self.base_estimator_.predict(X)

        # Build a DataFrame for per-patient polynomial smoothing.
        tmp = pd.DataFrame(
            {"patient_id": X["patient_id"].values, "age": X["age"].values, "pred": raw_pred},
            index=np.arange(len(X)),
        )

        smoothed = np.empty(len(tmp), dtype=float)
        for pid, grp in tmp.groupby("patient_id", sort=False):
            ages = grp["age"].values
            preds = grp["pred"].values
            n_distinct = len(np.unique(ages))
            deg = min(self.degree, n_distinct - 1)
            # Centre ages to improve numerical stability.
            age_centre = ages.mean()
            ages_c = ages - age_centre
            if deg < 1:
                smoothed[grp.index] = preds
            else:
                coeffs = np.polyfit(ages_c, preds, deg)
                smoothed[grp.index] = np.polyval(coeffs, ages_c)

        return smoothed


def build_learner(**kwargs) -> Pipeline:
    """Return the unfit ``20_hgb_baseline`` learner (DropColumns → HGB).

    Parameters
    ----------
    **kwargs :
        Ignored; accepted for API compatibility.
    """
    return Pipeline(
        [
            ("drop", DropColumns()),
            ("hgb", HistGradientBoostingRegressor(random_state=0)),
        ]
    )


def build_patient_learner(**kwargs) -> Pipeline:
    """Return the unfit ``21_hgb_patient`` learner (PatientFeatures → HGB).

    The pipeline accepts X with ``patient_id`` present; ``PatientFeatures``
    derives all longitudinal features and drops ``Index`` / ``patient_id``
    before the regressor sees the data.

    Parameters
    ----------
    **kwargs :
        Ignored; accepted for API compatibility.
    """
    return make_pipeline(
        PatientFeatures(),
        HistGradientBoostingRegressor(random_state=0),
    )


def build_rater_smooth_learner(**kwargs) -> PatientSmoothedRegressor:
    """Return the unfit ``22_hgb_rater_smooth`` learner.

    Architecture: PatientSmoothedRegressor wrapping
    PatientFeatures(rater_adjust=True) → HGB with tuned hyperparameters.
    The outer smoother fits a per-patient degree-2 polynomial to the base
    predictions, reducing high-frequency noise while preserving the
    disease-progression trend.

    Parameters
    ----------
    **kwargs :
        Ignored; accepted for API compatibility.
    """
    return PatientSmoothedRegressor(
        make_pipeline(
            PatientFeatures(rater_adjust=True),
            HistGradientBoostingRegressor(
                random_state=0,
                learning_rate=0.03,
                max_iter=2000,
                early_stopping=True,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
            ),
        )
    )


# ---------------------------------------------------------------------------
# EnvelopeFeatures
# ---------------------------------------------------------------------------

# Bin edges (in hours) for time_since_intake_on.
_TSI_BINS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, np.inf]

# Default projection slope used for e_off and e_on projections.
_PROJ_SLOPE = 2.5


def _bin_tsi(tsi: pd.Series) -> pd.Series:
    """Map ``time_since_intake_on`` to an integer bin label (NaN → -1).

    Bin edges: [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, ∞).
    Missing values receive the sentinel label ``-1`` (separate bin).

    Parameters
    ----------
    tsi :
        Series of time-since-intake-on values (hours, may contain NaN).

    Returns
    -------
    pd.Series of int8, same index as ``tsi``.
    """
    binned = pd.cut(
        tsi,
        bins=_TSI_BINS,
        labels=False,
        right=False,
        include_lowest=True,
    ).astype("Int8")
    return binned.fillna(-1).astype(np.int8)


class EnvelopeFeatures(BaseEstimator, TransformerMixin):
    """Derive upper-envelope normalised visit scores from off/on measurements.

    Clinical rationale
    ------------------
    Each rater scores on a personal multiplicative scale; the measured ``off``
    is the true OFF score reduced by a rater-specific factor.  The measured
    ``on`` is the true OFF score further reduced by the pharmacological effect,
    whose magnitude depends on the delay since drug intake (``time_since_intake_on``).
    Visit noise is one-sided: a corrected score is rarely *above* the true score
    but often well below, so the true score follows the **upper envelope** of a
    patient's corrected scores, not their mean.  Because ``on`` and ``off`` of
    the same visit share the same noise, their ratio is stable within a patient.

    What is learnt in ``fit`` (on rows where ``y > 10``)
    -----------------------------------------------------
    rater_scale_ :
        Per-rater 75th percentile of ``off / y``, divided by the median of
        these percentiles across raters.  Unknown rater → 1.0.
    on_curve_ :
        Median of ``on / y / rater_scale`` per bin of ``time_since_intake_on``
        (11 finite bins + 1 missing-value bin).  Maps bin label → median ratio.
    off_top_ :
        75th percentile of ``off / y / rater_scale`` (one global scalar).

    Also fits an internal :class:`PatientFeatures` transformer.

    Features added by ``transform`` (X is never modified)
    -------------------------------------------------------
    All :class:`PatientFeatures` columns, plus:

    * **e_off** = ``off / rater_scale / off_top_``
    * **e_on**  = ``on  / rater_scale / on_curve_[bin]``
    * **k**     = ``e_off / max(e_on, 1)`` (only on visits with both)
    * **k_med** = per-patient median of ``k``; ``k_cnt`` = count
    * **e_on2** = ``e_on * k_med``  (``k_med = 1.0`` when no k available)
    * **e_best** = ``e_off`` when present, else ``e_on2``
    * **e_max**, **e_mean** = max / mean of ``(e_off, e_on2)``
    * **Projection features for e_off and e_on** — always computed at slope
      2.5 only: ``{src}_proj_max``, ``_proj_2nd``, ``_proj_p75``,
      ``_proj_med``, ``_proj_mean``, ``_proj_min``, ``_proj_std``.
    * **Projection features for e_best** — computed once per slope in
      ``slopes`` (default ``(2.5,)``); column names carry the slope value,
      e.g. ``e_best_s2.5_proj_max``, so multiple slopes produce separate
      feature sets without colliding.

    ``patient_id`` and ``Index`` are dropped at the end of ``transform``
    (handled by the internal :class:`PatientFeatures` call).

    Parameters
    ----------
    slopes : tuple of float, default (2.5,)
        Projection slopes at which e_best statistics are computed.
        e_off and e_on projections always use slope 2.5 only.
    patient_features_kwargs :
        Keyword arguments forwarded to :class:`PatientFeatures`.
    """

    def __init__(self, slopes: tuple = (2.5,), **patient_features_kwargs) -> None:
        self.slopes = slopes
        self.patient_features_kwargs = patient_features_kwargs

    def fit(self, X: pd.DataFrame, y) -> "EnvelopeFeatures":
        """Learn rater scale, on-curve, and off-top from training data.

        Only rows where ``y > 10`` are used for the envelope statistics,
        following the clinical rationale that very low true scores are
        dominated by floor effects and measurement artefacts.

        Parameters
        ----------
        X :
            Training feature matrix; must contain ``rater_id``, ``off``,
            ``on``, ``time_since_intake_on``, ``patient_id``, ``age``.
        y :
            Target vector (true OFF MDS-UPDRS motor score).

        Returns
        -------
        self
        """
        y = np.asarray(y)
        mask = y > 10

        df = X.copy()
        df["_y"] = y

        dfm = df[mask].copy()

        # --- rater_scale_: 75th pct of off/y per rater, normalised by cross-rater median ---
        if "off" in dfm.columns and "rater_id" in dfm.columns:
            valid_off = dfm[["rater_id", "off", "_y"]].dropna(subset=["off"]).copy()
            valid_off["rater_id"] = valid_off["rater_id"].astype(object)
            ratio_off = valid_off["off"].astype(float) / valid_off["_y"]
            p75 = ratio_off.groupby(valid_off["rater_id"]).quantile(0.75)
            self.rater_scale_: pd.Series = p75 / p75.median()
        else:
            self.rater_scale_ = pd.Series(dtype=float)

        # --- on_curve_: median of on/y/rater_scale per tsi bin ---
        if "on" in dfm.columns:
            rs = dfm["rater_id"].astype(object).map(self.rater_scale_).fillna(1.0).astype(float)
            on_ratio = dfm["on"].astype(float) / dfm["_y"] / rs.values
            tsi_col = dfm.get("time_since_intake_on")
            bins = _bin_tsi(tsi_col if tsi_col is not None else pd.Series(np.nan, index=dfm.index))
            tmp = pd.DataFrame({"ratio": on_ratio, "bin": bins})
            self.on_curve_: dict[int, float] = tmp.groupby("bin")["ratio"].median().to_dict()
        else:
            self.on_curve_ = {}

        # --- off_top_: 75th pct of off/y/rater_scale ---
        if "off" in dfm.columns:
            rs = dfm["rater_id"].astype(object).map(self.rater_scale_).fillna(1.0).astype(float)
            off_norm = dfm["off"].astype(float) / dfm["_y"] / rs.values
            self.off_top_: float = float(off_norm.dropna().quantile(0.75))
        else:
            self.off_top_ = 1.0

        # --- internal PatientFeatures ---
        self._pf = PatientFeatures(**self.patient_features_kwargs)
        self._pf.fit(X, y)
        self.pf_fitted_: bool = True  # trailing-underscore sentinel

        return self

    # ------------------------------------------------------------------
    # Projection helpers (vectorised self-merge, no patient loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _proj_stats(
        df: pd.DataFrame,
        src_col: str,
        slope: float = _PROJ_SLOPE,
        age_col: str = "age",
        pid_col: str = "patient_id",
        col_prefix: str | None = None,
    ) -> pd.DataFrame:
        """Compute projection statistics for ``src_col`` across patient visits.

        For each pair of visits ``(i, j)`` of the same patient where visit *j*
        has a non-NaN value of ``src_col``, the projection of visit *j* onto
        visit *i*'s age is:

            proj_{i,j} = src_j + slope * (age_i − age_j)

        This method returns, for each row *i*, the max, second largest,
        75th percentile, median, mean, min, and std of the projections over all
        valid *j* (NaN when none exist).

        Parameters
        ----------
        df :
            DataFrame with columns ``patient_id``, ``age``, and ``src_col``.
        src_col :
            Column whose values are projected.
        slope :
            Projection slope (score units per year of age difference).
        age_col :
            Age column name.
        pid_col :
            Patient-id column name.
        col_prefix :
            Prefix for output column names. Defaults to ``{src_col}_proj``.

        Returns
        -------
        pd.DataFrame
            Columns ``{col_prefix}_max``, ``_2nd``, ``_p75``,
            ``_med``, ``_mean``, ``_min``, ``_std``, indexed as ``df``.
        """
        prefix = col_prefix if col_prefix is not None else f"{src_col}_proj"
        sub = df[[pid_col, age_col, src_col]].copy()
        sub = sub.rename(columns={age_col: "_age_i", src_col: "_val_i"})
        sub["_row_i"] = np.arange(len(sub))

        # Keep only visits that have the source value (these are the "j" visits).
        donors = sub.dropna(subset=["_val_i"]).copy()
        donors = donors.rename(columns={"_age_i": "_age_j", "_val_i": "_val_j", "_row_i": "_row_j"})

        # Self-merge on patient_id to pair every (i, j) within a patient.
        merged = sub.merge(donors[[pid_col, "_age_j", "_val_j"]], on=pid_col, how="left")

        # proj_{i,j} = val_j + slope * (age_i - age_j)
        merged["_proj"] = merged["_val_j"] + slope * (merged["_age_i"] - merged["_age_j"])

        # Aggregate per row_i.
        grp = merged.groupby("_row_i", sort=False)["_proj"]

        out = pd.DataFrame(index=np.arange(len(sub)))
        out[f"{prefix}_max"] = grp.max()
        out[f"{prefix}_2nd"] = grp.apply(
            lambda s: float(np.partition(s.dropna(), -2)[-2]) if len(s.dropna()) >= 2 else np.nan
        )
        out[f"{prefix}_p75"] = grp.quantile(0.75)
        out[f"{prefix}_med"] = grp.median()
        out[f"{prefix}_mean"] = grp.mean()
        out[f"{prefix}_min"] = grp.min()
        out[f"{prefix}_std"] = grp.std()

        out.index = df.index
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive envelope features and return the enriched table.

        Parameters
        ----------
        X :
            Feature matrix with ``patient_id``, ``age``, ``rater_id``,
            ``off``, ``on``, ``time_since_intake_on`` present.

        Returns
        -------
        pd.DataFrame
            One row per input row, same order, with ``Index`` and
            ``patient_id`` dropped (via the internal PatientFeatures call).
        """
        # 1. PatientFeatures (drops Index and patient_id at the end).
        #    We need patient_id for projection steps, so work on a copy
        #    that keeps patient_id and age alongside.
        pf_out = self._pf.transform(X)  # patient_id already dropped here

        # Keep a working copy with patient_id and age for the join steps.
        work = X[["patient_id", "age"]].copy()
        work.index = X.index

        # 2. rater_scale per row (cast to object first so map works on Categorical columns).
        rs = (
            X["rater_id"].astype(object).map(self.rater_scale_).fillna(1.0).astype(float).values
            if "rater_id" in X.columns
            else np.ones(len(X))
        )

        # 3. e_off
        if "off" in X.columns:
            off_top = self.off_top_ if self.off_top_ != 0 else 1.0
            e_off = X["off"].astype(float) / rs / off_top
        else:
            e_off = pd.Series(np.nan, index=X.index)

        # 4. e_on
        if "on" in X.columns:
            tsi = X.get("time_since_intake_on", pd.Series(np.nan, index=X.index))
            bins = _bin_tsi(tsi)
            curve_vals = bins.map(self.on_curve_).astype(float)
            curve_vals = curve_vals.where(curve_vals > 0, other=np.nan)
            e_on = X["on"].astype(float) / rs / curve_vals.values
        else:
            e_on = pd.Series(np.nan, index=X.index)

        # 5. k = e_off / max(e_on, 1), only on visits having both.
        both_mask = e_off.notna() & e_on.notna()
        k = pd.Series(np.nan, index=X.index)
        k[both_mask] = e_off[both_mask] / np.maximum(e_on[both_mask], 1.0)

        # 6. k_med and k_cnt per patient.
        tmp_k = pd.DataFrame({"patient_id": X["patient_id"].values, "k": k.values}, index=X.index)
        k_med = tmp_k.groupby("patient_id")["k"].transform("median").fillna(1.0)
        k_cnt = tmp_k.groupby("patient_id")["k"].transform("count")

        # 7. e_on2 = e_on * k_med.
        e_on2 = e_on * k_med.values

        # 8. Composite envelope features.
        e_best = e_off.where(e_off.notna(), other=e_on2)
        e_max = pd.concat([e_off.rename("a"), e_on2.rename("b")], axis=1).max(axis=1)
        e_mean = pd.concat([e_off.rename("a"), e_on2.rename("b")], axis=1).mean(axis=1)

        # Assemble the envelope columns into work alongside patient_id / age.
        work["e_off"] = e_off.values
        work["e_on"] = e_on.values
        work["e_best"] = e_best.values

        work["k"] = k.values
        work["k_med"] = k_med.values
        work["k_cnt"] = k_cnt.values
        work["e_on2"] = e_on2.values
        work["e_max"] = e_max.values
        work["e_mean"] = e_mean.values

        # 9. Projection features.
        #    e_off and e_on: always at slope 2.5 only.
        #    e_best: once per slope in self.slopes (column names carry the slope).
        proj_frames = []
        for src in ("e_off", "e_on"):
            proj_frames.append(self._proj_stats(work, src_col=src, slope=_PROJ_SLOPE))
        for s in self.slopes:
            # Format slope as a short string: drop trailing zero after decimal.
            s_str = f"{s:g}"
            proj_frames.append(
                self._proj_stats(
                    work,
                    src_col="e_best",
                    slope=s,
                    col_prefix=f"e_best_s{s_str}_proj",
                )
            )

        # 10. Assemble final output: PatientFeatures output + envelope scalars + projections.
        envelope_cols = work.drop(columns=["patient_id", "age"])
        result = pd.concat([pf_out, envelope_cols] + proj_frames, axis=1)
        return result


# ---------------------------------------------------------------------------
# StackedEnvelopeRegressor
# ---------------------------------------------------------------------------


class StackedEnvelopeRegressor(BaseEstimator, RegressorMixin):
    """Two-stage stacked regressor built on top of EnvelopeFeatures.

    Stage 1 gives a first score trajectory per patient; stage 2 compares each
    corrected envelope score to that trajectory to detect visits that were
    under-scored relative to the patient's own history.

    Parameters
    ----------
    n_inner : int, default 5
        Number of folds in the inner GroupKFold (on patient_id) used to
        produce out-of-fold stage-1 predictions.
    """

    def __init__(self, n_inner: int = 5) -> None:
        self.n_inner = n_inner

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tuned_hgb() -> HistGradientBoostingRegressor:
        """Return HGB with the hyperparameters tuned in 22_hgb_rater_smooth."""
        return HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=0.03,
            max_iter=2000,
            early_stopping=True,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
        )

    @staticmethod
    def _smooth_patient_poly(
        pid: np.ndarray,
        age: np.ndarray,
        values: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """Fit a polynomial of ``degree`` in centred age per patient and return fitted values.

        Mirrors the smoothing logic of :class:`PatientSmoothedRegressor`.
        """
        out = np.empty(len(values), dtype=float)
        tmp = pd.DataFrame(
            {"patient_id": pid, "age": age, "val": values},
            index=np.arange(len(values)),
        )
        for _pid, grp in tmp.groupby("patient_id", sort=False):
            ages = grp["age"].values
            vals = grp["val"].values
            n_distinct = len(np.unique(ages))
            deg = min(degree, n_distinct - 1)
            age_centre = ages.mean()
            ages_c = ages - age_centre
            if deg < 1:
                out[grp.index] = vals
            else:
                coeffs = np.polyfit(ages_c, vals, deg)
                out[grp.index] = np.polyval(coeffs, ages_c)
        return out

    @staticmethod
    def _stage2_features(
        Z: pd.DataFrame,
        patient_id: pd.Series,
        age: pd.Series,
        p1: np.ndarray,
    ) -> pd.DataFrame:
        """Build the stage-2 feature matrix from Z, patient grouping, and stage-1 predictions.

        Parameters
        ----------
        Z :
            EnvelopeFeatures output (index aligned with patient_id / age / p1).
        patient_id :
            Patient identifier series (same index as Z).
        age :
            Age series (same index as Z).
        p1 :
            Stage-1 raw predictions array (same length as Z).

        Returns
        -------
        pd.DataFrame
            Z plus p1, patient p1 stats, polynomial smoothed p1, and
            ratio-residual (rr) features from e_off and e_on2.
        """
        pid_arr = patient_id.values
        age_arr = age.values

        # --- p1 and its per-patient mean/std ---
        tmp_p1 = pd.DataFrame(
            {"patient_id": pid_arr, "p1": p1},
            index=Z.index,
        )
        p1_mean = tmp_p1.groupby("patient_id")["p1"].transform("mean").values
        p1_std = tmp_p1.groupby("patient_id")["p1"].transform("std").fillna(0.0).values

        # --- p1_s1: p1 smoothed per patient by a degree-1 polynomial in age ---
        p1_s1 = StackedEnvelopeRegressor._smooth_patient_poly(pid_arr, age_arr, p1, degree=1)

        # --- p1_s2: p1 smoothed per patient by a degree-2 polynomial in age ---
        p1_s2 = StackedEnvelopeRegressor._smooth_patient_poly(pid_arr, age_arr, p1, degree=2)

        # --- ratio-residual features from e_off and e_on2 ---
        # rr = c / max(p1_s2, 3) for each envelope column c in (e_off, e_on2).
        rr_frames = {}
        for c in ("e_off", "e_on2"):
            if c not in Z.columns:
                continue
            c_vals = Z[c].values.astype(float)
            rr = c_vals / np.maximum(p1_s2, 3.0)
            rr_series = pd.Series(rr, index=Z.index, name=f"rr_{c}")
            tmp_rr = pd.DataFrame(
                {"patient_id": pid_arr, "rr": rr},
                index=Z.index,
            )
            rr_max = tmp_rr.groupby("patient_id")["rr"].transform("max").values
            rr_p75 = tmp_rr.groupby("patient_id")["rr"].transform(lambda s: s.quantile(0.75)).values
            rr_med = tmp_rr.groupby("patient_id")["rr"].transform("median").values
            rr_frames[f"rr_{c}"] = rr
            rr_frames[f"rr_{c}_pat_max"] = rr_max
            rr_frames[f"rr_{c}_pat_p75"] = rr_p75
            rr_frames[f"rr_{c}_pat_med"] = rr_med

        feat = Z.copy()
        feat["p1"] = p1
        feat["p1_pat_mean"] = p1_mean
        feat["p1_pat_std"] = p1_std
        feat["p1_s1"] = p1_s1
        feat["p1_s2"] = p1_s2
        for col_name, arr in rr_frames.items():
            feat[col_name] = arr

        return feat

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y) -> "StackedEnvelopeRegressor":
        """Fit the two-stage stacked model.

        Parameters
        ----------
        X :
            Training feature matrix; must contain ``patient_id`` and ``age``.
        y :
            Target vector.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)

        # --- EnvelopeFeatures with three slopes for e_best projections ---
        self.env_: EnvelopeFeatures = EnvelopeFeatures(slopes=(1.5, 2.5, 3.5))
        self.env_.fit(X, y)
        Z = self.env_.transform(X)

        # --- Stage 1: out-of-fold predictions on Z ---
        patient_id = X["patient_id"]
        age = X["age"]

        inner_cv = GroupKFold(n_splits=self.n_inner)
        inner_groups = patient_id.values
        p1_oof = np.zeros(len(y), dtype=float)

        stage1_base = HistGradientBoostingRegressor(random_state=0)
        for train_idx, val_idx in inner_cv.split(Z, y, groups=inner_groups):
            fold_model = clone(stage1_base)
            fold_model.fit(Z.iloc[train_idx], y[train_idx])
            p1_oof[val_idx] = fold_model.predict(Z.iloc[val_idx])

        # Refit stage-1 on all of Z.
        self.stage1_: HistGradientBoostingRegressor = clone(stage1_base)
        self.stage1_.fit(Z, y)

        # --- Stage 2: build features and fit ---
        feat2 = self._stage2_features(Z, patient_id, age, p1_oof)
        self.stage2_: HistGradientBoostingRegressor = self._tuned_hgb()
        self.stage2_.fit(feat2, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate stacked predictions.

        Parameters
        ----------
        X :
            Feature matrix; must contain ``patient_id`` and ``age``.

        Returns
        -------
        np.ndarray
            Stage-2 predictions in the same row order as ``X``.
        """
        Z = self.env_.transform(X)
        p1 = self.stage1_.predict(Z)
        feat2 = self._stage2_features(Z, X["patient_id"], X["age"], p1)
        return self.stage2_.predict(feat2)


def build_envelope_stacked_learner(**kwargs) -> PatientSmoothedRegressor:
    """Return the unfit ``24_hgb_envelope_stacked`` learner.

    Architecture: ``PatientSmoothedRegressor(degree=2)`` wrapping
    ``StackedEnvelopeRegressor(n_inner=5)``.

    Stage 1 fits EnvelopeFeatures with three projection slopes (1.5, 2.5, 3.5)
    on the training data, produces out-of-fold trajectory predictions via an
    inner GroupKFold on patient_id, then refits on all data. Stage 2 augments
    the envelope features with trajectory-comparison signals (smoothed p1,
    per-patient p1 stats, ratio residuals) and fits a tuned HGB. The outer
    degree-2 polynomial smoother removes high-frequency noise from the final
    predictions.

    Parameters
    ----------
    **kwargs :
        Ignored; accepted for API compatibility.

    Returns
    -------
    PatientSmoothedRegressor
    """
    return PatientSmoothedRegressor(StackedEnvelopeRegressor(), degree=2)


def build_envelope_learner(tuned: bool = False) -> PatientSmoothedRegressor:
    """Return the unfit ``23_hgb_envelope`` learner.

    Architecture: ``PatientSmoothedRegressor`` wrapping
    ``EnvelopeFeatures() → HGB``, with a degree-2 polynomial smoother.

    Clinical rationale
    ------------------
    Each rater scores on a personal multiplicative scale; ``on`` is the true
    OFF score reduced by a factor that depends on the delay since drug intake;
    visit noise is one-sided (a corrected score is rarely above the true score,
    often well below), so the true score follows the upper envelope of a
    patient's corrected scores, not their mean; ``on`` and ``off`` of the same
    visit share the same noise, so their ratio is stable within a patient.

    Parameters
    ----------
    tuned : bool, default False
        When ``False``, uses ``HistGradientBoostingRegressor(random_state=0)``
        with default hyperparameters.  When ``True``, uses the tuned
        hyperparameters from experiment ``22_hgb_rater_smooth``
        (learning_rate=0.03, max_iter=2000, early_stopping=True,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0).

    Returns
    -------
    PatientSmoothedRegressor
    """
    if tuned:
        hgb = HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=0.03,
            max_iter=2000,
            early_stopping=True,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
        )
    else:
        hgb = HistGradientBoostingRegressor(random_state=0)

    return PatientSmoothedRegressor(
        make_pipeline(EnvelopeFeatures(), hgb),
        degree=2,
    )
