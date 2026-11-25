"""Data loading for the Parkinson's MDS-UPDRS motor score prediction task.

Owns how raw CSV data is materialized into ``(X, y)`` for training and
``X_test`` for submission. Categorical columns (``cohort``, ``gene``,
``rater_id``) that carry clinical grouping information are cast to
``pandas.Categorical`` so that downstream estimators (e.g.
``HistGradientBoostingRegressor``) can handle them natively without
one-hot expansion and without imputing their missing values.

``split_kwargs`` exposes ``patient_id`` as the ``groups`` vector for
``GroupKFold``, ensuring that no patient's visits appear in both the
training folds and the validation fold. This is the clinically correct
evaluation strategy because the competition scores on unseen patients.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

# Columns encoded as pandas Categorical so HGB can exploit native
# categorical support (no NaN imputation, no one-hot explosion).
_CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]

# Columns excluded from model features: row identifier and patient
# grouping key (patient_id leaks patient identity to the model).
_DROP_COLS = ["Index", "patient_id"]


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast clinical grouping columns to pandas Categorical dtype.

    HistGradientBoostingRegressor uses the category dtype to treat
    these columns natively — preserving their NaN values and avoiding
    the dimensionality explosion of one-hot encoding. This is
    important for ``gene`` (32 % missing) and ``rater_id`` (whose
    mean bias ranges −10.5 to +2.2, making it a strong confounder).

    Parameters
    ----------
    df :
        Raw feature DataFrame from CSV.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``cohort``, ``gene``, and ``rater_id``
        converted to ``category`` dtype.
    """
    df = df.copy()
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_dataset(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and target, returning ``(X, y)``.

    Reads ``X_train.csv`` and ``y_train.csv`` from *data_dir*, joins
    them on ``Index``, and returns the feature matrix and target vector.
    The ``Index`` and ``patient_id`` columns are kept in ``X`` so that
    ``split_kwargs`` can extract ``patient_id`` for group-aware CV.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``PROJECT_ROOT / "data"``.

    Returns
    -------
    X : pd.DataFrame
        Training feature matrix (all columns from ``X_train.csv``),
        with categorical columns cast to ``category`` dtype.
    y : pd.Series
        Training target: the true OFF MDS-UPDRS motor score.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    X = pd.read_csv(data_dir / "X_train.csv")
    y_df = pd.read_csv(data_dir / "y_train.csv")
    y = y_df.set_index("Index")["target"].reindex(X["Index"]).reset_index(drop=True)
    X = _cast_categoricals(X)
    return X, y


def load_test_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Load test features for the Kaggle submission CSV.

    Reads ``X_test.csv`` from *data_dir* and applies the same
    categorical dtype casting as training so the fitted pipeline sees
    an aligned table.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_test.csv``.
        Defaults to ``PROJECT_ROOT / "data"``.

    Returns
    -------
    pd.DataFrame
        Test feature matrix with ``Index`` preserved (needed for the
        submission header) and categorical columns cast to ``category``.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    X_test = pd.read_csv(data_dir / "X_test.csv")
    X_test = _cast_categoricals(X_test)
    return X_test


def split_kwargs(X: pd.DataFrame) -> dict:
    """Return the ``groups`` vector for ``GroupKFold``.

    Extracts ``patient_id`` from the feature matrix so
    ``skore.evaluate`` / ``GroupKFold`` can ensure that each
    patient's visits are held out together. Clinically, this
    reflects the competition's evaluation on entirely unseen patients.

    Parameters
    ----------
    X :
        Feature matrix returned by ``load_dataset``.

    Returns
    -------
    dict
        ``{"groups": pd.Series}`` keyed for ``GroupKFold``'s
        ``groups`` parameter.
    """
    return {"groups": X["patient_id"]}
