# %% [markdown]
# # Experiment: 25_estimator_reports
#
# **Purpose:** Produce skore EstimatorReports (not CrossValidationReports) for
#   the two envelope learners: the competition required each Kaggle submission
#   to link a skore Hub report.
#   Uses a patient-grouped holdout: train on the train indices, test on the test
#   indices of the first GroupKFold(5) pair.
#
# **Result:** Holdout RMSE 3.14 (23_hgb_envelope_est), 2.92 (24_hgb_envelope_stacked_est);
#   Kaggle public leaderboard RMSE 2.918 for 24_hgb_envelope_stacked_est;
#   Kaggle private leaderboard RMSE 2.695 for 24_hgb_envelope_stacked_est.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skore import EstimatorReport

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.tracking import get_project
from parkinson.pipeline import build_envelope_learner, build_envelope_stacked_learner

# %%
DATA_DIR = PROJECT_ROOT / "data"

project = get_project()  # None when no skore Hub credentials are available

# %%
X, y = load_dataset(DATA_DIR)
groups = X["patient_id"]

cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
train_idx, test_idx = cv_splits[0]

X_train_h, X_test_h = X.iloc[train_idx], X.iloc[test_idx]
y_train_h, y_test_h = y.iloc[train_idx], y.iloc[test_idx]

# %% [markdown]
# ## 23 — tuned envelope learner (EstimatorReport)

# %%
learner_23 = build_envelope_learner(tuned=True)
report_23 = EstimatorReport(
    learner_23,
    X_train=X_train_h,
    y_train=y_train_h,
    X_test=X_test_h,
    y_test=y_test_h,
)
rmse_23 = float(report_23.metrics.rmse())
print(f"23_hgb_envelope_est  holdout RMSE: {rmse_23:.4f}")

if project is not None:
    project.put("23_hgb_envelope_est", report_23)

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_kaggle = load_test_dataset(DATA_DIR)

fresh_23 = clone(learner_23)
fresh_23.fit(X, y)
preds_23 = fresh_23.predict(X_test_kaggle)

sub_23 = X_test_kaggle[["Index"]].copy()
sub_23["target"] = preds_23
sub_23.to_csv(submissions_dir / "23_hgb_envelope_est.csv", index=False)

ref_path_23 = submissions_dir / "23_hgb_envelope.csv"
if ref_path_23.exists():  # written by experiments/23_hgb_envelope.py
    ref_23 = pd.read_csv(ref_path_23)
    diff_23 = np.abs(sub_23["target"].values - ref_23["target"].values).max()
    print(f"23 max abs diff vs 23_hgb_envelope.csv: {diff_23:.6f}")

# %% [markdown]
# ## 24 — stacked envelope learner (EstimatorReport)

# %%
learner_24 = build_envelope_stacked_learner()
report_24 = EstimatorReport(
    learner_24,
    X_train=X_train_h,
    y_train=y_train_h,
    X_test=X_test_h,
    y_test=y_test_h,
)
rmse_24 = float(report_24.metrics.rmse())
print(f"24_hgb_envelope_stacked_est  holdout RMSE: {rmse_24:.4f}")

if project is not None:
    project.put("24_hgb_envelope_stacked_est", report_24)

# %%
fresh_24 = build_envelope_stacked_learner()
fresh_24.fit(X, y)
preds_24 = fresh_24.predict(X_test_kaggle)

sub_24 = X_test_kaggle[["Index"]].copy()
sub_24["target"] = preds_24
sub_24.to_csv(submissions_dir / "24_hgb_envelope_stacked_est.csv", index=False)

ref_path_24 = submissions_dir / "24_hgb_envelope_stacked.csv"
if ref_path_24.exists():  # written by experiments/24_hgb_envelope_stacked.py
    ref_24 = pd.read_csv(ref_path_24)
    diff_24 = np.abs(sub_24["target"].values - ref_24["target"].values).max()
    print(f"24 max abs diff vs 24_hgb_envelope_stacked.csv: {diff_24:.6f}")
