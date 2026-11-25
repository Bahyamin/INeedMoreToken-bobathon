# %% [markdown]
# # Experiment: 22_hgb_rater_smooth
#
# **Hypothesis:** Adding per-rater bias correction (PatientFeatures rater_adjust=True)
#   and a per-patient polynomial post-smoother (PatientSmoothedRegressor degree=2)
#   on top of a well-tuned HGB will reduce RMSE below the 4.45 of 21_hgb_patient.
#
# **Method:** PatientSmoothedRegressor(PatientFeatures(rater_adjust=True) → HGB(tuned));
#   GroupKFold(n_splits=5) with groups=patient_id; X passed with patient_id present.
#
# **Result:** Grouped-CV RMSE 3.99 ± 0.07 (5-fold GroupKFold on patient_id).

# %%
import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.tracking import get_project
from parkinson.pipeline import build_rater_smooth_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Experiment tracking (optional skore Hub)

# %%
project = get_project()  # None when no skore Hub credentials are available

# %% [markdown]
# ## Data and learner
#
# X is passed with patient_id present; PatientFeatures uses it for per-patient
# aggregates and rater-bias correction, then drops Index and patient_id before HGB.

# %%
X, y = load_dataset(DATA_DIR)
learner = build_rater_smooth_learner()

# %% [markdown]
# ## Evaluate
#
# Pre-generate GroupKFold index pairs (passed as a list so skore can reuse them).
# X is passed with patient_id so PatientFeatures can groupby on each fold.

# %%
groups = X["patient_id"]
cv_splits = list(splitter.split(X, y, groups=groups))
report = skore.evaluate(learner, X=X, y=y, splitter=cv_splits)
report

# %% [markdown]
# ## Check RMSE threshold

# %%
rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_rmse = float(rmse_df.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

if mean_rmse > 4.4:
    raise RuntimeError(
        f"RMSE {mean_rmse:.4f} exceeds threshold of 4.4 — stopping. "
        "Check rater-bias correction or polynomial smoother for issues."
    )

# %% [markdown]
# ## Persist

# %%
if project is not None:
    project.put("22_hgb_rater_smooth", report)

# %% [markdown]
# ## Submit

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final = build_rater_smooth_learner()
final.fit(X, y)
test_predictions = final.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "22_hgb_rater_smooth.csv", index=False)
print(f"Submission written: {len(submission)} rows")
