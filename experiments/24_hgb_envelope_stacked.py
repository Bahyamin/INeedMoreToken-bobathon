# %% [markdown]
# # Experiment: 24_hgb_envelope_stacked
#
# **Hypothesis:** A two-stage stacked model gives the HGB a richer signal by
#   first building a per-patient trajectory (stage 1, EnvelopeFeatures with
#   three projection slopes) and then letting stage 2 compare each corrected
#   visit score to that trajectory to detect under-scored visits.
#
# **Method:** PatientSmoothedRegressor(degree=2) wrapping
#   StackedEnvelopeRegressor(n_inner=5): EnvelopeFeatures(slopes=(1.5,2.5,3.5))
#   → stage-1 HGB (out-of-fold, then refit) → stage-2 tuned HGB on Z + trajectory
#   features; GroupKFold(n_splits=5) on patient_id.
#
# **Result:** Grouped-CV RMSE 2.88 ± 0.04 (5-fold GroupKFold on patient_id);
#   Kaggle public leaderboard RMSE 2.918; Kaggle private leaderboard RMSE 2.695.

# %%
import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.tracking import get_project
from parkinson.pipeline import build_envelope_stacked_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Experiment tracking (optional skore Hub)

# %%
project = get_project()  # None when no skore Hub credentials are available

# %% [markdown]
# ## Data

# %%
X, y = load_dataset(DATA_DIR)
groups = X["patient_id"]
cv_splits = list(splitter.split(X, y, groups=groups))

# %% [markdown]
# ## Evaluate

# %%
learner = build_envelope_stacked_learner()
report = skore.evaluate(learner, X=X, y=y, splitter=cv_splits)
report

# %%
rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_rmse = float(rmse_df.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

RMSE_23 = 3.11
print(f"Delta vs 23_hgb_envelope: {mean_rmse - RMSE_23:+.4f}")

if mean_rmse > 3.2:
    raise RuntimeError(
        f"RMSE {mean_rmse:.4f} exceeds threshold 3.2 — stopping. "
        "Review StackedEnvelopeRegressor implementation before proceeding."
    )

# %% [markdown]
# ## Hub put

# %%
if project is not None:
    project.put("24_hgb_envelope_stacked", report)

# %% [markdown]
# ## Submit

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final_learner = build_envelope_stacked_learner()
final_learner.fit(X, y)
test_predictions = final_learner.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "24_hgb_envelope_stacked.csv", index=False)
print(f"Submission written: {len(submission)} rows")
