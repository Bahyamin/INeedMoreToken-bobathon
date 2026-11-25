# %% [markdown]
# # Experiment: 21_hgb_patient
#
# **Hypothesis:** Adding per-patient longitudinal features (visit rank,
#   age since first visit, per-patient aggregates and OLS trend of off/on
#   vs age, previous/next off/on, missing flags) will help HGB capture the
#   smooth within-patient disease-progression curve and reduce RMSE below
#   the 7.57 baseline.
#
# **Method:** PatientFeatures → HGB(random_state=0); GroupKFold(n_splits=5)
#   with groups=patient_id; X passed with patient_id present so the
#   transformer can compute per-patient aggregates on each fold.
#
# **Result:** Grouped-CV RMSE 4.45 ± 0.06 (5-fold GroupKFold on patient_id).

# %%
import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.tracking import get_project
from parkinson.pipeline import build_patient_learner

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
# X is passed with patient_id present; PatientFeatures uses it to compute
# per-patient aggregates, then drops Index and patient_id before HGB.

# %%
X, y = load_dataset(DATA_DIR)
learner = build_patient_learner()

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
# rmse_df has MultiIndex columns; extract mean and std from the Aggregate level.
mean_rmse = float(rmse_df.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_rmse = float(rmse_df.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

if mean_rmse > 5.0:
    raise RuntimeError(
        f"RMSE {mean_rmse:.4f} exceeds threshold of 5.0 — stopping. "
        "Check PatientFeatures for data leakage or transformation errors."
    )

# %% [markdown]
# ## Persist

# %%
if project is not None:
    project.put("21_hgb_patient", report)

# %% [markdown]
# ## Submit

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final = build_patient_learner()
final.fit(X, y)
test_predictions = final.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "21_hgb_patient.csv", index=False)
print(f"Submission written: {len(submission)} rows")
