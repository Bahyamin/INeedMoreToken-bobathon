# %% [markdown]
# # Experiment: 23_hgb_envelope
#
# **Hypothesis:** Upper-envelope normalisation of off/on scores
#   (per-rater multiplicative scale + pharmacodynamic on-curve calibration)
#   will improve RMSE over 22_hgb_rater_smooth by giving the model a target
#   that is already in a common scale across raters and intake-delay conditions.
#
# **Method:** EnvelopeFeatures → HGB wrapped in PatientSmoothedRegressor(degree=2);
#   both untuned (default HGB) and tuned (22_hgb_rater_smooth params) variants;
#   GroupKFold(n_splits=5) with groups=patient_id.
#
# **Result:** Grouped-CV RMSE 3.11 ± 0.06 (tuned variant, 5-fold GroupKFold on patient_id).

# %%
import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.tracking import get_project
from parkinson.pipeline import build_envelope_learner

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
# ## Evaluate untuned variant

# %%
learner_untuned = build_envelope_learner(tuned=False)
report_untuned = skore.evaluate(learner_untuned, X=X, y=y, splitter=cv_splits)
report_untuned

# %%
rmse_df_u = report_untuned.metrics.rmse()
mean_u = float(rmse_df_u.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_u = float(rmse_df_u.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"Untuned  RMSE: {mean_u:.4f} ± {std_u:.4f}")

if mean_u > 3.6:
    raise RuntimeError(
        f"Untuned RMSE {mean_u:.4f} exceeds threshold 3.6 — stopping. "
        "Review EnvelopeFeatures implementation before proceeding."
    )

# %% [markdown]
# ## Evaluate tuned variant

# %%
learner_tuned = build_envelope_learner(tuned=True)
report_tuned = skore.evaluate(learner_tuned, X=X, y=y, splitter=cv_splits)
report_tuned

# %%
rmse_df_t = report_tuned.metrics.rmse()
mean_t = float(rmse_df_t.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_t = float(rmse_df_t.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"Tuned    RMSE: {mean_t:.4f} ± {std_t:.4f}")

# %% [markdown]
# ## Choose the better variant and persist

# %%
RMSE_22 = 3.99  # 22_hgb_rater_smooth baseline

if mean_u <= mean_t:
    best_report = report_untuned
    best_mean, best_std = mean_u, std_u
    best_label = "untuned"
    best_learner = build_envelope_learner(tuned=False)
else:
    best_report = report_tuned
    best_mean, best_std = mean_t, std_t
    best_label = "tuned"
    best_learner = build_envelope_learner(tuned=True)

print(f"Best variant : {best_label}  RMSE {best_mean:.4f} ± {best_std:.4f}")
print(f"Delta vs 22  : {best_mean - RMSE_22:+.4f}")

# %%
if project is not None:
    project.put("23_hgb_envelope", best_report)

# %% [markdown]
# ## Submit

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

best_learner.fit(X, y)
test_predictions = best_learner.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "23_hgb_envelope.csv", index=False)
print(f"Submission written: {len(submission)} rows")
