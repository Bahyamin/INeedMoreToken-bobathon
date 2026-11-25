# %% [markdown]
# # Experiment: 20_hgb_baseline
#
# **Hypothesis:** HistGradientBoostingRegressor on all X columns (native
#   categoricals + NaN handling) will substantially beat the dummy
#   baseline (RMSE 16.50) by learning rater-bias and age-progression
#   signals from ``off``, ``rater_id``, ``age``, and medication columns.
#
# **Method:** sklearn Pipeline with DropColumns → HGB(random_state=0);
#   GroupKFold(n_splits=5) with groups=patient_id; fit on all 44,590
#   training visits for submission.
#
# **Result:** Grouped-CV RMSE 7.57 ± 0.14 (5-fold GroupKFold on patient_id).

# %%
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset, split_kwargs
from parkinson.evaluate import splitter
from parkinson.tracking import get_project
from parkinson.pipeline import build_learner

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

# %%
X, y = load_dataset(DATA_DIR)
learner = build_learner()

# Build the feature matrix seen by the model (Index and patient_id dropped).
X_model = X.drop(columns=["Index", "patient_id"])

# %% [markdown]
# ## Evaluate
#
# ``GroupKFold(n_splits=5)`` requires the ``groups`` vector at split time.
# skore's ``CrossValidationReport`` calls ``splitter.split(X, y)`` without
# ``groups`` for plain sklearn estimators, so we pre-generate the indices
# as a generator of ``(train_idx, test_idx)`` tuples — a form that
# skore's ``splitter`` parameter accepts directly.

# %%
groups = split_kwargs(X)["groups"]
cv_splits = list(splitter.split(X_model, y, groups=groups))
report = skore.evaluate(learner, X=X_model, y=y, splitter=cv_splits)
report

# %%
rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.xs("mean", axis=1, level="Aggregate").iloc[0, 0])
std_rmse = float(rmse_df.xs("std", axis=1, level="Aggregate").iloc[0, 0])
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

# %% [markdown]
# ## Persist

# %%
if project is not None:
    project.put("20_hgb_baseline", report)

# %% [markdown]
# ## Submit
#
# Refit on all training rows (no CV split); predict test visits;
# write ``submissions/20_hgb_baseline.csv`` with header ``Index,target``.

# %%
import pandas as pd

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)
X_test_model = X_test.drop(columns=["Index", "patient_id"])

final = build_learner()
final.fit(X_model, y)
test_predictions = final.predict(X_test_model)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "20_hgb_baseline.csv", index=False)
print(f"Submission written: {len(submission)} rows")
