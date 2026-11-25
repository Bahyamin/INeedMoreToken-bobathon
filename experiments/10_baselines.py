# %% [markdown]
# # Experiment: 10_baselines
#
# **Purpose:** Reference scores that every later model must beat, evaluated
#   with the same patient-grouped cross-validation as experiments 20–25.
#
# **Method:** numeric clinical columns only (categoricals and patient_id
#   excluded); GroupKFold(n_splits=5) on patient_id.
#   - DummyRegressor(strategy="mean")
#   - SimpleImputer(median) + Ridge, alpha in {0.1, 1, 10, 100}
#
# **Result:** Dummy 16.50 ± 0.31; Ridge 10.60 ± 0.24 for every alpha tested
#   (regularisation is not the bottleneck: the linear model family is).

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from parkinson.data import load_dataset
from parkinson.evaluate import splitter

# %%
X, y = load_dataset()
groups = X["patient_id"]
numeric_cols = [
    c
    for c in X.columns
    if c not in {"Index", "patient_id"} and pd.api.types.is_numeric_dtype(X[c])
]
X_num = X[numeric_cols]
cv_splits = list(splitter.split(X_num, y, groups=groups))

models = {"dummy (mean)": DummyRegressor(strategy="mean")}
for alpha in (0.1, 1.0, 10.0, 100.0):
    models[f"ridge (alpha={alpha:g})"] = make_pipeline(
        SimpleImputer(strategy="median"), Ridge(alpha=alpha)
    )

# %%
for name, model in models.items():
    scores = -cross_val_score(
        model, X_num, y, cv=cv_splits, scoring="neg_root_mean_squared_error"
    )
    print(f"{name:<20} grouped-CV RMSE {scores.mean():.2f} ± {scores.std():.2f}")
