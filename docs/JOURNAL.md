# Experiment journal

Chronological log of the modelling experiments. Every score is a **5-fold
`GroupKFold` on `patient_id`** RMSE (mean ± std) unless stated otherwise: all
visits of a patient stay in the same fold, matching the competition, which
scores entirely unseen patients.

## Data facts that drove the design

- 44,590 training visits from 5,576 patients (4–12 visits each); 11,013 test
  visits from 1,395 other patients. The same 50 `rater_id` appear in train and test.
- The target (true OFF motor score) is a smooth curve of age within a patient
  (residual RMSE ≈ 1.6 around a per-patient line).
- Measured `off` = target + bias (mean −5.4, std 8.7); the bias depends on the
  rater (per-rater mean from −10.5 to +2.2) and on the time since drug intake.
- Measured `on` is ≈ 22 points below the target and depends on `time_since_intake_on`.
- Missing values: `off` 42 %, `on` 30 %, `ledd` 37 %, `gene` 32 %,
  `time_since_intake_off` 79 %.

## History

| Script | Idea | Grouped-CV RMSE |
|---|---|---|
| `10_baselines` | Mean predictor; median imputation + Ridge (alpha 0.1 → 100) on numeric columns | 16.50 ± 0.31 (mean) · 10.60 ± 0.24 (Ridge, every alpha) |
| `20_hgb_baseline` | `HistGradientBoostingRegressor` on all columns, native categoricals and NaN handling | 7.57 ± 0.14 |
| `21_hgb_patient` | `PatientFeatures`: visit rank, per-patient aggregates, OLS trend of `off`/`on` vs age, previous/next measurements, missing flags | 4.45 ± 0.06 |
| `22_hgb_rater_smooth` | Per-rater bias correction + tuned HGB + per-patient degree-2 polynomial smoother | 3.99 ± 0.07 |
| `23_hgb_envelope` | `EnvelopeFeatures`: multiplicative rater scale and drug-timing curve learnt on the training fold, upper envelope of corrected scores | 3.11 ± 0.06 (tuned; untuned 3.20 ± 0.06) |
| `24_hgb_envelope_stacked` | Two-stage stacking: stage-1 HGB out-of-fold trajectory → stage-2 HGB comparing each visit to it; projection slopes 1.5 / 2.5 / 3.5 | **2.88 ± 0.04** |
| `25_estimator_reports` | Single patient-grouped holdout (fold 0) reports for the submitted models; submissions identical to 23/24 | holdout 3.14 (23) · 2.92 (24) |

Final submission (`24`): **Kaggle public RMSE 2.918, private RMSE 2.695, rank 28 / 294.**

## Notes per step

**10 → 20.** The linear baseline plateaus at 10.60 whatever the regularisation:
the limit is the model family. HGB handles the 30–42 % missingness natively
(NaN is a split direction, so missingness itself is a signal) and uses
`cohort`, `gene`, `rater_id` as categoricals without one-hot expansion.

**20 → 21.** The target is smooth within a patient, so each visit is enriched
with information from the patient's other visits. All features come from X
only, so the transformer behaves identically on the test set.

**21 → 22.** Raters score on personal scales; correcting their bias and
smoothing each patient's predicted trajectory with a degree-2 polynomial in
age enforces the expected smooth progression.

**22 → 23.** The rater effect is multiplicative (`off ≈ target × rater_scale`)
and noise is one-sided (a visit is rarely scored above the true state). The
rater scale (75th percentile of `off / y` per rater) and the drug-response
curve (median `on / y` per bin of time since intake) are learnt on the
training fold only (rows with `y > 10`, avoiding floor effects), then the
upper envelope of each patient's corrected scores is used as features.

**23 → 24.** Stage 1 produces out-of-fold predictions (inner `GroupKFold`)
that describe each patient's trajectory; stage 2 compares every visit to that
trajectory and can detect under-scored visits.

**Validation check.** The grouped-CV estimate of the final model (2.88) lies
between the public (2.918) and private (2.695) leaderboard scores: the
patient-grouped CV was a reliable guide for model selection.

**Technical note.** CV splits are pre-computed and passed to `skore.evaluate`
as a list: a generator (`iter()`) leaves the report's estimator unfitted in
skore 0.26.
