"""Cross-validation strategy for the Parkinson's motor score experiments.

Holds only the objects passed to ``skore.evaluate``:

- ``splitter`` — the cross-validator.

``GroupKFold(n_splits=5)`` is the correct splitter for this task
because the competition evaluates on entirely unseen patients.
Splitting at the patient level (via ``groups=patient_id``) ensures
that all visits from the same patient appear in exactly one fold,
giving an honest estimate of out-of-patient generalisation. Using a
visit-level splitter (e.g. ``KFold``) would leak intra-patient
temporal information and inflate the CV score.
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# GroupKFold with 5 folds ensures each held-out fold contains ~20%
# of the 5,576 patients. The ``groups`` vector (patient_id) is
# passed via ``split_kwargs`` in data.py and forwarded to
# skore.evaluate through the experiment script.
splitter = GroupKFold(n_splits=5)
