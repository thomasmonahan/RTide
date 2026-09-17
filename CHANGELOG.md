# Changelog

## 1.0.1

Patch release. Apart from the two behaviour changes listed below, feature matrices and predictions
from saved models are numerically identical to 1.0.0 (enforced by golden-master tests in `test/golden/`).

### Fixed

- **B1** – The station-mode feature cache (`{path}_global_tide.csv`) is no longer reused for a different dataset or station prepared with identical kwargs. A data fingerprint is stored in a new `{path}_fingerprint.json`; caches written by 1.0.0 are recomputed once. The cache log message now names `_inputs.pickle`.
- **B2** – `Predict` works in pure-forecast mode (all observations NaN): rows are selected using the feature columns only. Partial-NaN behaviour is unchanged.
- **B3** – Calling `Prepare_Inputs` again on the same object (including via `Predict`) now builds the same feature columns as training, which fixes the shape mismatch with `multivariate_lags='standard'`/`'negative'`. Kwargs passed explicitly on a repeat call now override the earlier settings instead of being silently ignored, and `_inputs.pickle` records the effective settings.
- **B4** – `Prep()` can be called directly with a list of `multivariate_lags` without raising `AttributeError`.
- **B5** – `Train(loss='SSP')` no longer raises `NameError`, and models trained with it can be reloaded.
- **B6** – `Predict` and `Shap_Analysis` load the saved model automatically when none is in memory, and raise `RuntimeError` if none exists. `Shap_Analysis` falls back to the training features when `Predict` has not been called.
- **B7** – `Prepare_Inputs` raises a clear `ValueError` when no training rows have complete observations and features (e.g. lags longer than the record), instead of an opaque scikit-learn error in `Train`.
- **B8** – Models trained with `trend='linear'` or `'quadratic'` can be reloaded with `Load_Model()`.
- **B9** – `Shap_Analysis` uses feature-wise X scaling for models trained with `featurewise_scaling=True` (previously it crashed for them).
- **B10** – `scikit-learn` and `joblib` added to `install_requires`; `python_requires='>=3.9'`.

### Behaviour changes

- **B11 – Tide-only (exogenous inputs removed) predictions with lagged exogenous inputs.**
  Affects models with exogenous columns whose `multivariate_lags` produce lagged inputs
  (`'standard'`, `'negative'`, `'all'`, or a list containing any lag other than 0).
  1.0.0 zeroed one exogenous lag plus the last tidal forcing columns; 1.0.1 zeroes exactly the exogenous columns.
  Changed outputs: `train_predictions['RTide_nomulti']`, `train_predictions['rtide_tide_train']`,
  `rtide_ha`, and therefore `test_predictions['rtide_tide_test']`.
  Unchanged: `rtide_train`, `rtide_test`, the UTide baselines (`utide_tide_*`, `irls_utide_*`, `utide_ha`, `utide_irls`)
  and saved model files. Models without exogenous inputs, or with realtime-only exogenous inputs (the default), are unaffected.
- **B12 – Trend warm start.** For models trained with `trend='linear'` or `'quadratic'`, the trend weights now start
  from the least-squares fit computed before training (previously they always started at zero). All trained outputs of
  trend models may therefore differ from 1.0.0. Models without a trend are unaffected.

### Deprecations / warnings

- **B13** – `Train(featurewise_X_scaling=...)` (as used in the example notebooks) is still ignored, but now emits a
  `FutureWarning`; use `featurewise_scaling=...`. If both are passed, `featurewise_scaling` wins as before, with a
  warning when they differ. A future release will honour `featurewise_X_scaling`.
