"""Regression tests for the RTide 1.0.1 bug fixes (B1-B13)."""
import numpy as np
import pandas as pd
import pytest

from rtide import RTide

LAT, LON = 44.9062, -66.996201


def _index(periods, start="2024-01-01"):
    # Explicit ns resolution: pandas 3 defaults to us, for which RTide infers sample_rate=1000.
    return pd.date_range(start=start, periods=periods, freq="1h", tz="UTC").as_unit("ns")


def _tide(t_hours, phase=0.0):
    return (1.2 * np.cos(2 * np.pi * t_hours / 12.4206012 + phase)
            + 0.3 * np.cos(2 * np.pi * t_hours / 23.93447213 + 2 * phase))


def elevation_df(periods=168, n_exog=0, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    index = _index(periods, start)
    t = np.arange(periods, dtype=np.float64)
    data = {"observations": _tide(t, phase=seed) + 0.02 * rng.standard_normal(periods)}
    for i in range(n_exog):
        data[f"exog{i}"] = np.sin(2 * np.pi * t / (50.0 + 20 * i)) + 0.05 * rng.standard_normal(periods)
        data["observations"] = data["observations"] + 0.1 * data[f"exog{i}"]
    return pd.DataFrame(data, index=index)


def currents_df(periods=168, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    t = np.arange(periods, dtype=np.float64)
    return pd.DataFrame(
        {"u": _tide(t, 0.3) + 0.02 * rng.standard_normal(periods),
         "v": 0.5 * _tide(t, 1.1) + 0.02 * rng.standard_normal(periods)},
        index=_index(periods, start),
    )


# ---------------------------------------------------------------------------
# B1: feature cache must not be reused across different datasets / stations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("difference", ["data", "latlon"])
def test_b1_cache_not_reused_for_different_station(difference):
    first = elevation_df(seed=1)
    if difference == "data":
        second, lat2, lon2 = elevation_df(seed=2), LAT, LON
    else:
        second, lat2, lon2 = first.copy(), LAT + 5.0, LON + 5.0

    RTide(first, LAT, LON).Prepare_Inputs(verbose=False)
    model = RTide(second, lat2, lon2)
    model.Prepare_Inputs(verbose=False)

    np.testing.assert_array_equal(model.prepped_dfs["observations"].to_numpy(), second["observations"].to_numpy())
    if difference == "latlon":
        fresh = RTide(second, lat2, lon2)
        fresh.Prepare_Inputs(save=False)
        np.testing.assert_allclose(model.prepped_dfs.to_numpy(float), fresh.prepped_dfs.to_numpy(float), rtol=1e-6)


def test_b1_cache_reused_for_same_data(monkeypatch):
    data = elevation_df(seed=3)
    first = RTide(data, LAT, LON)
    first.Prepare_Inputs(verbose=False)

    def fail(*args, **kwargs):
        raise AssertionError("features recomputed instead of loaded from cache")

    monkeypatch.setattr(RTide, "_compute_global_tide_base", fail)
    second = RTide(data, LAT, LON)
    second.Prepare_Inputs(verbose=False)
    # The CSV round trip of the legacy cache is not bit-exact in the last digit.
    np.testing.assert_allclose(second.prepped_dfs["observations"].to_numpy(), data["observations"].to_numpy(),
                               rtol=1e-12)
    assert list(second.prepped_dfs.columns) == list(first.prepped_dfs.columns)


def test_b1_cache_without_fingerprint_is_recomputed(tmp_path, monkeypatch):
    data = elevation_df(seed=4)
    RTide(data, LAT, LON).Prepare_Inputs(verbose=False)
    sidecar = tmp_path / "rtide_saves" / "RTide_fingerprint.json"
    assert sidecar.exists()
    sidecar.unlink()  # simulate a cache written by 1.0.0

    calls = []
    original = RTide._compute_global_tide_base

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RTide, "_compute_global_tide_base", counting)
    RTide(data, LAT, LON).Prepare_Inputs(verbose=False)
    assert calls, "a cache without a fingerprint sidecar must be recomputed"
    assert sidecar.exists()


# ---------------------------------------------------------------------------
# B2: forecasting with all-NaN observations
# ---------------------------------------------------------------------------
def _train_default(df, **train_kwargs):
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(verbose=False)
    model.Train(**{"standard_epochs": 2, "verbose": False, **train_kwargs})
    return model


def test_b2_pure_forecast_with_nan_observations():
    full = elevation_df(periods=168 + 48, seed=5)
    future = full.iloc[168:].copy()
    future["observations"] = np.nan

    model = _train_default(full.iloc[:168])
    model.Predict(future)
    assert len(model.test_prediction_df) == len(future.index)
    assert np.isfinite(model.test_prediction_df["rtide"].to_numpy()).all()
    assert np.size(model.test_predictions["test_observations"]) == 0

    fresh = RTide(future, LAT, LON)
    fresh.path = "./rtide_saves/RTide"
    fresh.Load_Model()
    fresh.Predict(future)
    assert len(fresh.test_prediction_df) == len(future.index)
    assert np.isfinite(fresh.test_prediction_df["rtide"].to_numpy()).all()
    assert np.size(fresh.test_predictions["test_observations"]) == 0
    np.testing.assert_allclose(fresh.test_predictions["rtide_test"], model.test_predictions["rtide_test"], rtol=1e-5)


def test_b2_partial_nan_observations_unchanged():
    full = elevation_df(periods=168 + 48, seed=6)
    future = full.iloc[168:].copy()
    nan_rows = [0, 5, 17]
    future.iloc[nan_rows, 0] = np.nan

    model = _train_default(full.iloc[:168])
    model.Predict(future)

    legacy = model.prediction_dfs.dropna()  # legacy row selection
    assert len(legacy) == len(future) - len(nan_rows)
    assert model.test_prediction_df.index.equals(legacy.index)
    np.testing.assert_array_equal(model.test_predictions["test_observations"], legacy["observations"].to_numpy())


# ---------------------------------------------------------------------------
# B3: repeat Prepare_Inputs / Predict on the same object uses the raw settings
# ---------------------------------------------------------------------------
MULTIVARIATE_LAG_MODES = [False, "standard", "negative", "all", [-3, -2, -1]]


@pytest.mark.parametrize("repeat_kwargs", [True, False], ids=["kwargs-repeated", "kwargs-omitted"])
@pytest.mark.parametrize("mvl", MULTIVARIATE_LAG_MODES, ids=str)
def test_b3_prediction_columns_match_training(mvl, repeat_kwargs):
    df = elevation_df(periods=240, n_exog=1, seed=7)
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(multivariate_lags=mvl, save=False)
    train_columns = list(model.prepped_dfs.columns)
    processed_lags = list(model.multivariate_lags)

    second_kwargs = {"multivariate_lags": mvl} if repeat_kwargs else {}
    model.Prepare_Inputs(prediction=True, save=False, **second_kwargs)
    assert list(model.prediction_dfs.columns) == train_columns
    assert list(model.multivariate_lags) == processed_lags  # public attribute still holds the processed list


def test_b3_train_and_predict_same_object_standard_lags():
    df = elevation_df(periods=240, n_exog=1, seed=8)
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(multivariate_lags="standard", verbose=False)
    model.Train(standard_epochs=2, verbose=False)
    model.Predict(df)
    assert list(model.prediction_dfs.columns) == list(model.prepped_dfs.columns)
    assert np.isfinite(model.test_predictions["rtide_test"]).all()


def test_b3_explicit_kwargs_override_stored_settings():
    model = RTide(elevation_df(seed=9), LAT, LON)
    model.Prepare_Inputs(symmetrical=False, save=False)
    n_asymmetric = model.prepped_dfs.shape[1]
    model.Prepare_Inputs(symmetrical=True, save=False)
    assert model.prepped_dfs.shape[1] != n_asymmetric


def test_b3_saved_inputs_describe_effective_settings():
    from rtide.utils import load_inputs_from_pickle

    df = elevation_df(seed=10)
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(symmetrical=True, verbose=False)
    model.Prepare_Inputs(uniform_lags=[3, 1], verbose=False)  # symmetrical not passed: stays True

    saved = load_inputs_from_pickle(model.path)
    assert saved["symmetrical"] is True
    assert saved["uniform_lags"] == [3, 1]

    fresh = RTide(df, LAT, LON)
    fresh.Prepare_Inputs(**dict(saved, save=False, prediction=True))
    assert list(fresh.prediction_dfs.columns) == list(model.prepped_dfs.columns)


# ---------------------------------------------------------------------------
# B5: loss='SSP'
# ---------------------------------------------------------------------------
def test_b5_train_with_ssp_loss_and_reload():
    df = elevation_df(seed=11)
    model = _train_default(df, standard_epochs=1, loss="SSP")
    assert model.model is not None

    fresh = RTide(df, LAT, LON)
    fresh.path = "./rtide_saves/RTide"
    fresh.Load_Model()
    fresh.Predict(df)
    model.Predict(df)
    np.testing.assert_allclose(fresh.test_predictions["rtide_test"], model.test_predictions["rtide_test"], rtol=1e-5)


# ---------------------------------------------------------------------------
# B6: auto-load of a saved model in Predict / Shap_Analysis
# ---------------------------------------------------------------------------
def test_b6_predict_autoloads_saved_model():
    df = elevation_df(seed=12)
    trained = _train_default(df)
    trained.Predict(df)

    fresh = RTide(df, LAT, LON)
    fresh.path = "./rtide_saves/RTide"
    fresh.Predict(df)  # no Load_Model()
    assert fresh.model is not None
    np.testing.assert_allclose(fresh.test_predictions["rtide_test"], trained.test_predictions["rtide_test"], rtol=1e-5)


def test_b6_predict_without_any_model_raises_runtime_error():
    df = elevation_df(seed=13)
    model = RTide(df, LAT, LON)
    model.path = "./rtide_saves/does_not_exist"
    with pytest.raises(RuntimeError, match="No model has been trained or saved at ./rtide_saves/does_not_exist"):
        model.Predict(df)


def test_b6_shap_analysis_autoloads_and_uses_prepped_dfs():
    df = elevation_df(periods=40, seed=14)
    _train_default(df)

    fresh = RTide(df, LAT, LON)
    fresh.Prepare_Inputs(save=False)  # prediction_dfs stays None
    fresh.path = "./rtide_saves/RTide"
    fresh.Shap_Analysis(plot=False)
    assert fresh.model is not None
    assert np.shape(fresh.shap_values)[0] == len(fresh.prepped_dfs.dropna())


# ---------------------------------------------------------------------------
# B7: training frame with zero usable rows
# ---------------------------------------------------------------------------
def test_b7_no_usable_training_rows_raises(tmp_path):
    model = RTide(elevation_df(n_exog=1, seed=15), LAT, LON)
    with pytest.raises(ValueError, match="No training rows have complete observations and features"):
        model.Prepare_Inputs(multivariate_lags=[-1000], verbose=False)
    assert not (tmp_path / "rtide_saves" / "RTide_global_tide.csv").exists()


def test_b7_never_raises_in_prediction_mode():
    df = elevation_df(n_exog=1, seed=16)
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(multivariate_lags=[-1000], prediction=True, save=False)
    assert model.prediction_dfs.dropna().empty


# ---------------------------------------------------------------------------
# B8: trend models can be saved and reloaded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["elevation", "currents"])
@pytest.mark.parametrize("trend", ["linear", "quadratic"])
def test_b8_trend_model_reload_matches_in_session(trend, kind):
    df = elevation_df(seed=17) if kind == "elevation" else currents_df(seed=17)
    model = _train_default(df, standard_epochs=1, trend=trend)
    model.Predict(df)

    fresh = RTide(df, LAT, LON)
    fresh.path = "./rtide_saves/RTide"
    fresh.Load_Model()
    fresh.Predict(df)
    assert fresh.trend == trend
    np.testing.assert_allclose(fresh.test_predictions["rtide_test"], model.test_predictions["rtide_test"], rtol=1e-5)


# ---------------------------------------------------------------------------
# B9: Shap_Analysis uses the scaling mode the model was trained with
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("featurewise", [False, True])
def test_b9_shap_analysis_uses_trained_scaling(featurewise):
    df = elevation_df(periods=40, seed=18)
    model = _train_default(df, featurewise_scaling=featurewise)
    assert model.featurewise_X_scaling is featurewise

    captured = {}
    original = model._transform_X

    def spy(X, featurewise):
        captured["featurewise"] = featurewise
        return original(X, featurewise=featurewise)

    model._transform_X = spy
    model.Predict(df)
    model.Shap_Analysis(plot=False)
    assert captured["featurewise"] is featurewise
    assert np.shape(model.shap_values)[0] == len(model.prediction_dfs.dropna())


# ---------------------------------------------------------------------------
# B11: tide-only prediction zeroes exactly the exogenous columns (behaviour change)
# ---------------------------------------------------------------------------
def _features(model):
    df = model.prepped_dfs.dropna()
    return df.to_numpy()[:, model.n_outputs:], [str(c) for c in df.columns[model.n_outputs:]]


def _legacy_tide_only_X(model, X):
    n_exog = len(model.exog_columns)
    n_lagged_exog = 0 if model.multivariate_lags == [0] else n_exog * len(model.multivariate_lags)
    X_tide = X.copy()
    if n_exog > 0:
        X_tide[:, 0:n_exog] = 0
    if n_lagged_exog > 0:
        X_tide[:, -n_lagged_exog:] = 0
    return X_tide


def _zeroed_columns(X_before, X_after):
    changed = np.any(X_before != X_after, axis=0)
    assert np.all(X_after[:, changed] == 0)
    return [int(i) for i in np.flatnonzero(changed)]


@pytest.mark.parametrize("n_exog", [1, 2])
@pytest.mark.parametrize("self_prediction", [False, [-1, -2]], ids=["no-selfpred", "selfpred"])
def test_b11_realtime_exog_zeroing_identical_to_legacy(n_exog, self_prediction):
    model = RTide(elevation_df(n_exog=n_exog, seed=19), LAT, LON)
    model.Prepare_Inputs(self_prediction=self_prediction, save=False)
    X, columns = _features(model)

    X_tide = model._tide_only_X(X, columns)
    assert _zeroed_columns(X, X_tide) == list(range(n_exog))
    np.testing.assert_array_equal(X_tide, _legacy_tide_only_X(model, X))


@pytest.mark.parametrize(
    "n_exog, mvl",
    [(1, "standard"), (1, [-3, -2, -1]), (2, [-3, -2, -1]), (2, "standard")],
    ids=["1exog-standard", "1exog-list", "2exog-list", "2exog-standard"],
)
@pytest.mark.parametrize("self_prediction", [False, [-1, -2]], ids=["no-selfpred", "selfpred"])
def test_b11_lagged_exog_zeroes_only_exog_columns(n_exog, mvl, self_prediction):
    model = RTide(elevation_df(periods=240, n_exog=n_exog, seed=20), LAT, LON)
    model.Prepare_Inputs(multivariate_lags=mvl, self_prediction=self_prediction, save=False)
    X, columns = _features(model)

    exog_idx = [i for i, c in enumerate(columns)
                if any(c == e or c.startswith(f"{e}_") for e in model.exog_columns)]
    assert len(exog_idx) == n_exog * len(model.multivariate_lags)

    X_tide = model._tide_only_X(X, columns)
    assert _zeroed_columns(X, X_tide) == exog_idx
    untouched = [i for i, c in enumerate(columns)
                 if c.startswith(("Gravitational_", "Radiational_", "observations_"))]
    assert len(untouched) == len(columns) - len(exog_idx)
    np.testing.assert_array_equal(X_tide[:, untouched], X[:, untouched])
    # The legacy slicing zeroed tidal forcing columns instead.
    assert not np.array_equal(X_tide, _legacy_tide_only_X(model, X))


def test_b11_train_uses_exact_exog_zeroing():
    df = elevation_df(periods=240, n_exog=2, seed=21)
    model = RTide(df, LAT, LON)
    model.Prepare_Inputs(multivariate_lags=[-3, -2, -1], save=False)
    model.Train(standard_epochs=2, verbose=False, save_weights=False)

    X, columns = _features(model)
    scaled = model._transform_X(model._tide_only_X(X, columns), featurewise=False)
    expected = model.scaler_Y.inverse_transform(model.model.predict(scaled, verbose=0)).reshape(-1)
    np.testing.assert_allclose(model.train_predictions["RTide_nomulti"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# B12: trend warm-start coefficients are used (behaviour change)
# ---------------------------------------------------------------------------
def _initial_trend_weights(trend, n_outputs, coeffs, architecture="response"):
    from rtide import models

    model = models.build_model(
        architecture=architecture, input_dims=6, n_outputs=n_outputs, hidden_nodes=6, depth=2,
        trend=trend, trend_initial_coeffs=coeffs,
    )
    layer = model.get_layer("trend_layer")
    if trend == "linear":
        return {"c1": layer.trend_weights.numpy(), "c0": layer.trend_bias.numpy()}
    return {"c2": layer.trend_weights_quad.numpy(), "c1": layer.trend_weights_lin.numpy(),
            "c0": layer.trend_bias.numpy()}


@pytest.mark.parametrize("architecture", ["response", "siren"])
@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize("trend", ["linear", "quadratic"])
def test_b12_fitted_coefficients_initialise_trend_weights(trend, n_outputs, architecture):
    rng = np.random.default_rng(22)
    keys = ["c0", "c1"] + (["c2"] if trend == "quadratic" else [])
    coeffs = {k: rng.normal(size=n_outputs).astype(np.float32) for k in keys}

    weights = _initial_trend_weights(trend, n_outputs, coeffs, architecture)
    assert set(weights) == set(keys)
    for key in keys:
        np.testing.assert_allclose(weights[key], coeffs[key], rtol=1e-6)


@pytest.mark.parametrize("trend", ["linear", "quadratic"])
def test_b12_legacy_coefficient_names_still_accepted(trend):
    if trend == "linear":
        legacy, expected = {"slope": [0.5, -0.25], "intercept": [1.5, 2.0]}, {"c1": [0.5, -0.25], "c0": [1.5, 2.0]}
    else:
        legacy = {"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]}
        expected = {"c2": [0.1, 0.2], "c1": [0.3, 0.4], "c0": [0.5, 0.6]}
    weights = _initial_trend_weights(trend, 2, {k: np.asarray(v, np.float32) for k, v in legacy.items()})
    for key, value in expected.items():
        np.testing.assert_allclose(weights[key], value, rtol=1e-6)


def test_b12_fit_trend_initial_coeffs_round_trip():
    from rtide.utils import fit_trend_initial_coeffs

    t = np.linspace(0, 1, 50)
    y = np.column_stack([0.3 + 1.2 * t - 0.7 * t**2, -0.1 + 0.4 * t + 0.2 * t**2])
    coeffs = fit_trend_initial_coeffs(t_norm=t, y_scaled=y, trend="quadratic")
    weights = _initial_trend_weights("quadratic", 2, coeffs)
    np.testing.assert_allclose(weights["c0"], [0.3, -0.1], atol=1e-5)
    np.testing.assert_allclose(weights["c1"], [1.2, 0.4], atol=1e-5)
    np.testing.assert_allclose(weights["c2"], [-0.7, 0.2], atol=1e-5)


# ---------------------------------------------------------------------------
# B13: Train(featurewise_X_scaling=...) warns but is still ignored
# ---------------------------------------------------------------------------
def _featurewise_x_warnings(record):
    return [w for w in record if issubclass(w.category, FutureWarning) and "featurewise_X_scaling" in str(w.message)]


def test_b13_featurewise_X_scaling_warns_and_is_ignored():
    df = elevation_df(periods=40, seed=23)
    with pytest.warns(FutureWarning, match=r"Train\(featurewise_X_scaling=\.\.\.\) is currently ignored; "
                                           r"use featurewise_scaling=\.\.\. \. A future release will honour"):
        model = _train_default(df, featurewise_X_scaling=True)
    assert model.featurewise_X_scaling is False


@pytest.mark.parametrize("x_value, value, expect_warning", [(True, True, False), (False, True, True), (True, False, True)])
def test_b13_featurewise_scaling_wins_when_both_given(x_value, value, expect_warning):
    import warnings

    df = elevation_df(periods=40, seed=24)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = _train_default(df, featurewise_X_scaling=x_value, featurewise_scaling=value)
    assert model.featurewise_X_scaling is value
    assert bool(_featurewise_x_warnings(record)) is expect_warning


def test_b13_no_warning_without_featurewise_X_scaling():
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _train_default(elevation_df(periods=40, seed=25), featurewise_scaling=True)
    assert not _featurewise_x_warnings(record)
