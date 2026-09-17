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
