"""Golden-master tests: current code must reproduce the 1.0.0 baseline fixtures exactly.

Fixtures are generated ONLY from the untouched baseline (see generate_baseline.py).
If one of these tests fails after a change, the change is wrong: never regenerate.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rtide import RTide

from ._golden_io import frame_from_arrays, index_from_arrays

DATA = Path(__file__).resolve().parent / "data"
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
CACHE_PLACEHOLDER = "<CACHE_DIR>"


def _assert_same_index(actual, z, prefix):
    expected = index_from_arrays(z, prefix)
    assert actual.unit == expected.unit
    assert str(actual.tz) == str(expected.tz)
    np.testing.assert_array_equal(actual.as_unit("ns").asi8, expected.as_unit("ns").asi8)


@pytest.mark.parametrize("config_id", MANIFEST["feature_configs"])
def test_golden_features(config_id, tmp_path, monkeypatch):
    with np.load(DATA / "features" / f"{config_id}.npz", allow_pickle=False) as z:
        kwargs = {
            k: (str(tmp_path / "precomputed_cache") if v == CACHE_PLACEHOLDER else v)
            for k, v in json.loads(str(z["kwargs"])).items()
        }
        if bool(z["isolated_home"]):
            monkeypatch.setenv("HOME", str(tmp_path / "home"))

        model = RTide(frame_from_arrays(z, "in"), float(z["lat"]), float(z["lon"]))
        model.Prepare_Inputs(**kwargs)
        prepped = model.prepped_dfs

        assert [str(c) for c in prepped.columns] == [str(c) for c in z["out_columns"]]
        assert [str(t) for t in prepped.dtypes] == [str(t) for t in z["out_dtypes"]]
        _assert_same_index(prepped.index, z, "out")
        np.testing.assert_allclose(
            prepped.to_numpy(dtype=np.float64), z["out_values"], rtol=1e-6, atol=1e-9, equal_nan=True
        )


@pytest.mark.parametrize("config_id", MANIFEST["prediction_configs"])
def test_golden_predictions(config_id, tmp_path):
    source = DATA / "predictions" / config_id
    saves = tmp_path / "rtide_saves"
    saves.mkdir(exist_ok=True)
    for name in MANIFEST["prediction_artifacts"]:
        shutil.copy2(source / name, saves / name)

    with np.load(source / "expected.npz", allow_pickle=False) as z:
        pred_df = frame_from_arrays(z, "in")
        model = RTide(pred_df, float(z["lat"]), float(z["lon"]))
        model.path = "./rtide_saves/RTide"
        model.Load_Model()
        model.Predict(pred_df)

        np.testing.assert_allclose(
            np.asarray(model.test_predictions["rtide_test"], dtype=np.float64), z["rtide_test"], rtol=1e-5
        )
        df = model.test_prediction_df
        assert [str(c) for c in df.columns] == [str(c) for c in z["df_columns"]]
        _assert_same_index(df.index, z, "df")
        np.testing.assert_allclose(df.to_numpy(dtype=np.float64), z["df_values"], rtol=1e-5, equal_nan=True)
