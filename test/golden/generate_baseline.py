"""Generate golden-master fixtures from the ORIGINAL (1.0.0) RTide code.

Run ONLY with the frozen baseline worktree on the path, from a scratch directory
containing de421.bsp:

    cd /tmp && mkdir -p golden_gen && cd golden_gen && cp <path-to>/de421.bsp .
    PYTHONPATH=<repo>/../rtide-baseline python <repo>/test/golden/generate_baseline.py --out <repo>/test/golden/data

Never regenerate fixtures from modified code: if a golden test fails after a change,
the change is wrong. The script refuses to run if `rtide` is imported from this
repository, from a checkout with modified tracked files, or if fixtures already exist.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _golden_io import frame_from_arrays, frame_to_arrays  # noqa: E402

SCRIPT_REPO = Path(__file__).resolve().parents[2]
CACHE_PLACEHOLDER = "<CACHE_DIR>"
ARTIFACT_SUFFIXES = [
    "_model_weights.keras",
    "_scaler_X.save",
    "_scaler_Y.save",
    "_meta.json",
    "_inputs.pickle",
]
SITES = {
    "elevation": (44.9062, -66.996201),
    "currents": (58.6557103353, 3.0),
}
HOURLY_TRAIN_ROWS = 240       # ~10 days hourly
HOURLY_TOTAL_ROWS = 336       # 14 days: predictions use the last 240 rows
QUARTER_HOURLY_ROWS = 288     # ~3 days at 15-min

# (kind, n_exog, cadence) and Prepare_Inputs kwargs. All feature builds use save=False.
_BASE_FEATURE_CONFIGS = {
    "F1": (("elevation", 0, "1h"), {}),
    "F2": (("elevation", 0, "1h"), {"symmetrical": True}),
    "F3": (("elevation", 0, "1h"), {"uniform_lags": [3, 1], "symmetrical": True}),
    "F4": (("elevation", 0, "1h"), {"radiational": False}),
    "F5": (("elevation", 0, "15min"), {"sample_rate": 4, "symmetrical": True}),
    "F6": (("elevation", 0, "1h"), {"location_mode": "fixed", "use_precomputed_inputs": True,
                                     "precomputed_cache_dir": CACHE_PLACEHOLDER}),
    "F6b": (("elevation", 0, "15min"), {"fixed_location": True, "sample_rate": 4, "symmetrical": True}),
    "F7": (("elevation", 1, "1h"), {}),
    "F8": (("elevation", 1, "1h"), {"multivariate_lags": "standard"}),
    "F9": (("elevation", 1, "1h"), {"multivariate_lags": "negative"}),
    "F10": (("elevation", 2, "1h"), {"multivariate_lags": [-3, -2, -1]}),
    "F11": (("elevation", 1, "1h"), {"multivariate_lags": [-24, -12], "multivariate_realtime": False}),
    "F12": (("elevation", 0, "1h"), {"self_prediction": [-1, -2]}),
    "F13": (("currents", 0, "1h"), {}),
    "F14": (("currents", 1, "1h"), {"symmetrical": True, "multivariate_lags": [-3, -2, -1]}),
}
# Configs whose forcing cache defaults to ~/.cache/rtide: HOME is pointed at a temp dir.
_ISOLATED_HOME = {"F6b"}
# Microsecond-index copies (pandas 3 date_range default) of a representative subset.
# F8/F9 are deliberately absent: with a us index 1.0.0 infers sample_rate=1000, so
# 'standard'/'negative' exog lags (e.g. -150.385h) never align with hourly data and
# every row is NaN - a config that cannot be trained, not one to freeze.
_US_VARIANTS = ["F1", "F5", "F6", "F7", "F10", "F13"]


def _feature_configs():
    configs = {}
    for cid, (data, kwargs) in _BASE_FEATURE_CONFIGS.items():
        configs[cid] = {"data": data, "unit": "ns", "kwargs": kwargs, "isolated_home": cid in _ISOLATED_HOME}
    for cid in _US_VARIANTS:
        data, kwargs = _BASE_FEATURE_CONFIGS[cid]
        configs[f"{cid}_us"] = {"data": data, "unit": "us", "kwargs": kwargs,
                                "isolated_home": cid in _ISOLATED_HOME}
    return configs


FEATURE_CONFIGS = _feature_configs()

# Trained with standard_epochs=3, verbose=False (+ extra Train kwargs); reloaded in a fresh object.
PREDICTION_CONFIGS = {
    "P_F1": ("F1", {}),
    "P_F1_featurewise": ("F1", {"featurewise_scaling": True}),
    "P_F7": ("F7", {}),
    "P_F8": ("F8", {}),
    "P_F10": ("F10", {}),
    "P_F13": ("F13", {}),
    "P_F1_us": ("F1_us", {}),
}


def _check_baseline():
    import rtide

    root = Path(rtide.__file__).resolve().parents[1]
    if root == SCRIPT_REPO:
        sys.exit(
            f"Refusing to run: rtide is imported from the working repository ({root}).\n"
            "Put the frozen baseline worktree first on PYTHONPATH."
        )
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"Refusing to run: baseline checkout {root} has modified tracked files:\n{dirty}")
    return root, sha


def _library_versions():
    versions = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "tensorflow", "keras", "sklearn", "joblib", "scipy",
                 "skyfield", "utide", "shap", "matplotlib"]:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception as exc:  # pragma: no cover
            versions[name] = f"unavailable ({type(exc).__name__})"
    return versions


def _seed_everything(seed=0):
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover
        tf.keras.utils.set_random_seed(seed)


def _tide(t_hours, components):
    return sum(a * np.cos(2 * np.pi * t_hours / period + np.deg2rad(phase)) for a, period, phase in components)


def make_series(kind, n_exog, cadence, unit):
    """Deterministic synthetic series. The generated arrays are stored in the fixtures."""
    if cadence == "1h":
        periods, start, seed = HOURLY_TOTAL_ROWS, "2024-03-01 00:00", 11
    else:
        periods, start, seed = QUARTER_HOURLY_ROWS, "2024-06-01 00:00", 22
    seed += 100 * n_exog + (0 if kind == "elevation" else 1000)
    rng = np.random.default_rng(seed)

    index = pd.date_range(start=start, periods=periods, freq=cadence, tz="UTC").as_unit(unit)
    ns = index.as_unit("ns").asi8
    t_hours = (ns - ns[0]) / 3.6e12

    exog = {}
    if n_exog >= 1:
        exog["river"] = 0.5 + 0.3 * np.sin(2 * np.pi * t_hours / 120.0) + 0.05 * rng.standard_normal(periods)
    if n_exog >= 2:
        exog["wind"] = 0.2 * np.cos(2 * np.pi * t_hours / 79.2) + 0.05 * rng.standard_normal(periods)

    columns = {}
    if kind == "elevation":
        obs = _tide(t_hours, [(1.2, 12.4206012, 10.0), (0.4, 12.0, 40.0),
                              (0.25, 23.93447213, 75.0), (0.18, 25.81933871, 120.0)])
        columns["observations"] = obs + 0.02 * rng.standard_normal(periods) + sum(0.15 * v for v in exog.values())
    else:
        u = _tide(t_hours, [(0.8, 12.4206012, 20.0), (0.2, 23.93447213, 60.0)])
        v = _tide(t_hours, [(0.5, 12.4206012, 110.0), (0.1, 25.81933871, 30.0)])
        columns["u"] = u + 0.02 * rng.standard_normal(periods) + sum(0.1 * x for x in exog.values())
        columns["v"] = v + 0.02 * rng.standard_normal(periods)
    columns.update(exog)
    return pd.DataFrame(columns, index=index)


def _roundtrip(df, prefix="in"):
    arrays = frame_to_arrays(df, prefix)
    return frame_from_arrays(arrays, prefix), arrays


@contextlib.contextmanager
def isolated_dir(ephemeris, isolated_home=False):
    previous_cwd = os.getcwd()
    previous_home = os.environ.get("HOME")
    work = Path(tempfile.mkdtemp(prefix="rtide_golden_", dir=previous_cwd))
    os.symlink(ephemeris, work / "de421.bsp")
    os.chdir(work)
    if isolated_home:
        os.environ["HOME"] = str(work / "home")
    try:
        yield work
    finally:
        os.chdir(previous_cwd)
        if isolated_home:
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home


def generate_features(out_dir, ephemeris):
    from rtide import RTide

    dest = out_dir / "features"
    dest.mkdir(parents=True, exist_ok=True)
    for cid, cfg in FEATURE_CONFIGS.items():
        kind, n_exog, cadence = cfg["data"]
        series = make_series(kind, n_exog, cadence, cfg["unit"])
        if cadence == "1h":
            series = series.iloc[:HOURLY_TRAIN_ROWS]
        ts, in_arrays = _roundtrip(series)
        kwargs = dict(cfg["kwargs"], save=False)

        with isolated_dir(ephemeris, cfg["isolated_home"]) as work:
            run_kwargs = {k: (str(work / "precomputed_cache") if v == CACHE_PLACEHOLDER else v)
                          for k, v in kwargs.items()}
            lat, lon = SITES[kind]
            model = RTide(ts, lat, lon)
            model.Prepare_Inputs(**run_kwargs)
            prepped = model.prepped_dfs

        np.savez_compressed(
            dest / f"{cid}.npz",
            kwargs=np.asarray(json.dumps(kwargs, sort_keys=True), dtype="U"),
            isolated_home=np.asarray(cfg["isolated_home"]),
            lat=np.asarray(lat), lon=np.asarray(lon),
            **in_arrays,
            **frame_to_arrays(prepped, "out"),
        )
        print(f"[golden] {cid}: features {prepped.shape} sample_rate={model.sample_rate}")


def generate_predictions(out_dir, ephemeris):
    from rtide import RTide

    dest_root = out_dir / "predictions"
    dest_root.mkdir(parents=True, exist_ok=True)
    for pid, (feature_id, train_extra) in PREDICTION_CONFIGS.items():
        cfg = FEATURE_CONFIGS[feature_id]
        kind, n_exog, cadence = cfg["data"]
        assert cadence == "1h", "prediction configs use hourly data"
        series = make_series(kind, n_exog, cadence, cfg["unit"])
        train_df, _ = _roundtrip(series.iloc[:HOURLY_TRAIN_ROWS])
        pred_df, pred_arrays = _roundtrip(series.iloc[HOURLY_TOTAL_ROWS - HOURLY_TRAIN_ROWS:])
        prep_kwargs = dict(cfg["kwargs"])  # save=True (default) so _inputs.pickle is written
        train_kwargs = dict({"standard_epochs": 3, "verbose": False}, **train_extra)
        lat, lon = SITES[kind]

        with isolated_dir(ephemeris, cfg["isolated_home"]) as work:
            _seed_everything(0)
            trained = RTide(train_df, lat, lon)
            trained.Prepare_Inputs(**prep_kwargs)
            trained.Train(**train_kwargs)

            fresh = RTide(pred_df, lat, lon)
            fresh.path = "./rtide_saves/RTide"
            fresh.Load_Model()
            fresh.Predict(pred_df)

            dest = dest_root / pid
            dest.mkdir(parents=True, exist_ok=True)
            saves = work / "rtide_saves"
            for suffix in ARTIFACT_SUFFIXES:
                shutil.copy2(saves / f"RTide{suffix}", dest / f"RTide{suffix}")

            np.savez_compressed(
                dest / "expected.npz",
                prep_kwargs=np.asarray(json.dumps(prep_kwargs, sort_keys=True), dtype="U"),
                train_kwargs=np.asarray(json.dumps(train_kwargs, sort_keys=True), dtype="U"),
                lat=np.asarray(lat), lon=np.asarray(lon),
                rtide_test=np.asarray(fresh.test_predictions["rtide_test"], dtype=np.float64),
                **pred_arrays,
                **frame_to_arrays(fresh.test_prediction_df, "df"),
            )
            print(f"[golden] {pid}: rtide_test {np.shape(fresh.test_predictions['rtide_test'])}, "
                  f"saved files {sorted(p.name for p in saves.iterdir())}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ephemeris", default="de421.bsp", type=Path,
                        help="path to de421.bsp (default: ./de421.bsp)")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    ephemeris = args.ephemeris.resolve()
    if not ephemeris.is_file():
        sys.exit(f"Ephemeris not found: {ephemeris}")
    if (out_dir / "MANIFEST.json").exists():
        sys.exit(f"Refusing to overwrite existing fixtures in {out_dir}. Golden fixtures are never regenerated.")

    root, sha = _check_baseline()
    print(f"[golden] baseline rtide from {root} @ {sha}")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_features(out_dir, ephemeris)
    generate_predictions(out_dir, ephemeris)

    files = {str(p.relative_to(out_dir)): _sha256(p)
             for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    manifest = {
        "description": "RTide golden-master fixtures generated from the unmodified 1.0.0 baseline.",
        "baseline_sha": sha,
        "rtide_imported_from": str(root),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "feature_configs": list(FEATURE_CONFIGS),
        "prediction_configs": list(PREDICTION_CONFIGS),
        "prediction_artifacts": ["RTide" + s for s in ARTIFACT_SUFFIXES],
        "notes": [
            "Indexes are stored as int64 ns (UTC) plus the original unit; *_us configs use microsecond "
            "indexes (pandas 3 date_range default), for which 1.0.0 infers sample_rate=1000.",
            "Prediction fixtures hold only the files needed by Load_Model()/Predict(); the "
            "_global_tide.csv feature cache is not included.",
        ],
        "files_sha256": files,
    }
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[golden] wrote {len(files)} files + MANIFEST.json to {out_dir}")


if __name__ == "__main__":
    main()
