import os

# Must be set before TensorFlow / matplotlib are imported by any test module.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import keras
except ImportError:  # pragma: no cover - very old TF
    keras = tf.keras

REPO_ROOT = Path(__file__).resolve().parents[1]
EPHEMERIS = "de421.bsp"
SEED = 1234


def _find_ephemeris(cache_dir: Path) -> Path:
    """Locate a single de421.bsp, downloading it at most once into the pytest cache."""
    candidates = []
    if os.environ.get("RTIDE_EPHEMERIS"):
        candidates.append(Path(os.environ["RTIDE_EPHEMERIS"]))
    candidates.append(REPO_ROOT / EPHEMERIS)
    try:
        import skyfield_data

        candidates.append(Path(skyfield_data.get_skyfield_data_path()) / EPHEMERIS)
    except ImportError:
        pass
    candidates.append(cache_dir / EPHEMERIS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    from skyfield.api import Loader

    Loader(str(cache_dir))(EPHEMERIS)
    return (cache_dir / EPHEMERIS).resolve()


@pytest.fixture(scope="session")
def ephemeris_path(request, tmp_path_factory):
    cache = getattr(request.config, "cache", None)
    if cache is not None:
        cache_dir = Path(cache.mkdir("rtide_ephemeris"))
    else:
        cache_dir = tmp_path_factory.mktemp("rtide_ephemeris")
    return _find_ephemeris(cache_dir)


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch, ephemeris_path):
    """Run every test in its own directory (own ./rtide_saves/) with the ephemeris available."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / EPHEMERIS
    try:
        target.symlink_to(ephemeris_path)
    except OSError:
        shutil.copyfile(ephemeris_path, target)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    yield tmp_path

    plt.close("all")
