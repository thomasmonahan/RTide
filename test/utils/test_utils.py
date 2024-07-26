import pytest
import pandas as pd
import numpy as np
import rtide
from rtide.utils import *
import tensorflow as tf


@pytest.fixture
def sample_signals():
    signal1 = np.array([1, 2, 3, 4, 5])
    signal2 = np.array([1, 2, 3, 4, 6])
    return signal1, signal2

def test_cosd():
    assert cosd(90) == pytest.approx(0)
    assert cosd(0) == pytest.approx(1)
    assert cosd(180) == pytest.approx(-1)

def test_sind():
    assert sind(90) == pytest.approx(1)
    assert sind(0) == pytest.approx(0)
    assert sind(180) == pytest.approx(0)

def test_custom_round():
    lags = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_rate = 2
    assert custom_round(lags, sample_rate) == [0.0, 0.0, 0.5, 0.5, 0.5]

    ## For example with 4 measurements per hour we should get lags every 0.25:
    lags = [0.534, 0.222, 0.699]
    sample_rate = 4
    assert custom_round(lags, sample_rate) == [0.5, 0.25, 0.75]

def test_compute_ssp(sample_signals):
    signal1, signal2 = sample_signals

    # Compute SSP using the function
    ssp = compute_ssp(signal1, signal2)

    # Since we don't have a predefined expected value for SSP,
    # we will perform basic checks to ensure it returns a float between 0 and 1.
    assert isinstance(ssp, tf.Tensor)
    assert ssp.numpy() >= 0.0
    assert ssp.numpy() <= 1.0

    # You can also check for a known output with simpler inputs
    signal1 = np.array([1, 1, 1, 1, 1])
    signal2 = np.array([1, 1, 1, 1, 1])
    expected_ssp = 0.0  # Since the signals are identical, the SSP should be 0

    ssp = compute_ssp(signal1, signal2)
    assert np.isclose(ssp.numpy(), expected_ssp), f"Expected {expected_ssp}, got {ssp.numpy()}"

    signal1 = np.array([1, 2, 3, 4, 5])
    signal2 = np.array([5, 4, 3, 2, 1])
    # Compute a rough expected SSP value using the formula for verification if needed

    ssp = compute_ssp(signal1, signal2)
    assert isinstance(ssp, tf.Tensor)
    assert ssp.numpy() >= 0.0
    assert ssp.numpy() <= 1.0

def test_calc_stats():
    signal1 = np.array([1, 1, 1, 1, 1])
    signal2 = np.array([1, 1, 1, 1, 1])
    r2, mse, mae, mape = calc_stats(signal1, signal2, verbose=False)
    assert r2 == 1.0
    assert mse == 0.0
    assert mae == 0.0
    assert mape == 0.0

