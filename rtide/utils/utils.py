import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error
import tensorflow as tf
from keras.layers import Layer
import pickle


def cosd(x):
  """
  Helper function to compute cos going from deg to rad

  Inputs:
  --------
  x: float
    Angle in degrees

  Returns:
  --------
  cos(x)
  """
  return np.cos(np.deg2rad(x))

def sind(x):
  """
  Helper function to compute sin going from deg to rad

  Inputs:
  --------
  x: float
    Angle in degrees

  Returns:
  --------
  sin(x)
  """
  return np.sin(np.deg2rad(x))

def calc_stats(test_predictions, test_labels, verbose = True):
  r2 = r2_score(test_predictions, test_labels)
  mse = mean_squared_error(test_predictions, test_labels)
  mae = mean_absolute_error(test_predictions, test_labels)
  mape = mean_absolute_percentage_error(test_predictions, test_labels)

  if verbose:
    print(f"r2: {r2} MSE: {mse} MAE : {mae} MAPE : {mape}")
  return r2, mse,mae,mape


def fit_trend_initial_coeffs(t_norm, y_scaled, trend):
    """
    Fit initial polynomial trend coefficients in *scaled Y space*.

    Parameters
    ----------
    t_norm : (N,) array
        Normalized time in [0,1] (or extrapolated) used by the model.
    y_scaled : (N,) or (N,1) or (N,2) array
        Scaled targets (i.e., output of scaler_Y.fit_transform(train_Y)).
    trend : {'linear','quadratic'}

    Returns
    -------
    dict with keys c0,c1,(c2) where each value is shape (n_outputs,)
    """
    t = np.asarray(t_norm, dtype=np.float64).reshape(-1)
    Y = np.asarray(y_scaled, dtype=np.float64)

    # Make Y always 2D: (N, n_outputs)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    elif Y.ndim != 2:
        raise ValueError(f"y_scaled must be 1D or 2D, got shape {Y.shape}")

    if Y.shape[0] != t.shape[0]:
        raise ValueError(f"Length mismatch: t has {t.shape[0]} rows, Y has {Y.shape[0]}")

    if trend == "linear":
        A = np.column_stack([np.ones_like(t), t])          # (N,2)
    elif trend == "quadratic":
        A = np.column_stack([np.ones_like(t), t, t**2])    # (N,3)
    else:
        raise ValueError(f"Unsupported trend={trend!r}. Use None, 'linear', or 'quadratic'.")

    # Multi-output least squares: coef shape (p, n_outputs)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)

    out = {
        "c0": coef[0].astype(np.float32).reshape(-1),
        "c1": coef[1].astype(np.float32).reshape(-1),
    }
    if coef.shape[0] == 3:
        out["c2"] = coef[2].astype(np.float32).reshape(-1)

    return out


def custom_round(lags, sample_rate):
    """
    Round lags to the nearest multiple of 1/sample_rate.

    Parameters:
    lags (list of floats): The lags to be rounded.
    sample_rate (int): The multiplier value defining the spacing between lags.

    Returns:
    list of floats: The rounded lags.

    """
    spacing = 1 / sample_rate
    rounded_lags = [round(lag / spacing) * spacing for lag in lags]
    return rounded_lags

def compute_ssp(signal1, signal2):
    """
    Surface similarity loss function: (bounded between 0 and 1)

    - provides equal penalization of amplitude/phase

    Computed based on the paper:
    https://doi.org/10.1016/j.neunet.2022.09.023

    inputs:
    --------
    signal1: numpy array
    signal2: numpy array

    returns:
    --------
    ssp: float (0,1)
    """
    # Compute the FFT of the signals
    fft_signal1 = tf.signal.rfft(signal1)
    fft_signal2 = tf.signal.rfft(signal2)

    # Calculate the difference in the frequency domain
    numerator = tf.square(tf.abs(fft_signal1 - fft_signal2))
    denom1 = tf.square(tf.abs(fft_signal1))
    denom2 = tf.square(tf.abs(fft_signal2))

    # Sum the absolute values of the differences across all frequencies
    numerator_sum = tf.reduce_sum(numerator)
    denom_sum1 = tf.reduce_sum(denom1)
    denom_sum2 = tf.reduce_sum(denom2)

    ssp = 0.5 * numerator_sum / (denom_sum1 + denom_sum2)

    return ssp

def save_inputs_to_pickle(inputs, path):
  """
      Helper function to save inputs to a json file.
  """
  with open(path +'_inputs.pickle', 'wb') as file:
      pickle.dump(inputs, file, pickle.HIGHEST_PROTOCOL)

def load_inputs_from_pickle(path):
  """
      Helper function to save inputs to a json file.
  """
  with open(path +'_inputs.pickle', 'rb') as file:
      dictionary = pickle.load(file)
  return dictionary

@tf.keras.utils.register_keras_serializable(package="rtide")
class AddLayer(Layer):
  """Additive output layer for *scalar* RTide models (single output).

  RTide's original "response method" uses an additive final layer that sums the
  last hidden representation across features to produce a single scalar output.

  For multi-output problems (e.g., tidal currents u/v), **do not** use this
  layer; use the multi-output head provided in `models.py` instead.
  """

  def __init__(self, n_outputs: int = 1, **kwargs):
      super().__init__(**kwargs)
      self.n_outputs = int(n_outputs)

  def call(self, inputs):
      if self.n_outputs != 1:
          raise ValueError(
              "AddLayer only supports n_outputs=1. "
              "Use the multi-output head in models.py for n_outputs>1."
          )
      return tf.reduce_sum(inputs, axis=-1, keepdims=True)

  def get_config(self):
      config = super().get_config()
      config.update({"n_outputs": self.n_outputs})
      return config
