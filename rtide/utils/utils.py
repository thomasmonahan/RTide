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


class AddLayer(Layer):
  """
    Custom layer that adds all outputs of NN, rather than using a standard dense connection.
  """
  def __init__(self, **kwargs):
      super(AddLayer, self).__init__(**kwargs)

  def call(self, inputs):
      return tf.reduce_sum(inputs, axis=-1, keepdims=True)
