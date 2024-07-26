import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import utide
import shap

import scipy.io
from skyfield.api import N,S,E,W, wgs84
from skyfield.api import load
from tqdm import tqdm, tqdm_notebook
import math

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Layer, Input, Add
from keras import regularizers
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from .utils import *

class RTide:
  """
  RTide class for empirical estimation of the oceanic response to a set of 
  input functions from measurement data. Uses data-driven methods to estimate
  the response based on the method outlined in: 

  https://doi.org/10.21203/rs.3.rs-3289185/v2

  Parameters
  ----------
  ts : DataFrame
      time series data containing the observations and optionally other input 
      functions. Must have datetime index, though data can be unevenly spaced.
  lat: float
      latitude of the station
  lon: float
      longitude of the station 

  Examples
  ----------
  >>> from rtide import RTide
  >>> import pandas as pd
  >>> import numpy as np
  >>> constit_freqs = [8.05114007e-02, 7.89992487e-02, 8.33333333e-02, 8.20235526e-02]
  >>> mplitudes = [2.69536915e+00, 5.72051193e-01, 4.07506583e-01, 1.77388101e-01]
  >>> phases = [100.17759146,  67.89243773, 139.00250991, 137.86574464]
  >>> times = pd.date_range(start = '2024-01-01 00:00:00+00:00', periods=24*365, freq="1H",tz='UTC')
  >>> time_vals = times.to_julian_date().to_numpy()
  >>> individual_tides = []
  >>> for i,j in enumerate(constit_freqs):
  >>>    individual_tides.append(amplitudes[i]*np.cos(24*j*np.array(time_vals) * 2 * np.pi + (phases[i] *np.pi /180)))
  >>> tide = np.sum(individual_tides, axis = 0) + 0.1*np.random.normal(len(individual_tides))
  >>> df = pd.DataFrame({'observations':tide}, index = times)
  >>> model = RTide(df, lat = 44.9062, lon = -66.996201)
  >>> model.Prepare_Inputs()
  >>> model.Train()
  """
  def __init__(self, ts, lat, lon):
      self.ts = ts # time series
      
      self.lat = lat # latitude
      self.lon = lon # longitude

      self.M_E = 5.9722*10**24 ## Mass Earth
      self.M_M = 7.3 *10**22 ## Mass Moon
      self.M_S = 1.989 * 10**30 ## Mass Sun
      self.E_r = 6371.01*10**3 ## Earth Radius (denoted a in Response Paper)
      self.solar_constant = 1.946/100  ## Note in cm
      self.multi = True if np.shape(ts)[1] > 1 else False
      self._validate_inputs()
  
  def _validate_inputs(self):
      if not isinstance(self.ts.index, pd.DatetimeIndex):
          raise ValueError("The DataFrame does not have a DatetimeIndex.")
      if not isinstance(self.lat, (int, float)):
          raise ValueError("Latitude must be a number.")
      if not isinstance(self.lon, (int, float)):
          raise ValueError("Longitude must be a number.")
  
  def Spherical_Harmonic(self,degree, order):
    '''PreDefined Legendre Polynomials for speed:

       Stored in a nested list defined by:
       P_vals where i is the degree of the associated Legendre polynomials.
       Each value in the list is the Legendre polynomial of degree i and order
       equal to its index value in the list.

       Theta = geographical colatitude of station
       Lambda = longitude of station

       For Example:
       P_2 = [P_2^0,P_2^1, P_2^2]

       Inputs:
       --------
       degree: int (0,3)
       order: int  (0,3)

       Returns
       --------
       : numpy array
          complex spherical harmonics at associated degree and order
       '''
    P_0 = [1]
    P_1 = [cosd(self.lat), sind(self.lat)]
    P_2 = [(1.5)*cosd(self.lat)**2 - .5, 3*sind(self.lat)*cosd(self.lat), 3*sind(self.lat)**2]
    P_3 = [(2.5)*cosd(self.lat)**3 - 1.5*cosd(self.lat), 1.5*sind(self.lat) * (5*cosd(self.lat)**2 -1), 15*sind(self.lat)**2 * cosd(self.lat), 15* sind(self.lat)**3]
    P_vals = [P_0,P_1,P_2,P_3]
    try:
      Y = (-1)**(order) *(2*degree + 1 / (4*np.pi))**(1/2) * (math.factorial(degree-order)/math.factorial(degree+order)) *cosd(self.lat) * np.exp(1j*order*self.lon)*P_vals[degree][order]
      return Y
    except ValueError:
      print("Order must be less than or equal to the degree")


  def Radiational(self, degree, order):
    '''
    Computes the Global Radiation Function for all zenith angles

    Inputs:
    --------
    degree: int (1,2)
    order: int  (0,degree)

    Returns
    --------
    : numpy array

    Sum of Radiational Forces from the Sun at specified degree, and order.

    '''
    degree= int(degree)
    if degree > 2:
      raise ValueError('Currently only equipped to handle k = 0,1,2.')
    parallax = 1.0/23455.0
    k_n = [1/4 + (1/6)*parallax, (1/2) + (3/8)*parallax, (5/16) + (1/3)*parallax]

    SPH = self.Spherical_Harmonic(degree, order)
    k = k_n[degree]

    zenith_sun = self.astro['zenith_sun']
    station2sun = self.astro['station2sun']
    mu_sun = np.array(self.mu_sun)

    # Calculate initial radiation values
    rad = self.solar_constant * (self.mean_r_sun / station2sun) * k * mu_sun**degree

    # Set radiation to zero where zenith_sun is between 90 and 180
    mask = (zenith_sun >= 90) & (zenith_sun <= 180)
    rad[mask] = 0

    # Normalize by spherical harmonics
    return rad / SPH

  def Gravitational(self, degree, order):
    """Computes the Gravitational Input function for each t of the observed time series:

       Assumes that the combined gravitational input function is equal to the sum of the
       lunar and solar input functions.

       **Currently using predefined Legendre functions as defined in Munk & Carwright

       Inputs
       --------
       degree: int (2,3)
       order: int  (0,degree)
       lag_ind: index associated with input lag function (0 if even spaced)

       Returns
       --------
       : DataFrame

       Sum of gravitational forces from the moon and sun at specified degree.
       """
    degree= int(degree)
    if degree > 3:
       raise ValueError('Currently only equipped to handle k = 0,1,2,3.')
    SPH = self.Spherical_Harmonic(degree, order)

    earth2moon = self.astro['earth2moon']
    earth2sun = self.astro['earth2sun']

    ## Computing associated k values
    K_n_Moon = self.E_r * (self.M_M / self.M_E) * (self.E_r/ earth2moon)**(degree+1)
    K_n_Sun = self.E_r * (self.M_S / self.M_E) * (self.E_r/ earth2sun)**(degree+1)

    grav_Moon= K_n_Moon * (self.mean_r_moon / earth2moon)**(degree+1) * self.Lunar_Legendre[degree]
    grav_Sun= K_n_Sun * (self.mean_r_sun / earth2sun)**(degree+1)  * self.Solar_Legendre[degree]

    moon_plus_sun =  grav_Moon + grav_Sun

    return moon_plus_sun / SPH ## Normalizing by spherical harmonics

  def Global_Tide(self):
    ''' Computes global tide function according to specified input functions
        Note** Radiational function commences with P_1(mu) and gravitational
        commences at P_2(mu)

        Going to have the primary output still be the dictionary, however,
        the respective functions will also be added to self.helper_df. One of these
        functionalities can be removed for increased speed.

        Returns
        --------
        : DataFrame
        Format: {'Radiational':{degree: Associated Radiational Input Values,..., lag}
                 'Gravitational':{degree: Associated Gravitational Input Values,..., lag}}

        Values are numpy arrays containing the instantaneous values of the input function for each time.

        returned as: self.global_tide
        '''
    self.Astro()
    self.global_tide = {'Radiational':{i: {k: 0 for k in range(1,i+1)} for i in range(1,3)}, 'Gravitational':{i:{k: 0 for k in range(1,i+1)} for i in range(2,4)}}
    for i in tqdm(self.global_tide, leave=False, desc='Computing Global Tide Function'):
      for j in self.global_tide[i]:
        if i == 'Radiational':
          for k in range(1,j+1):
            self.global_tide[i][j][k] = self.Radiational(degree = j, order = k) ## Computes the Radiational Function at each Solar Zenith Angle For degree j.

        elif i == 'Gravitational':
          for k in range(1,j+1):
            self.global_tide[i][j][k] = self.Gravitational(degree = j, order = k) ## Computes the Gravitational Function at each Solar/Lunar Zenith Angle  for degree j

  def Astro(self):
    """
     Computes the associated astronomical positions and angles of the Moon and Sun
     relative to the station and center of the Earth.

     Returns
     --------
      : List of dictionaries

      Dictionaries correspond to the input functions at different time lags.
      If evenly spaced then this is the list of all timelags.

      returned as self.astro:

    """
    ## Converting time index to appropriate format
    timestouse =  self.padded_ts.index
    tindex = timestouse.to_pydatetime()

    tscale = load.timescale()
    times = tscale.utc(tindex)
    ## Loading planet objects from skyfield
    planets = load('de421.bsp')
    earth, moon, sun = planets['earth'], planets['moon'], planets['sun']

    ## Computing Distances:
    earth_moon = [(moon.at(t) - earth.at(t)).distance().m for t in times]## Returns Moons distance in meters
    earth_sun = [(sun.at(t) - earth.at(t)).distance().m for t in times] ## Returns Suns distance in meters

    ## Computing Angles:
    moon_stat_dist = []
    sun_stat_dist = []
    zenith_moon = []
    zenith_sun = []
    azimuth_moon = []
    azimuth_sun = []
    for t in tqdm(times):
      station_location = earth + wgs84.latlon(self.lat, self.lon)
      astro_moon = station_location.at(t).observe(moon)
      astro_sun = station_location.at(t).observe(sun)

      app_moon = astro_moon.apparent()
      app_sun = astro_sun.apparent()


      alt_moon, az_moon, stat_dist_moon = app_moon.altaz()
      alt_sun, az_sun, stat_dist_sun = app_sun.altaz()


      ## Zenith angle is equal to +90 for alt.
      zenith_moon.append(90 - alt_moon.degrees)
      zenith_sun.append(90 - alt_sun.degrees)

      azimuth_moon.append(az_moon.degrees)
      azimuth_sun.append(az_sun.degrees)

      moon_stat_dist.append(stat_dist_moon.m)
      sun_stat_dist.append(stat_dist_sun.m)

    self.mu_moon = cosd(zenith_moon) ## Mu Moon
    self.mu_sun = cosd(zenith_sun) ## Mu Sun
    self.alpha = zenith_sun ## Solar Zenith Angle
    self.mean_r_sun = np.sum(earth_sun) / len(earth_sun) ## Average Distance to sun
    self.mean_r_moon = np.sum(earth_moon) / len(earth_moon)

    ## Associated Legendre Functions
    self.Solar_Legendre =  [1,self.mu_sun, [(3/2)*mew**2 - (1/2) for mew in self.mu_sun], [(5/2)*mew**3 - (3/2)*mew for mew in self.mu_sun]]
    self.Lunar_Legendre = [1,self.mu_moon, [(3/2)*mew**2 - (1/2) for mew in self.mu_moon], [(5/2)*mew**3 - (3/2)*mew for mew in self.mu_moon]]
    # Creating the dataframe
    self.astro = {
        'station2moon': np.array(moon_stat_dist),
        'station2sun': np.array(sun_stat_dist),
        'earth2moon': np.array(earth_moon),
        'earth2sun': np.array(earth_sun),
        'zenith_moon': np.array(zenith_moon),
        'zenith_sun': np.array(zenith_sun),
        'azimuth_moon': np.array(azimuth_moon),
        'azimuth_sun': np.array(azimuth_sun)
    }

  def Prep(self, uniform, symmetrical, multivariate_lags):
    """
      Function to compute appropriate lags and pad the time series.

      Inputs:
      --------
      uniform: False or [s,tau] <- s = num lags, tau = lag spacing (hours)

      symmetrical: Bool ; whether or not to include positive lags

      multivariate_lags: False, or other options
          Specifies whether special lags should be given for multivariate inputs.

    """
    def neg(lst):
      ## Helper function to return negative/realtime lags only
      return [x for x in lst if x <= 0]

    if uniform:
      ## Uniform lag embedding. Standard approach used in MC:
      ## s = 3, tau = 2 days = 48 hours
      assert isinstance(uniform, list), "uniform must be a list"
      assert len(uniform) == 2, "uniform must be a list of length 2"
      assert isinstance(uniform[0], int), "uniform[0] must be an integer"

      nonuni = []
      for s in range(-uniform[0],uniform[0]+1):
        nonuni.append(self.sample_rate*s*uniform[1])

      if symmetrical:
        self.const_names = [str(ba) for ba in nonuni]
      else:
        nonuni = neg(nonuni)
        self.const_names = [str(ba) for ba in nonuni]


    else:
      ## Standard non-uniform RTide Lags:
      constits = ['K1', 'M2', 'M3', 'M4', '2MK5', '2SK5', 'M6', '3MK7', 'M8']
      frequencies = [0.04178075, 0.0805114,  0.1207671,  0.1610228,  0.20280355, 0.20844741, 0.2415342,  0.28331495, 0.3220456]

      ## Sorting Lags and Determining Largest Constit
      nonuni = []
      for i,j in enumerate(frequencies):
        if (len(nonuni) != 0) and (nonuni[-1] == -self.sample_rate*2*np.pi / j):
          pass
        else:
          nonuni.append(-self.sample_rate*2*np.pi / (j))
      nonuni.append(0)

      for i in np.flip(frequencies):
        if nonuni[-1] == self.sample_rate*2*np.pi / (i):
          pass
        else:
          nonuni.append(self.sample_rate*2*np.pi / (i))


      ## Sorting names for positive and negative lags
      new_vals = constits.copy()

      new_vals.append('real') # adding real-time lag (tau = 0)
      if not symmetrical:
        new_vals2 = []
        nonuni= neg(nonuni)
        self.const_names = new_vals

      else:
        new_vals2 = list(np.flip(constits))
        self.const_names = new_vals+ new_vals2


    self.max_lag = abs(min(nonuni)/self.sample_rate)
    if self.sample_rate <= 1:
      roundval = 1
    else:
      roundval = self.sample_rate
    lagstouse = [lag / self.sample_rate for lag in nonuni]
    self.nonunilags = custom_round(lagstouse, roundval)
    self.Compute_Time_Lags()

    if self.multi:
      if not multivariate_lags:
        self.multivariate_lags = [0]
      elif multivariate_lags == 'standard':
        ## Need to prevent using realtime data twice: (This is due to the fact that the realtime values are their own column in the dataframe)
        index_of_zero = self.nonunilags.index(0)
        multi_lags_list = self.nonunilags.copy()
        # Remove the zero from the list
        multi_lags_list.pop(index_of_zero)
        self.multivariate_lags = multi_lags_list
        ## Lagging multivariate input functions
      elif multivariate_lags == 'negative':
        def neg(lst):
          return [x for x in lst if x < 0]
        self.multivariate_lags = neg(self.nonunilags)
      elif isinstance(multivariate_lags, list):
        print("Custom Lags Are:", multivariate_lags)
        self.multivariate_lags = multivariate_lags
      else:
        print('only including real_time multivariate lags')
        self.multivariate_lags = [0]

  def Prepare_Inputs(self, **kwargs):
    """
        Function to compute and format the input functions used in a response analysis.

        Inputs:
        --------
        kwargs: dictionary
            A dictionary of parameters with the following keys:
                - uniform_lags: False or [s,tau] <- s = num lags, tau = lag spacing (hours)
                    Used to perform uniform lag embedding. Standard approach used in MC: s = 3, tau = 2 days = 48 hours

                - multivariate_lags: False or [lags] <-lags are given in (hours)
                    Option to specify a separate set of lags for the multivariate inputs. If False, standard nonunilags are used.

                - self_prediction: False or
                    Option to perform self-prediction; using past observations to predict future values.

                - radiational: bool (default == True)
                    Option to include the radiational input function.

                - symmetrical: bool
                    Option to perform symmetrical lag analysis (using both positive and negative lags)

                - path: str or None
                    Path to save the json file containing input parameters

                - sample_rate: Optional (float or int)
                    Sampling rate of the data provided in **samples per hour.

                - prediction: bool
                    Used to compute lags for prediction. Not intended to be called by users directly, automatically used in the Predict function.

        Outputs:
        --------
        self.prepped_dfs: DataFrame
            fully prepared response input functions
    """

    # Default values
    defaults = {
        'uniform_lags': False,
        'multivariate_lags': False,
        'self_prediction': False,
        'radiational': True,
        'symmetrical': False,
        'path': None,
        'sample_rate': None,
        'prediction': False,
        'save': True
    }

    # Update defaults with kwargs
    inputs = {**defaults, **kwargs}

    if inputs['prediction']:
      if inputs['save']:
        inputs = load_inputs_from_pickle(self.path)
      else:
        print("warning save is set to false, default inputs are being used.")
      radiational = inputs['radiational']
      multivariate_lags = inputs['multivariate_lags']
      self_prediction = inputs['self_prediction']
      path = inputs['path']
      symmetrical = inputs['symmetrical']
      sample_rate = inputs['sample_rate']
      uniform_lags = inputs['uniform_lags']
      prediction = True
      save = inputs['save']
    else:
      # Assign inputs to instance variables
      radiational = inputs['radiational']
      multivariate_lags = inputs['multivariate_lags']
      self_prediction = inputs['self_prediction']
      path = inputs['path']
      symmetrical = inputs['symmetrical']
      sample_rate = inputs['sample_rate']
      uniform_lags = inputs['uniform_lags']
      prediction = False
      save = inputs['save']

    if inputs['sample_rate']:
      assert isinstance(sample_rate, float) or isinstance(sample_rate, int), "sample_rate must be a float or int (number of measurements per hour) if the data is even spaced. Note gaps can still exist, however, the sampling must have a set frequency"
      self.sample_rate = sample_rate
    else:
      # Compute the differences between consecutive elements
      time_diffs = self.ts.index.to_series().diff().dropna()

      # Find the minimum difference
      min_distance = time_diffs.min()
      hours = min_distance.total_seconds() / 3600.0

      self.sample_rate = (1/hours)
    save_directory = './rtide_saves/'
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    if path is not None:
      save = True
    
    self.path = save_directory + path if path is not None else save_directory+ './RTide'

    try:
      if not save:
        raise print('Save is set to false, recomputing')
      prepped_dfs = pd.read_csv(f'{self.path}_global_tide.csv', index_col=0)
      inputs_used = load_inputs_from_pickle(self.path)
      if inputs_used != inputs:
        raise print('Inputs are not equal, recomputing based on new input criteria.')
      if prediction == True:
        raise print('Computing input functions for prediction')

      self.Prep(uniform_lags, symmetrical, multivariate_lags)
      prepped_dfs.index = self.ts.index
      self.prepped_dfs = prepped_dfs
    except Exception as e:
      print('Input function path either has not specified, has not yet been computed, or different inputs were provided. Computing now...')
      print(f'Actual error was {e}')
      self.Prep(uniform_lags, symmetrical, multivariate_lags)
      self.Global_Tide()
      ts_to_concat = [self.ts]
      if radiational == False:
        input_types = ['Gravitational']
      else:
        input_types = ['Gravitational', 'Radiational']
      for input_type in input_types:
        for degree in self.global_tide[input_type]:
          for order in self.global_tide[input_type][degree]:
            for lagbeingused in self.nonunilags:
              gtide = self.global_tide[input_type][degree][order].real
              og_index = self.ts.index
              lagged_index = self.padded_ts.index - pd.Timedelta(hours=lagbeingused)
              # Convert the indexes to sets for faster intersection
              second_set = set(lagged_index)
              og_set = set(og_index)
              # Find the common indexes using set intersection
              common_indexes = og_set & second_set
              # Use boolean indexing to filter the measurements
              mask = np.array([ts in common_indexes for ts in lagged_index])
              filtered_gtide = gtide[mask]


              ts_to_concat.append(pd.DataFrame({f'{input_type}_{str(degree)}^{str(order)}_{lagbeingused}': filtered_gtide}, index = self.ts.index))
      if self.multi:
        if self.multivariate_lags == [0]:
          pass
        else:
          for inputfunc in self.ts.columns[1:]:
            multi = self.ts[inputfunc].to_numpy()
            og_index = self.ts.index
            for lagbeingused in self.multivariate_lags:
              lagged_index = self.ts.index - pd.Timedelta(hours=lagbeingused)
              inter_df = pd.DataFrame({inputfunc+str(lagbeingused):multi},index = lagged_index)

              merged = inter_df.reindex(self.ts.index).merge(self.ts, left_index=True, right_index=True, how='left')

              ts_to_concat.append(pd.DataFrame({f'{inputfunc}_{lagbeingused}': merged[inputfunc+str(lagbeingused)].to_numpy()}, index = self.ts.index))
        
      prepped_dfs = pd.concat(ts_to_concat, join='outer', axis=1)
      if prepped_dfs.dropna().shape[0] == 0:
        raise ValueError('No common indexes found between the input functions. Please ensure that the input functions have the same time index as the observations/ change the lag values.')
      if prediction:
        self.prediction_dfs = prepped_dfs
        if save:
          prepped_dfs.to_csv(f'{self.path}_global_tide_prediction.csv')
      else:
        self.prepped_dfs = prepped_dfs
        if save:
          self.prepped_dfs.to_csv(f'{self.path}_global_tide.csv')
          save_inputs_to_pickle(inputs, self.path)

  def Compute_Time_Lags(self):
    """
    Function to compute non-uniformly spaced time stamps associated with each time lag:

    """
    times = self.ts.index.to_numpy() ##pandas datetime index
    times_with_lags = []
    for lagval in self.nonunilags:
      time_lag = pd.Timedelta(hours=lagval)
      shifted_times = times + time_lag
      # Convert the NumPy array to a Pandas DatetimeIndex
      shifted_index = pd.DatetimeIndex(shifted_times)
      # Convert the DatetimeIndex to an array of Python datetime objects
      times_with_lags.append(shifted_index)
    # Combine the list of TimedeltaIndex objects into one
    combined_timedelta_index = times_with_lags[0]
    for t_index in times_with_lags[1:]:
        combined_timedelta_index = combined_timedelta_index.union(t_index)

    padded_df = pd.DataFrame(index=combined_timedelta_index)
    # Merge the existing DataFrame with the padded DataFrame based on the index
    padded_df = padded_df.merge(self.ts, how='left', left_index=True, right_index=True)
    self.padded_ts = padded_df


  def Train(self, **kwargs):
    """
    Function to train RTide model using a standard neural network.

    Inputs:
    --------
    kwargs: dictionary
        A dictionary of parameters with the following keys:
            - lr: float
                Learning rate for the training process.

            - loss: str
                Loss function to be used ('MAE' or any other loss function).

            - train_epochs: int
                Number of training epochs for the initial training phase.

            - train_epochs2: int
                Number of training epochs for the second training phase.

            - verbose: bool
                Option to print training progress and details.

            - regularization_strength: float
                Strength of the regularization term.

            - hidden_nodes: str or int
                Configuration for hidden nodes ('standard' or a specific number).

            - depth: int
                Depth of the neural network (number of hidden layers).

            - early_stoppage: bool
                Option to enable early stopping during training.

            - save_weights: bool
                Option to save the weights of the trained model.

    Outputs:
    --------
    self.model: Model
        Trained neural network model.

    self.train_predictions:  dictionary
        A dictionary of training predictions with the following keys:
            - rtide_train: np.array
                Array containing the predictions for the RTide model.

            - train_observations: np.array
                Array containing the actual observations.

            - utide_tide_train: np.array
                Array containing standard UTide OLS predictions.

            - rtide_tide_train: np.array
                Array containing RTide harmonic equivalent predictions.

            - irls_utide_train: np.array
                Array containing UTide IRLS predictions.
    
    self.train_prediction_df: DataFrame
        DataFrame containing the hindcasted training predictions alongside 
        input observations.

    """

    # Default values
    defaults = {
        'lr': 0.0001,
        'loss': 'MAE',
        'linear_epochs': 0,
        'standard_epochs': 500,
        'verbose': True,
        'regularization_strength': 0.0,
        'hidden_nodes': 'standard',
        'depth': 3,
        'early_stoppage': False,
        'save_weights': True,
    }

    # Update defaults with kwargs
    inputs = {**defaults, **kwargs}

    # Assign inputs to instance variables
    lr = inputs['lr']
    loss = inputs['loss']
    train_epochs = inputs['linear_epochs']
    train_epochs2 = inputs['standard_epochs']
    verbose = inputs['verbose']
    regularization_strength = inputs['regularization_strength']
    hidden_nodes = inputs['hidden_nodes']
    depth = inputs['depth']
    early_stoppage = inputs['early_stoppage']
    save_weights = inputs['save_weights']

    if verbose:
      print('#### Model Overview ####')
      print('Learning Rate:', lr)
      print('Loss:', loss)
      print('Linear Epochs:', train_epochs)
      print('Standard Epochs:', train_epochs2)
      print('Regularization:', regularization_strength)
      print('Number of Layers:', depth)
      print("Multi:", self.multi)
      print("Save Weights:", save_weights)

    df = self.prepped_dfs.dropna()

    dataset = df.values
    num_features = np.shape(dataset)[1]
    try:
      short_utide = utide.solve(
        df[:].index,
        df[:].observations,
        lat=self.lat,
        method="ols",
        conf_int="none",
        verbose=False,
        trend = False,
        nodal = False,
        phase = 'raw'
        )

      short_pred = utide.reconstruct(df[:].index, short_utide, verbose = False)
      utide_preds= short_pred['h']
    except Exception as e:
      print(f"UTide error, this is not a problem with RTide. Do not use these values: {e}")
      if train_epochs != 0:
        raise print("^see above error message. Something has gone wrong with UTide's svd convergence and the linear training procedure cannot be run. To rerun, simply set train_epochs = 0")
      utide_preds = np.zeros(len(df[:].observations.to_numpy()))


    ### Setting up Linear Training
    if (self.multi == False) and (train_epochs > 0):
      train_X_LIN = dataset[:len(utide_preds), 1:num_features]

      train_Y_LIN = utide_preds
    else:
      train_X_LIN_standard = dataset[:len(utide_preds), np.shape(self.ts)[1]:num_features-(np.shape(self.ts)[1]-1)*(len(self.nonunilags) + 1)] ### All non self.ts values
      train_X_LIN_multi_start = np.zeros(np.shape(dataset[:len(utide_preds), 1:np.shape(self.ts)[1]])) ### All non self.ts values
      train_X_LIN_multi_end = np.zeros(np.shape(dataset[:len(utide_preds), num_features- (np.shape(self.ts)[1]-1)*(len(self.nonunilags) + 1) : num_features])) ### All non self.ts values

      train_X_LIN = np.concatenate((train_X_LIN_multi_start, train_X_LIN_standard, train_X_LIN_multi_end), axis=1)
      train_Y_LIN = utide_preds

    ### Setting up Global Training
    train_X = dataset[:, 1:num_features]

    train_Y = dataset[:, 0:1]

    test_X = dataset[:, 1:num_features]
    test_Y = dataset[:, 0:1]

    scaler_X_LIN = StandardScaler()
    scaled_train_X_LIN = scaler_X_LIN.fit_transform(train_X_LIN.reshape(-1, 1)).reshape(train_X_LIN.shape)

    scaler_Y_LIN = StandardScaler()
    scaled_train_Y_LIN = scaler_Y_LIN.fit_transform(train_Y_LIN.reshape(-1, 1))

    self.scaler_X = StandardScaler()
    scaled_train_X = self.scaler_X.fit_transform(train_X.reshape(-1,1)).reshape(train_X.shape)
    scaled_test_X = self.scaler_X.transform(test_X.reshape(-1,1)).reshape(test_X.shape)
    self.scaler_Y = StandardScaler()
    scaled_train_Y = self.scaler_Y.fit_transform(train_Y.reshape(-1, 1))

    input_dims = num_features-1
    if hidden_nodes == 'standard':
        hidden_nodes = num_features-1

    if not early_stoppage:
      early_stoppage = train_epochs2
    custom_early_stopping = EarlyStopping(
      monitor='val_loss',
      patience=early_stoppage,
      min_delta=0.0001,
      mode='min'
    )

    if early_stoppage < 10:
      reduce_lr_patience = early_stoppage
    elif early_stoppage == train_epochs2:
      reduce_lr_patience = 30
    else:
      reduce_lr_patience = early_stoppage - 10

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=reduce_lr_patience, min_lr=0.00000001)


    # Set the regularization strengths (you can adjust these values)
    l1_strength = 0
    l2_strength = regularization_strength

    # Create the model
    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=input_dims, activation='tanh', name='nonlinear_layer1', kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength)))
    for layer_number in range(2,depth+1):
      model.add(Dense(input_dims, activation='tanh', name=f'nonlinear_layer_{layer_number}', kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength)))

    model.add(AddLayer(name='add_layer'))

    if loss == 'SSP':
      model.compile(loss=compute_ssp, optimizer = tf.keras.optimizers.Adam(learning_rate=lr))
    else:
      model.compile(loss=loss, optimizer = tf.keras.optimizers.Adam(learning_rate=lr))

    # Assuming scaled_train_X and scaled_train_Y are your training data
    # Shuffle the training data (note the keras shuffle param shuffles AFTER the validation split (hence, important to shuffle first))
    scaled_train_X_shuffled, scaled_train_Y_shuffled = shuffle(scaled_train_X, scaled_train_Y, random_state=0)

    # Compile Model
    if train_epochs > 0:
      history = model.fit(scaled_train_X_LIN, scaled_train_Y_LIN, epochs=train_epochs, batch_size=32, verbose = 2, shuffle = True, validation_split = .15, callbacks=[custom_early_stopping,reduce_lr])
      plt.plot(history.history['loss'], label = 'Loss U')
      plt.plot(history.history['val_loss'], label = 'Val Loss U')
      plt.legend()
      plt.title('Linear Training')
      plt.show()

    history2 = model.fit(scaled_train_X_shuffled, scaled_train_Y_shuffled, epochs=train_epochs2, batch_size=32, verbose = 2, shuffle = True, validation_split = 0.15, callbacks=[custom_early_stopping,reduce_lr])
    if train_epochs2 > 0:
      plt.plot(history2.history['loss'], label = 'Loss U')
      plt.plot(history2.history['val_loss'], label = 'Val Loss U')
      plt.legend()
      plt.title('Standard Training')
      plt.show()

    ## Generating test predictions for different intervals: U
    train_predictions = self.scaler_Y.inverse_transform(model.predict(scaled_train_X))
    train_labels = df.values[:, 0:1]


    preds_train = train_predictions.flatten()
    labels_train = train_labels.flatten()


    ## Obtaining pure tidal estimates by setting external forcing to zero 
    ## Note that these results are identical to the test_predictions given above if no multivariate forcing is present.
    ## This feature is still in beta and may not work as expected.
    if self.multi == True:
      no_multi = np.zeros(np.shape(dataset[:, 1:np.shape(self.ts)[1]]))
      no_multi_all = dataset[:, np.shape(self.ts)[1]:num_features-len(self.multivariate_lags) + 1]
      if len(self.multivariate_lags) == 1:
        test_X_TIDE = np.concatenate((no_multi, no_multi_all), axis=1)
      else:
        no_multi_lagged = np.zeros(np.shape(dataset[:, num_features - len(self.multivariate_lags) + 1:num_features]))
        test_X_TIDE = np.concatenate((no_multi, no_multi_all, no_multi_lagged), axis=1)

      scaled_test_X_TIDE = self.scaler_X.transform(test_X_TIDE.reshape(-1,1)).reshape(test_X_TIDE.shape)
      rtide_noforcing = self.scaler_Y.inverse_transform(model.predict(scaled_test_X_TIDE)).flatten()
    else:
      rtide_noforcing = preds_train.copy()

    ### Transforming tidal prediction into harmonics:
    try:
      short_utide = utide.solve(
        df[:].index,
        df[:].observations,
        lat=self.lat,
        method="ols",
        conf_int="none",
        verbose=False,
        trend = False,
        nodal = True
        )
      utide_train_pred = utide.reconstruct(df[:].index, short_utide, verbose = False)
      utide_train= utide_train_pred['h']

      rtide_utide = utide.solve(
        df[:].index,
        rtide_noforcing,
        lat=self.lat,
        method="ols",
        conf_int="none",
        verbose=False,
        trend = False,
        nodal = True,
        )

      irls_utide = utide.solve(
          df[:].index,
          df[:].observations,
          lat=self.lat,
          method="robust",
          conf_int="none",
          verbose=False,
          trend = False,
          nodal = True,
          )

      rtide_utide_reconstruction  =utide.reconstruct(df[:].index, rtide_utide, verbose = False)
      rtide_tide= rtide_utide_reconstruction['h']

      irls_utide_reconstruction  =utide.reconstruct(df[:].index, irls_utide, verbose = False)
      irls_tide= irls_utide_reconstruction['h']

      self.rtide_ha = rtide_utide ## standard OLS UTide
      self.utide_ha = short_utide ## standard OLS UTide
      self.utide_irls = irls_utide ## Iteratively reweighted least squares UTide


    except Exception as e:
      print(f"UTide error, this is not a problem with RTide {e}")
      utide_train = []
      utide_test = []
      RTide_Tide = []
      IRLS_Tide = []

    self.model = model
    if save_weights:
      model.save(f'{self.path}_model_weights.keras') ## SAVING MODEL
      joblib.dump(self.scaler_X, f'{self.path}_scaler_X.save') ## SAVING SCALARS
      joblib.dump(self.scaler_Y, f'{self.path}_scaler_Y.save')

    self.train_predictions = {'rtide_train': preds_train, 'train_observations': labels_train, 'utide_tide_train': utide_train, 'rtide_tide_train': rtide_tide, 'irls_utide_train': irls_tide, 'RTide_nomulti': rtide_noforcing}
    self.model_predictions = self.train_predictions
    self.train_prediction_df = pd.DataFrame({'observations': df.observations.to_numpy(), 'rtide':preds_train}, index = df.index)

  def Predict(self, df):
    """
      Function to generate predictions using the learned model at associated times.

      Inputs:
      --------
      df: dataframe
          Dataframe with datetime index at the desired timestamps and observations column.
          Observations should contain NaNs, or the actualy observations.
          If concurrent observations are included, RTide will automatically compute summary statistics.
          **Note if additional multivariate input functions were included, they must be passed as an additional column.

      Returns:
      --------
      self.test_predictions:  dictionary
        A dictionary of test predictions with the following keys:
            - rtide_test: np.array
                Array containing the predictions for the RTide model.

            - testobservations: np.array
                Array containing the actual observations (NaNs if not provided).

            - utide_tide_test np.array
                Array containing standard UTide OLS predictions.

            - rtide_tide_test: np.array
                Array containing RTide harmonic equivalent predictions.

            - irls_utide_test: np.array
                Array containing UTide IRLS predictions.
    """
    self.ts = df
    try:
      model = self.model
      scaler_X = self.scaler_X
      scaler_Y = self.scaler_Y
    except:
      try:
        # Define a dictionary with the custom layer
        custom_objects = {'AddLayer': AddLayer}

        # Load the model with the custom layer
        model = self.model = tf.keras.models.load_model(f'{self.path}_model_weights.keras', custom_objects=custom_objects)
        self.scaler_X = joblib.load(f'{self.path}_scaler_X.save')
        self.scaler_Y = joblib.load(f'{self.path}_scaler_Y.save')
      except:
        raise print("No model has been trained or has been saved.")

    ### Now need to compute inputs associated with new datapoints
    inputs = {
      'prediction': True
    }
    self.Prepare_Inputs(**inputs)

    df = self.prediction_dfs.dropna()

    dataset = df.values
    num_features = np.shape(dataset)[1]
    test_X = dataset[:, 1:num_features]
    scaled_test_X = self.scaler_X.transform(test_X.reshape(-1,1)).reshape(test_X.shape)

    rtide_test = self.scaler_Y.inverse_transform(model.predict(scaled_test_X))
    try:
      ols_utide_reconstruction = utide.reconstruct(df[:].index, self.utide_ha, verbose = False)
      ols_utide= ols_utide_reconstruction['h']

      rtide_utide_reconstruction  =utide.reconstruct(df[:].index, self.rtide_ha, verbose = False)
      rtide_tide= rtide_utide_reconstruction['h']

      irls_utide_reconstruction  =utide.reconstruct(df[:].index, self.utide_irls, verbose = False)
      irls_tide= irls_utide_reconstruction['h']
    except:
      ols_utide = []
      rtide_tide = []
      irls_tide = []
    nan_mask = np.isnan(df.values[:, 0:1])

    # Check if there are any NaNs in the array
    has_nans = np.any(nan_mask)

    # If there are NaNs, replace them with zeros
    if has_nans:
      test_observations = []
    else:
      test_observations = df.values[:, 0:1]
    self.predict_indexes = df.index
    self.test_predictions = {'rtide_test': rtide_test.flatten(), 'test_observations': test_observations.flatten(), 'utide_tide_test': ols_utide, 'rtide_tide_test': rtide_tide, 'irls_utide_test': irls_tide}
    self.test_prediction_df = pd.DataFrame({'observations': df.observations.to_numpy(), 'rtide':rtide_test.flatten()}, index = df.index)
  def Visualize_Residuals(self, savefig = False, returnfig = False, tides = False):
    """
        Function to visualize the residuals of the model.

        Inputs:
        -------
        savefig: False or str
            Option to save the figure. if not False, user must specify a file name.
        returnfig: bool
            Option to return the figure without plotting so the user can edit it.

        Returns:
        ---------
        Optionally returns figure object if return == True.

    """
    fig = plt.figure(figsize=(12, 6), dpi = 200)  # Adjust the width and height as needed
    try:
      train_predictions = self.train_predictions
      ### PLOTTING TRAINING PREDICTIONS
      plt.subplot(1, 2, 1)
      train_predictions = self.train_predictions
      indexs = self.prepped_dfs.dropna().index

      rtide_residuals = train_predictions['rtide_train'] - train_predictions['train_observations']
      plt.plot(indexs,rtide_residuals, label = f'RTide: $\mu$ =  {round(np.average(rtide_residuals),4)}\n$\sigma$ = {round(np.std(rtide_residuals),4)}', color = 'red')
      if tides:
        if len(train_predictions['utide_tide_train']) > 0:
          rtide_tide_residuals = train_predictions['rtide_tide_train'] - train_predictions['train_observations']
          plt.plot(indexs,rtide_tide_residuals, label = f'RTide Tide: $\mu$ =  {round(np.average(rtide_tide_residuals),4)}\n$\sigma$ = {round(np.std(rtide_tide_residuals),4)}', color = 'green')
          utide_residuals = train_predictions['utide_tide_train'] - train_predictions['train_observations']
          plt.plot(indexs, utide_residuals, label = f'UTide: $\mu$ =  {round(np.average(utide_residuals),4)}\n$\sigma$ = {round(np.std(utide_residuals),4)}', color = 'blue')

      plt.xlim(indexs[0], indexs[-1])
      plt.legend()
      plt.gcf().autofmt_xdate()
      plt.ylabel("Residual (units)")
      plt.title('Train Prediction Residuals')
    except:
      print("No training data available to plot. Retrain model if desired.")

    try:
      test_predictions = self.test_predictions
    except:
      test_predictions = None


    ### PLOTTING TEST PREDICTIONS
    if test_predictions != None:
      if len(test_predictions['test_observations'] > 0):
        plt.subplot(1, 2, 2)

        indexs = self.predict_indexes

        rtide_residuals = np.array(test_predictions['rtide_test']) - np.array(test_predictions['test_observations'])
        plt.plot(indexs,rtide_residuals, label = f'RTide: $\mu$ =  {round(np.average(rtide_residuals),4)}\n$\sigma$ = {round(np.std(rtide_residuals),4)}', color = 'red')
        if tides:
          if len(test_predictions['utide_tide_test']) > 0:
            rtide_tide_residuals = test_predictions['rtide_tide_test'] - np.array(test_predictions['test_observations'])
            plt.plot(indexs,rtide_tide_residuals, label = f'RTide Tide: $\mu$ =  {round(np.average(rtide_tide_residuals),4)}\n$\sigma$ = {round(np.std(rtide_tide_residuals),4)}', color = 'green')
            utide_residuals = test_predictions['utide_tide_test'] - np.array(test_predictions['test_observations'])
            plt.plot(indexs, utide_residuals, label = f'UTide: $\mu$ =  {round(np.average(utide_residuals),4)}\n$\sigma$ = {round(np.std(utide_residuals),4)}', color = 'blue')


        plt.xlim(indexs[0], indexs[-1])
        plt.legend()
        plt.ylabel("Residual (units)")
        plt.gcf().autofmt_xdate()

        plt.title('Test Prediction Residuals')

    if savefig:
      plt.savefig(f'{savefig}.png', dpi =300)

    if returnfig:
      return fig
    else:
      plt.show()
  
  def Visualize_Predictions(self, savefig = False, returnfig = False, verbose = True):
    """
    Helper function to visualize the train/test predictions of the model.

    Function will automatically plot whatever data is available. e.g. if you have loaded
    a model weights and only have test predictions, on the test predictions will display.

    Inputs:
    --------
    savefig: False or str
        Option to save the figure. if not False, user must specify a file name.
    returnfig: bool
        Option to return the figure without plotting so the user can edit it.
    verbose: bool
        Option to print out summary statistics of the model. (R^2, RMSE, MAE, MAPE)

    Returns:
    --------  
    Optionally returns figure object if return == True.
    """
    fig = plt.figure(figsize=(18, 9), dpi = 200)  # Adjust the width and height as needed
    try:
      train_predictions = self.train_predictions
    except:
      train_predictions = None
    if train_predictions != None:
      indexs = self.prepped_dfs.dropna().index
      ### Train Predictions
      plt.subplot(2, 1, 1)
      plt.plot(indexs, self.train_predictions['train_observations'], color = 'k', label = 'Actual')
      plt.plot(indexs, self.train_predictions['rtide_train'], color = 'red', label = 'RTide')
      plt.gcf().autofmt_xdate()
      if verbose:
        print("Train Results")
        _ = calc_stats(self.train_predictions['rtide_train'], self.train_predictions['train_observations'])
      plt.title("Train results")
      plt.ylabel('Sea Level (units)')
      plt.xlim(indexs[0], indexs[-1])
      plt.legend()

    ### Test Predictions
    try:
      test_predictions = self.test_predictions
    except:
      test_predictions = None
    if test_predictions != None:
      indexs = self.prediction_dfs.dropna().index
      plt.subplot(2, 1, 2)
      plt.plot(indexs, self.test_predictions['test_observations'], color = 'k', label = 'Actual')
      plt.plot(indexs, self.test_predictions['rtide_test'], color= 'red', label = 'RTide')

      if verbose:
        print("Test Results")
        _ = calc_stats(self.test_predictions['rtide_test'], self.test_predictions['test_observations'])
      plt.title("Test results")
      plt.ylabel('Sea Level (units)')
      plt.xlabel('Time')
      plt.xlim(indexs[0], indexs[-1])
      plt.gcf().autofmt_xdate()
    if savefig:
      plt.savefig(f'{savefig}.png', dpi =300)

    if returnfig:
      return fig
    else:
      plt.show()
    
  def Shap_Analysis(self, plot = True):
    """
    Function to compute and plot the SHAP values for the learned model.

    Inputs:
    --------
    plot: bool
        Option to plot the SHAP values.
    
    Returns:
    --------
    self.shap_explainer

    self.shap_values: dictionary
    """
    try:
      model = self.model
      scaler_X = self.scaler_X
      scaler_Y = self.scaler_Y
    except:
      try:
        # Define a dictionary with the custom layer
        custom_objects = {'AddLayer': AddLayer}

        # Load the model with the custom layer
        model = self.model = tf.keras.models.load_model(f'{self.path}_model_weights.keras', custom_objects=custom_objects)
        self.scaler_X = joblib.load(f'{self.path}_scaler_X.save')
        self.scaler_Y = joblib.load(f'{self.path}_scaler_Y.save')
      except:
        raise print("No model has been trained or has been saved.")
    df = self.prediction_dfs.dropna()
    dataset = df.values
    num_features = np.shape(dataset)[1]
    
    test_X = df.values[:, 1:num_features]
    scaled_test_X = self.scaler_X.transform(test_X.reshape(-1,1)).reshape(test_X.shape)
    
    def f(x):
      shape1 = np.shape(x)[0]
      shape2 = np.shape(x)[1]
      return self.scaler_Y.inverse_transform(self.model.predict(x))

    self.shap_explainer = shap.KernelExplainer(f, shap.sample(scaled_test_X,15))

    self.shap_values = self.shap_explainer.shap_values(scaled_test_X[:], nsamples=300)
    column_names = df.columns[1:]
    if plot:
      for ind in range(np.shape(self.ts)[1]-1):
        shap.dependence_plot(ind, self.shap_values[:,:,0], scaled_test_X[:], show= True, feature_names = column_names, interaction_index = None)
