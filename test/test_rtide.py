import pytest
import pandas as pd
import numpy as np
import rtide
from rtide import RTide

@pytest.fixture
def sample_data():
    constit_freqs = [8.05114007e-02, 7.89992487e-02, 8.33333333e-02, 8.20235526e-02]
    amplitudes = [2.69536915e+00, 5.72051193e-01, 4.07506583e-01, 1.77388101e-01]
    phases = [100.17759146, 67.89243773, 139.00250991, 137.86574464]
    times = pd.date_range(start='2024-01-01 00:00:00+00:00', periods=24*7, freq="1H", tz='UTC')
    time_vals = times.to_julian_date().to_numpy()
    individual_tides = []
    for i, j in enumerate(constit_freqs):
        individual_tides.append(amplitudes[i]*np.cos(24*j*np.array(time_vals) * 2 * np.pi + (phases[i] * np.pi / 180)))
    tide = np.sum(individual_tides, axis=0) + 0.1*np.random.normal(len(individual_tides))
    df = pd.DataFrame({'observations': tide}, index=times)
    return df

@pytest.fixture
def sample_data_multi():
    constit_freqs = [8.05114007e-02, 7.89992487e-02, 8.33333333e-02, 8.20235526e-02]
    amplitudes = [2.69536915e+00, 5.72051193e-01, 4.07506583e-01, 1.77388101e-01]
    phases = [100.17759146, 67.89243773, 139.00250991, 137.86574464]
    
    times = pd.date_range(start='2024-01-01 00:00:00+00:00', periods=24*7, freq="1H", tz='UTC')
    time_vals = times.to_julian_date().to_numpy()
    multivar = time_vals * 2 * np.pi
    individual_tides = []
    for i, j in enumerate(constit_freqs):
        individual_tides.append(amplitudes[i]*np.cos(24*j*np.array(time_vals) * 2 * np.pi + (phases[i] * np.pi / 180)))
    tide = np.sum(individual_tides, axis=0) + 0.1*np.random.normal(len(individual_tides))
    df = pd.DataFrame({'observations': tide, 'exog': multivar}, index=times)
    return df

def test_rtide_init(sample_data):
    """
    ensure that the RTide class is initialized correctly for a single time series
    """
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data, lat, lon)
    assert model.ts.equals(sample_data)
    assert model.lat == lat
    assert model.lon == lon
    assert model.M_E == 5.9722*10**24
    assert model.M_M == 7.3*10**22
    assert model.M_S == 1.989*10**30
    assert model.E_r == 6371.01*10**3
    assert model.solar_constant == 1.946/100
    assert model.multi == False

def test_rtide_init_multi(sample_data_multi):
    """
    ensure that the RTide class is initialized correctly for a multi-input time series
    """
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data_multi, lat, lon)
    assert model.ts.equals(sample_data_multi)
    assert model.lat == lat
    assert model.lon == lon
    assert model.M_E == 5.9722*10**24
    assert model.M_M == 7.3*10**22
    assert model.M_S == 1.989*10**30
    assert model.E_r == 6371.01*10**3
    assert model.solar_constant == 1.946/100
    assert model.multi == True

def test_rtide_init_ts_index(sample_data):
    """
    Make sure that model only accepts pandas time-indexes
    """
    with pytest.raises(ValueError):
        bad_sample_data = sample_data.copy()
        bad_sample_data.index = range(len(sample_data))
        model = RTide(bad_sample_data, 44.9062, -66.996201)
    lat = '44.9062'
    lon = -66.996201
    with pytest.raises(ValueError):
        model = RTide(sample_data, lat, lon)

def test_prep(sample_data):
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data, lat, lon)
    model.sample_rate = 1 ## function isn't supposed to be called in isolation so setting manually
    uniform = [1,12]
    symmetrical = True
    multivariate_lags = None
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert model.nonunilags == [-12,0,12]

    uniform = [1,12]
    symmetrical = False
    multivariate_lags = None
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert model.nonunilags == [-12,0]

    uniform = [1.5,12]
    symmetrical = True
    multivariate_lags = None
    with pytest.raises(AssertionError):
        model.Prep(uniform, symmetrical, multivariate_lags)
    
    uniform = False
    symmetrical = False
    multivariate_lags = None
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert all(i <= 0 for i in model.nonunilags)

def test_prep_multi(sample_data_multi):
    """
    Ensure the Prep function works correctly for multivariate time series.
    """
    print('SHAPE OF MULTI:', np.shape(sample_data_multi))
    lat = 44.9062
    lon = -66.996201

    uniform = False
    symmetrical = True
    multivariate_lags = None
    model = RTide(sample_data_multi, lat, lon)
    model.sample_rate = 1 ## function isn't supposed to be called in isolation so setting manually
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert model.multivariate_lags == [0]

    uniform = False
    symmetrical = False
    multivariate_lags = 'negative'
    model = RTide(sample_data_multi, lat, lon)
    model.sample_rate = 1 ## function isn't supposed to be called in isolation so setting manually
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert all(i <= 0 for i in model.nonunilags)
    assert all(i <= 0 for i in model.multivariate_lags)

    uniform = False
    symmetrical = True
    multivariate_lags = [-21.3, 34.5]
    model = RTide(sample_data_multi, lat, lon)
    model.sample_rate = 1 ## function isn't supposed to be called in isolation so setting manually
    model.Prep(uniform, symmetrical, multivariate_lags)
    assert model.multivariate_lags == [-21.3, 34.5]

    uniform = False
    symmetrical = True
    multivariate_lags = 'standard'
    model = RTide(sample_data_multi, lat, lon)
    model.sample_rate = 1 ## function isn't supposed to be called in isolation so setting manually
    model.Prep(uniform, symmetrical, multivariate_lags)
    def remove_zero(lst):
        try:
            lst.remove(0)
        except ValueError:
            raise('something went wrong in test function, should be a zero in nonunilags')
        return lst
    news = remove_zero(model.nonunilags)
    assert model.multivariate_lags == news


def test_prepare_inputs(sample_data):
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs(symmetrical = True)
    assert np.shape(model.prepped_dfs)[1] == 153
    assert model.multi == False

def test_prepare_inputs_multi(sample_data_multi):
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data_multi, lat, lon)
    model.Prepare_Inputs(symmetrical = True)
    assert model.multivariate_lags == [0]
    assert np.shape(model.prepped_dfs)[1] == 154
    with pytest.raises(ValueError):
        model.Prepare_Inputs(multivariate_lags = [-1000])

def test_train(sample_data):
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs()
    model.Train()
    assert model.model is not None
    assert isinstance(model.train_prediction_df, pd.DataFrame)
    assert isinstance(model.train_predictions, dict)

def test_predict(sample_data):
    """
    Test the predict function
    """
    lat = 44.9062
    lon = -66.996201
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs()
    ### Condition where weights are loaded. 
    model.Predict(sample_data[:5])
    assert model.model is not None
    assert isinstance(model.test_prediction_df, pd.DataFrame)
    assert isinstance(model.test_predictions, dict)

def test_visualize_predictions(sample_data):
    """
    Test the visualize_predictions function
    """
    lat = 44.9062
    lon = -66.996201
    ## standard
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs()
    model.Train(standard_epochs = 5)
    model.Predict(sample_data[:5])
    model.Visualize_Predictions()

def test_visualize_predictions(sample_data):
    """
    Test the visualize_predictions function
    """
    lat = 44.9062
    lon = -66.996201
    ## standard
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs()
    model.Train(standard_epochs = 5)
    model.Predict(sample_data[:5])
    model.Visualize_Predictions(verbose = True)

def test_visualize_residuals(sample_data):
    """
    Test the visualize_predictions function
    """
    lat = 44.9062
    lon = -66.996201
    ## standard
    model = RTide(sample_data, lat, lon)
    model.Prepare_Inputs()
    model.Train(standard_epochs = 5)
    model.Predict(sample_data[:5])
    model.Visualize_Residuals()