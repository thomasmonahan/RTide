import numpy as np
import pandas as pd
import scipy.io as sio

from rtide import RTide
import time

import random

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

dir = './data/TE_Gauges/'


def load_gauge_data(gauge_name, data_dir=dir, resample_rate=None, interval=(1994,2014)):
    
    start_year = interval[0]
    end_year = interval[1]
    
    years = range(start_year, end_year + 1)

    dfs = []
    for i in years:
        file_path = f"{data_dir}{gauge_name}/{gauge_name}_{i}_p"
    
        mat = sio.loadmat(file_path)

        ts = mat["ts"].squeeze()
        wl = mat["wl"].squeeze()

        df_year = pd.DataFrame({
            "observations": wl,
        }, index= pd.to_datetime(ts - 719529, unit="D", utc = True))

        dfs.append(df_year)

    df_all = pd.concat([dfs[i] for i in range(len(dfs))], join='outer', axis=0)

    df_all = df_all.sort_index()
    df_all[df_all.columns] = df_all[df_all.columns].apply(pd.to_numeric, errors='coerce')
    
    # Resample if a resample rate is provided
    if resample_rate is not None:
        df_all = df_all.resample(f'{resample_rate}min').mean()
        print(f"Resampled data to every {resample_rate} minutes.")
        
    df_all = df_all.dropna()
    print('Nans dropped')
    print(f"Loaded data for {gauge_name} from {start_year} to {end_year}. Total samples: {len(df_all)}")
    return df_all

def train_gauge_model(df, train_interval, gauge_name, gauge_coords, include_outflow=False, lags=None, include_NEMO=False):
    
    train_df = df[(df.index.year >= train_interval[0]) & (df.index.year <= train_interval[1])]
    print('Training data loaded. Samples:', len(train_df))
    
    lat, lon = gauge_coords

    model = RTide(train_df, lat = lat, lon = lon)

    inputs = {
    'symmetrical': True,
    'sample_rate': 1,
    'path': f'{gauge_name}{train_interval[0]}_{train_interval[1]}{"_outflow" if include_outflow else ""}{"_NEMO" if include_NEMO else ""}{f"_{lags}lags" if lags else ""}',
    'fixed_location':True,
    'multivariate_lags': lags 
    }

    model.Prepare_Inputs(**inputs)
    
    train_inputs = {
    'loss': 'MSE',
    'lr': 1e-04,
    'early_stoppage': 30,
    'save_weights': True,
    'regularization_strength': 0.001, 
    'standard_epochs': 500,
    'featurewise_scaling':True, 
    }

    print(f"Training {gauge_name} {'WITH' if include_outflow else 'WITHOUT'} outflow")
    print(f"{'WITH' if include_NEMO else 'WITHOUT'} NEMO")
    print(f"Training period: {train_interval[0]}-{train_interval[1]}")
    model.Train(**train_inputs)
    
    return model  

def get_preds(model, df):
    
    model.Predict(df)
    
    predictions_df = pd.DataFrame({
        'timestamp': model.predict_indexes,
        'observations': model.test_predictions['test_observations'],
        'predictions': model.test_predictions['rtide_test']
    }).set_index('timestamp')
    
    return predictions_df

def year_metrics(predictions_df, test_interval):
    results = []
    
    for year in range(test_interval[0], test_interval[1] + 1):
        start = f'{year}-01-01'
        end = f'{year}-12-31'
        
        mask = (predictions_df.index >= start) & (predictions_df.index <= end)
        window = predictions_df[mask]
        
        if len(window) > 0:
            obs = window['observations']
            pred = window['predictions']
            
            metrics = {
                'year': year,
                'r2': r2_score(obs, pred),
                'rmse': np.sqrt(mean_squared_error(obs, pred)),
                'mae': mean_absolute_error(obs, pred),
                'mape': mean_absolute_percentage_error(obs, pred),
                'brier_score_99p5': np.mean(((pred >= np.percentile(obs, 99.5)).astype(int) - (obs >= np.percentile(obs, 99.5)).astype(int))**2),
                'n_samples': len(window)
            }
            results.append(metrics)
    
    df_results = pd.DataFrame(results)
    return df_results