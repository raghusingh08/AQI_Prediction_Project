import pandas as pd
import numpy as np

def parse_and_sort(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

def add_time_features(df, date_col='date'):
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df['dayofweek'] = dt.dt.dayofweek
    df['month'] = dt.dt.month
    df['dayofyear'] = dt.dt.dayofyear
    return df

def make_supervised(df, target='pm25', max_lag=14, roll_windows=(7,14)):
    df = df.copy()
    for l in range(1, max_lag+1):
        df[f'{target}_lag{l}'] = df[target].shift(l)
    for w in roll_windows:
        df[f'{target}_rollmean_{w}'] = df[target].rolling(w).mean()
    return df

def train_valid_split_time(df, test_size=0.2):
    n = len(df)
    split = int(n * (1 - test_size))
    train = df.iloc[:split].copy()
    valid = df.iloc[split:].copy()
    return train, valid
