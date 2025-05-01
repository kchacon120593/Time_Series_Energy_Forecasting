from src.install_import import install_if_missing

# Check and install required packages
install_if_missing("pandas")
install_if_missing("numpy")

import pandas as pd
import numpy as np

def load_data(path, timestamp_Col, demand_Col):
    """
    Loads data from csv file, separated by comma into a Pandas DataFrame

    Args:
        path(string): Filepath
        timestamp_Col(string): The name of the Timestamp column in the File
        demand_Col(string): The name of the Demand column in the File

    Returns:
        df(pd.DataFrame): A Pandas DataFrame with a timestamp and a numeric column representing demand in MW  
    """
    
    df = pd.read_csv(path
                     , sep=','
                     , infer_datetime_format=True
                     , low_memory=False)
    
    df = df[[timestamp_Col, demand_Col]]
    
    return df



def preprocess_data(df, timestamp_Col):
    """
    Formats dataframe index using timestamp column

    Args:
        df(pd.DataFrame): Pandas Dataframe with Timenstamp and Demand columns
        timestamp_Col(string): The name of the Timestamp column in the DataFrame

    Returns:
        df(pd.DataFrame): A Pandas DataFrame with a DateTime Index and a single column representing National Demand in MW
    """
    df.reset_index(drop=True, inplace=True)
    df = df.set_index(timestamp_Col)
    df.index = pd.to_datetime(df.index)
    
    return df



def create_time_features(df):
        
    """
    Create features based on time series index
    
    Args:
        df(pd.DataFrame): Pandas Dataframe with a DateTime Index and a Demand column

    Returns:
        df(pd.DataFrame): A Pandas DataFrame with a DateTime Index, time features (hour, date_of_week, month, quarter
                          , year, day_of_year, and hour_fix formatted as 00:00 for visualization purposes)
                          plus a column representing National Demand in MW
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('DataFrame index must be a DateTimeIndex')

    if df.shape[1] != 1:
        raise ValueError('DataFrame must have exactly one column representing demand')
    
    df = df.copy()
    df['hour'] = df.index.time
    df['day_of_week'] = df.index.day_of_week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    
    #Converting object column to datetime
    df['hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S')
    
    #Creating additional column for visualizations
    df['hour_fixed'] = pd.to_datetime(df['hour'], format='%H:%M:%S').dt.strftime('%H:%M')
    
    return df

def create_lag_features(df, demand_Col='nd'):
    """
    Adds lag and rolling features to a DataFrame with 30-minute interval data.

    Parameters:
    df (pd.DataFrame): Pandas Dataframe with a DateTime Index and a Demand column
    demand_Col(string): The name of the Demand column in the DataFrame

    Returns:
    df(pd.DataFrame): DataFrame with new lag and rolling features added.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame index must be a DateTimeIndex')

    df = df.copy()

    # Define lags (based on 30-minute intervals)
    lag_features = {
        'lag_1': 1,           # 30 minutes ago
        'lag_2': 2,           # 1 hour ago
        'lag_4': 4,           # 2 hours ago
        'lag_48': 48,         # 1 day ago
        'lag_336': 336,       # 1 week ago
        'lag_17520': 17520    # 1 year ago
    }

    # Add lag features
    for name, lag in lag_features.items():
        df[name] = df[demand_Col].shift(lag)

    # Add rolling statistics (with 1-day and 1-week windows)
    df['rolling_mean_48'] = df[demand_Col].shift(1).rolling(window=48).mean()
    df['rolling_std_48'] = df[demand_Col].shift(1).rolling(window=48).std()

    df['rolling_mean_336'] = df[demand_Col].shift(1).rolling(window=336).mean()
    df['rolling_std_336'] = df[demand_Col].shift(1).rolling(window=336).std()

    # Drop rows with missing values (due to shifting/rolling)
    df.dropna(inplace=True)

    return df

    