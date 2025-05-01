from src.install_import import install_if_missing
# Check and install required packages
install_if_missing("pandas")
install_if_missing("numpy")
install_if_missing("matplotlib")
install_if_missing("seaborn")
install_if_missing("statsmodels")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

color_pal = sns.color_palette()


def demand_over_time(df):
    """
    Plots National Demand over time with a linear trend line.
    
    Args:
        df(pd.DataFrame): A DataFrame with a DateTimeIndex and a single column representing demand in MW

    Returns:
        None: This function displays a plot and does not return any value.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame index must be a DateTimeIndex')

    if df.shape[1] != 1:
        raise ValueError('DataFrame must have exactly one column representing demand')
    
    df.plot(style='.'
            , figsize=(15,5)
            , color=color_pal[4]
            , title='National demand in MW'
            , label='National Demand' )


    # Adding trend line
    x = (df.index - df.index[0]).days
    y = df.iloc[:, 0].values 

    coefficients = np.polyfit(x, y, 1)
    trend = np.poly1d(coefficients)

    # Plot the trend line
    plt.plot(df.index, trend(x), color='black', linewidth=2, label='Trend Line')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend(['National Demand','Trend Line'])

    plt.show()



def fixedPeriod_Data(df, start_date, end_date, period):
    """
    Plots National Demand over specific periods of time.
    
    Args:
        df(pd.DataFrame): A DataFrame with a DateTimeIndex and a single column representing demand in MW
        start_date(string "dd-mm-yyyy"): A String representing the start date of the plot
        end_date(string "dd-mm-yyyy"): A String representing the end date of the plot
        period(string): A String indicating if plot shows "Daily", "Monthly", "Quarterly" or "Yearly" data

    Returns:
        None: This function displays a plot and does not return any value.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame index must be a DateTimeIndex')

    if df.shape[1] != 1:
        raise ValueError('DataFrame must have exactly one column representing demand')
    
    df.loc[(df.index > start_date) & (df.index < end_date)]\
        .plot(figsize=(15,5)
              , color=color_pal[4]
              , title=f'{period} National Demand in MW')
      
    plt.legend(['National Demand'])
    plt.xlabel('Period')
    plt.ylabel('Demain (MW)')
    plt.show()
    
    
    
def demand_by_Hour(df):
    """
    Plots Hourly National Demand using bloxplots
    
    Args:
        df(pd.DataFrame): A DataFrame with time features included

    Returns:
        None: This function displays a plot and does not return any value.
    """   
    
    if 'hour_fixed' not in df.columns:
        raise ValueError("Missing required column: 'hour_fixed'")
    
    fig,ax = plt.subplots(figsize=(30,12))
    sns.boxplot(data=df, x='hour_fixed', y='nd')
    ax.set_title('Demand by Hour')
    plt.legend(['National Demand'])
    plt.xlabel('Period')
    plt.ylabel('Demain (MW)')
    
def decompose_seasonality(df, period):


    """
    Decomposes the time series into trend, seasonal, and residual components.
        
    Args:
        df(pd.DataFrame): A DataFrame with a DateTimeIndex and a single column representing demand in MW
    
    Returns:
        None: This function displays a plot and does not return any value.
        """
        
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame index must be a DateTimeIndex')
    
    if df.shape[1] != 1:
        raise ValueError('DataFrame must have exactly one column representing demand')
        
    result = seasonal_decompose(df, model='multiplicative', period=period)  # Adjust period as needed
    result.plot()
    plt.show()