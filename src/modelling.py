from src.install_import import install_if_missing

# Check and install required packages
install_if_missing("pandas")
install_if_missing("numpy")
install_if_missing("matplotlib")
install_if_missing("seaborn")
install_if_missing("prophet")
install_if_missing("scikit-learn")
install_if_missing("plotly")
install_if_missing("prophet")
install_if_missing("nbformat")
install_if_missing("xgboost")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

from src.performance import track_resources




def train_test_split(df, date, demand_col, exclude_cols=None):
    """
    Splits the dataset into training and testing sets based on a DateTime Index
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    date (str): The name of the date column in the DataFrame
    demand_col (str): The name of the demand column in the DataFrame.
    exclude_cols (list, optional): A list of columns to exclude from the training and testing sets. Defaults to None.
        
    Returns:
    train (pd.DataFrame): The training set.
    test (pd.DataFrame): The testing set.
    X_train (pd.DataFrame): The training set features.
    y_train (pd.Series): The training set target variable.
    X_test (pd.DataFrame): The testing set features.
    y_test (pd.Series): The testing set target variable.
    FEATURES (list): A list of feature names used in the model.
    TARGET (str): The name of the target variable.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex): 
        raise ValueError('DataFrame index must be a DateTimeIndex')
    
    # Split the data into training and testing sets
    train = df.loc[df.index < date]
    test = df.loc[df.index >= date]
    
    #X,y for train and test datasets
    FEATURES = df.columns.tolist()
    if exclude_cols is not None:
        FEATURES = [col for col in FEATURES if col not in exclude_cols]
    FEATURES.remove(demand_col)
    
    TARGET = demand_col

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]
        

    # Check if the training and testing sets are empty
    if train.empty or test.empty:
        raise ValueError('Training or testing set is empty. Please check the date provided for splitting.')    
    
    return train, test, X_train, y_train, X_test, y_test, FEATURES, TARGET



def train_test_split_plot(df, date, demand_col):
    """
    Splits the dataset into training and testing sets based on a date column and plots the results.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    date (str): The name of the date column in the DataFrame
    demand_col (str): The name of the demand column in the DataFrame.
        
    Returns:
    None: This function displays a plot and does not return any value.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame index must be a DateTimeIndex')
    
    #Format the date column to datetime
    if not isinstance(date, pd.Timestamp):
        if not isinstance(date, str):
            raise ValueError('Date must be a string in the format "YYYY-MM-DD"')
        
        # Convert the date string to a datetime object
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError('Invalid date format. Please use "YYYY-MM-DD"')    
    else:
        date = pd.to_datetime(date)
    
    # Split the data into training and testing sets
    train = df.loc[df.index < date]
    test = df.loc[df.index >= date]

    fig, ax = plt.subplots(figsize = (20,8))

    train[[demand_col]].plot(ax=ax, label='Training Set', title='Data Train/Test Split')
    test[[demand_col]].plot(ax=ax, label='Test Set')
    
    ax.axvline(date, color = 'gray', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    plt.xlabel('Period')
    plt.ylabel('Demain (MW)')
    plt.show()
    
@track_resources
def prophet_model(X_train
                  , y_train
                  , X_test
                  , y_test
                  , FEATURES=None
                  , holidays=None
                  , yearly_seasonality=False
                  , weekly_seasonality=False
                  , daily_seasonality=False):
    """
    Fits a Prophet model to the data and returns the fitted model.
    
    Parameters:
    X_train (pd.DataFrame): The training set features.
    y_train (pd.Series): The training set target variable.
    X_test (pd.DataFrame): The testing set features.
    y_test (pd.Series): The testing set target variable.
    FEATURES (list, optional): A list of feature names to include in the model. Defaults to None.
    yearly_seasonality (bool, optional): Whether to include yearly seasonality. Defaults to False.
    weekly_seasonality (bool, optional): Whether to include weekly seasonality. Defaults to False.
    daily_seasonality (bool, optional): Whether to include daily seasonality. Defaults to False.
    holidays (pd.DataFrame, optional): A DataFrame containing holiday information. Defaults to None.
        
    Returns:
    model (Prophet): The fitted Prophet model.
    forecast (pd.DataFrame): The forecasted values for the test set.
    """
    
    
    # Prepare the data for Prophet
    train = X_train.copy()
    train['ds'] = X_train.index
    train['y'] = y_train.values


    test = X_test.copy()
    test['ds'] = X_test.index
    test['y'] = y_test.values
    
    #Initialize model

    model = Prophet(holidays=holidays
                    , yearly_seasonality=yearly_seasonality
                    , weekly_seasonality=weekly_seasonality
                    , daily_seasonality=daily_seasonality
                    )

    #Add regressors
    if FEATURES:
        for feature in FEATURES:
            model.add_regressor(feature)


    # Create and fit the Prophet model
    model.fit(train[['ds','y'] + FEATURES])
    
    
    # Create future dataframe that includes test dates
    future = test[['ds'] + FEATURES].copy()

    # Forecast
    forecast = model.predict(future)

    # Return both the model and forecasted test points
    return model, forecast


def prophet_model_evaluation(model, forecast, y_test, plot=True):
    """
    Evaluates the performance of the Prophet model using Mean Absolute Error (MAE).
    
    Parameters:
    model (Prophet): The trained Prophet model.
    forecast (pd.DataFrame): The forecasted values.
    y_test (pd.Series): The actual values for the test set.
    plot (bool, optional): Whether to plot the forecasted and actual values. Defaults to True.
        
    Returns:
    mae (float): The Mean Absolute Error of the model.
    """
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, forecast['yhat'])
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, forecast['yhat']))
    
    # Print the evaluation metrics
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    
    # Plot actual vs predicted if plot=True
    if plot:
        fig = plot_plotly(model, forecast)  # Use model instead of forecast
        fig.show()
    
    # Return a dictionary with the evaluation metrics
    return {'MAE': mae, 'RMSE': rmse}

    
@track_resources
def xgboost_model(X_train
                  , y_train
                  , X_test
                  , y_test
                  , n_estimators=100
                  , learning_rate=0.1
                  , max_depth=6
                  , subsample=0.8
                  , colsample_bytree=0.8
                  , random_state=42):
    """
    Fits an XGBoost model to the training data and evaluates it on the test data.
    
    Parameters:
    X_train (pd.DataFrame): The training set features.
    y_train (pd.Series): The training set target variable.
    X_test (pd.DataFrame): The test set features.
    y_test (pd.Series): The test set target variable.
    n_estimators (int, optional): The number of trees in the ensemble. Defaults to 100.
    learning_rate (float, optional): The learning rate for the model. Defaults to 0.1.
    max_depth (int, optional): The maximum depth of the trees. Defaults to 6.
    subsample (float, optional): The fraction of samples to be used for each tree. Defaults to 0.8.
    colsample_bytree (float, optional): The fraction of features to be used for each tree. Defaults to 0.8.
    ramdom_state (int, optional): The random seed for reproducibility. Defaults to 42.
        
    Returns:
    model (XGBRegressor): The fitted XGBoost model.
    """
    
    # Initialize the XGBoost model
    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             random_state=random_state)

    # Fit the model to the training data
    model.fit(X_train, y_train
              , eval_set=[(X_train, y_train)
              , (X_test, y_test)]
              , verbose=100)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    
    return model, y_pred


def xgboost_model_evaluation(y_test, y_pred):
    """
    Evaluates the performance of the XGBoost model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
        
    Parameters:
    y_test (pd.Series): The actual values for the test set.
    y_pred (pd.Series): The predicted values for the test set.
            
    Returns:
    None: This function prints the evaluation metrics.
    """
        
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
        
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
    # Print the evaluation metrics
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # Return a dictionary with the evaluation metrics
    return {'MAE': mae, 'RMSE': rmse}

    
