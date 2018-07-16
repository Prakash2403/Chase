import pandas as pd
import numpy as np
from config import *


def load_data(filename):
    return pd.read_csv(filename)


def denormalize(data, min_val, max_val):
    return np.asarray(data) * (max_val - min_val) + min_val


def destandardize(data, mean_val, std_val):
    return np.asarray(data) * std_val + mean_val


def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def standardize(data, mean_val, std_val):
    return (data - mean_val) / std_val


def window_transform_series(series, feature):
    """
    First column of series should be the feature which 
    we want to forecast.
    """
    X = []
    y = []
    for window in range(len(series) - WINDOW_SIZE):
        X.append(series[window:window + WINDOW_SIZE].values)
        y.append(series.iloc[[window + WINDOW_SIZE]][feature].values)
    return X, y


def health_check(data):
    """
    Returns the dataframe if data seems to healthy, None otherwise.

    return_type: pandas.DataFrame
    """
    return pd.DataFrame()


def get_data(**kwargs):
    """
    Returns data from a sensor or from a file
    
    return_type: pandas.DataFrame
    """
    return pd.DataFrame()


def add_minmax_bound(dataframe, feature, grp_std_dev):
    """
    Adds min max bounds of feature to the dataframe.
    :param grp_std_dev: Standard deviation of training data, grouped by hour and minute
    :param dataframe: Dataframe containing values
    :param feature: Feature of which min_max bound has to be calculuated.
    :return: Pandas Dataframe containing following columns.
    [feature, 'min_bound', 'max_bound']
    index is TImestamp in "YYYY-MM-DD HH:MM:SS" format.
    """
    dataframe.columns = [feature, ]
    dataframe = dataframe.dropna(axis='index')
    dataframe['min_bound'] = np.zeros(dataframe.shape[0])
    dataframe['max_bound'] = np.zeros(dataframe.shape[0])
    for i in range(dataframe.shape[0]):
        dataframe['min_bound'][i] = dataframe[feature][i] - \
                                         K * grp_std_dev[feature][dataframe.index[i].hour][dataframe.index[i].minute]
        dataframe['max_bound'][i] = dataframe[feature][i] + \
                                         K * grp_std_dev[feature][dataframe.index[i].hour][dataframe.index[i].minute]
    return dataframe
