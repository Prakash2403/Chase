from pandas import read_csv
import numpy as np
import quandl

from config import *
from utils.exceptions import UnknownModeException


def load_data(filename):
    return read_csv(filename)


def denormalize(data, min_val, max_val):
    return np.asarray(data) * (max_val - min_val) + min_val


def destandardize(data, mean_val, std_val):
    return np.asarray(data) * std_val + mean_val


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def standardize(data):
    return (data - data.mean()) / data.std()


def window_transform_series(series, feature):
    """
    First column of series should be the feature which
    we want to forecast.
    """
    x = []
    y = []
    for window in range(len(series) - WINDOW_SIZE):
        x.append(series[window:window + WINDOW_SIZE].values)
        y.append(series.iloc[[window + WINDOW_SIZE]][feature].values)
    x = np.asarray(x)
    y = np.asarray(y)
    y = np.reshape(y, (y.shape[0], OUTPUT_DIMENSION))
    assert len(y.shape) == 2
    return x, y


# def add_minmax_bound(dataframe, feature, grp_std_dev):
#     """
#     Adds min max bounds of feature to the dataframe.
#     :param grp_std_dev: Standard deviation of training data, grouped by hour and minute
#     :param dataframe: Dataframe containing values
#     :param feature: Feature of which min_max bound has to be calculuated.
#     :return: Pandas Dataframe containing following columns.
#     [feature, 'min_bound', 'max_bound']
#     index is TImestamp in "YYYY-MM-DD HH:MM:SS" format.
#     """
#     dataframe.columns = [feature, ]
#     dataframe = dataframe.dropna(axis='index')
#     dataframe['min_bound'] = np.zeros(dataframe.shape[0])
#     dataframe['max_bound'] = np.zeros(dataframe.shape[0])
#     for i in range(dataframe.shape[0]):
#         dataframe['min_bound'][i] = dataframe[feature][i] - \
#                                          K * grp_std_dev[feature][dataframe.index[i].hour][dataframe.index[i].minute]
#         dataframe['max_bound'][i] = dataframe[feature][i] + \
#                                          K * grp_std_dev[feature][dataframe.index[i].hour][dataframe.index[i].minute]
#     return dataframe


def get_data_from_quandl(stocks, save=True):
    """
    Returned dataframe is indexed by Date. So, no need to externally parse dates and change index column.
    :param stocks: Stocks to be downloaded
    :param save: If True, then save the downloaded stock.
    :return: A list of pandas dataframe containing the details of requested stocks.
    """
    quandl.ApiConfig.api_key = QUANDL_KEY
    df_list = []
    for stock in stocks:
        print("DOWNLOADING {0} DATA".format(stock))
        df = quandl.get(stock, start_date=RETRIEVAL_START_DATE, end_date=RETRIEVAL_END_DATE)
        df = df[REL_DATA_COLUMNS]
        df_list.append(df)
        if save:
            df.to_csv('{0}/{1}.csv'.format(DATA_DIR, stock.split('/')[-1]))
    return df_list


def get_datasets():
    if str(MODE).upper() == 'LOCAL':
        rel_cols = REL_DATA_COLUMNS.copy()
        rel_cols.append(INDEX_COLUMN)
        return [read_csv(DATA_DIR + stock.split('/')[-1] + '.csv', usecols=rel_cols, index_col=INDEX_COLUMN,
                         infer_datetime_format=True, parse_dates=True) for stock in STOCKS]
    elif str(MODE).upper() == 'QUANDL':
        return get_data_from_quandl(STOCKS, SAVE)
    else:
        raise UnknownModeException('Mode should be either "local" or "quandl"')
