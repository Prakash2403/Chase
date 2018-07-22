import pandas as pd

SMA_ROLLING_WINDOW = 5


def get_sma(data: pd.DataFrame, feature):
    return data[feature].rolling(SMA_ROLLING_WINDOW).mean()


def get_daily_returns(data, feature):
    return data[feature]/data[feature].shift() - 1
