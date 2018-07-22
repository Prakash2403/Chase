import pandas as pd

SMA_ROLLING_WINDOW = 5
EMA_ALPHA = 0.5


def sma(data: pd.DataFrame, feature):
    return data[feature].rolling(SMA_ROLLING_WINDOW).mean()


def daily_returns(data, feature):
    return data[feature]/data[feature].shift() - 1


def min_bollinger_band(data: pd.DataFrame, feature):
    return data[feature].rolling(SMA_ROLLING_WINDOW).mean() - 2 * data[feature].std()


def max_bollinger_band(data: pd.DataFrame, feature):
    return data[feature].rolling(SMA_ROLLING_WINDOW).mean() + 2 * data[feature].std()


def ema(data: pd.DataFrame, feature):
    return data[feature].ewm(alpha=EMA_ALPHA).mean()
