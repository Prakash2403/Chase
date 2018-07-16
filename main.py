import pandas

from network import *
import pandas as pd


def add_datetime(data, timestamp_col):
    """
    data: Data on which you want to work.
    timestamp_col: Column name which contains the unix timestamp
    """
    new_data = data.copy()
    new_data.index = pd.to_datetime(data[timestamp_col], unit='s')
    new_data = new_data.sort_index()
    new_data['hour'] = new_data.index.hour
    new_data['minute'] = new_data.index.minute
    return new_data


def get_predictions():
    """

    :return: dataframe containing predictions and min-max bounds.
    """
    data = pd.read_csv('data/emsdata0.csv')
    # filtered_data = health_check(data)
    data = add_datetime(data, 't')
    ret_data = None
    if data is not None:
        rel_features = ['bc', 'hour', 'minute']
        data_handler = DataHandler(data, rel_features)
        lstm_network = LSTMNetwork(data_handler)
        predictions = lstm_network.run_model(start_datetime='2018-05-05 21:09:05', end_datetime='2018-05-23 21:09:05'
                                             , weight_filename='bc_aaa_w.h5', train=True)
        ret_data = predictions
    return ret_data


def detect_anomalies():
    pass


if __name__ == "__main__":
    pandas.DataFrame(get_predictions()).plot()
    import matplotlib.pyplot as plt
    plt.show()
