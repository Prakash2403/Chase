from network import *


def get_predictions():
    """

    :return: dataframe containing predictions and min-max bounds.
    """
    data = get_datasets()
    ret_data = None
    if data is not None:
        data_handler = FinanceDataHandler(data)
        lstm_network = LSTMNetwork(data_handler)
        predictions = lstm_network.run_model(start_datetime='2018-06-07 21:09:05', end_datetime='2018-07-20 21:09:05'
                                             , weight_filename='bc_aaa_w.h5', train=True, evaluate=False, visualize=True)
        ret_data = predictions
    return ret_data


if __name__ == "__main__":
    print(get_predictions())
