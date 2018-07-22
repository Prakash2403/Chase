from config import TRAIN_NETWORK, VISUALIZE, STOCKS
from datahandler.stock import StockDataHandler
from network.LSTMNetwork import LSTMNetwork
from utils import get_datasets


def get_predictions():
    datasets = get_datasets()
    ret_data = []
    if datasets is not None:
        for i, data in enumerate(datasets):
            data_handler = StockDataHandler()
            data_handler.add(data, STOCKS[i])
            lstm_network = LSTMNetwork(data_handler)
            predictions = lstm_network.run_model(
                weight_filename='stock_{0}_weights.h5'.format(STOCKS[i].split('/')[-1]), train=TRAIN_NETWORK,
                evaluate=False, visualize=VISUALIZE)
            ret_data.append(predictions)
    return ret_data


if __name__ == "__main__":
    print(get_predictions())
