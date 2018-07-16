# TODO: SET A LOGGER

import abc

from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras import backend as K_b
from pandas._libs.tslib import Timestamp #TODO: Change it to datetime

from errors import DateError
from utils import *
from warnings import *


class DataHandler:

    def __init__(self, data, features):
        self.data = data
        self.rel_features = features
        self.set_relevant_data(features)
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None

    def set_relevant_data(self, features):
        self.data = self.data[features]

    def normalize_data(self):
        self.min_val = self.data[self.rel_features[0]].min()
        self.max_val = self.data[self.rel_features[0]].max()
        self.data[self.rel_features[0]] = normalize(self.data[self.rel_features[0]], self.min_val, self.max_val)

    def standardize_data(self):
        self.mean_val = self.data[self.rel_features[0]].mean()
        self.std_val = self.data[self.rel_features[0]].std()
        self.data[self.rel_features[0]] = standardize(self.data[self.rel_features[0]], self.mean_val, self.std_val)

    def denormalize_data(self):
        if self.min_val is None or self.max_val is None:
            raise DataNotNormalized("Data is not Normalized. ")
        self.data[self.rel_features[0]] = denormalize(self.data[self.rel_features[0]], self.min_val, self.max_val)

    def destandardize_data(self):
        if self.mean_val is None or self.std_val is None:
            raise DataNotNormalized("Data is not Standardized. ")
        self.data[self.rel_features[0]] = destandardize(self.data[self.rel_features[0]], self.mean_val, self.std_val)

    def save_data(self, filename):
        self.data.to_csv(DATA_DIR + filename)

    def infer_sampling_frequency(self):
        temp = pd.DataFrame()
        temp['dt'] = self.data.index
        temp['shifted'] = (temp['dt'] - temp['dt'].shift())
        elem, count = np.unique(temp['shifted'], return_counts=True)
        freq = elem[np.argmax(count)]
        freq = freq.astype('timedelta64[m]')
        return freq/np.timedelta64(1, 'm')

    def __len__(self):
        return self.data.shape[0]


class Network:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.model = None
        self.model_history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @abc.abstractmethod
    def preprocess_data(self):
        pass

    @abc.abstractmethod
    def set_train_test_split(self):
        pass

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def forecast_model(self, start_datetime, end_datetime, freq):
        pass

    def evaluate_model(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def save_model(self, filename):
        self.model.save_weights(MODEL_DIR + filename)

    def load_model(self, filename):
        self.model.load_weights(MODEL_DIR + filename)


class LSTMNetwork(Network):

    def __init__(self, data_handler):
        super(LSTMNetwork, self).__init__()
        self.data_handler = data_handler

    def preprocess_data(self, mode='standardize'):
        if mode == 'standardize':
            self.data_handler.standardize_data()
        elif mode == 'normalize':
            self.data_handler.normalize_data()

    def set_train_test_split(self):
        X, y = window_transform_series(self.data_handler.data, self.data_handler.rel_features[0])
        train_test_split = int(np.ceil(LSTM_TRAIN_TEST_SPLIT * len(y)))

        self.X_train = X[:train_test_split]
        self.X_test = X[train_test_split:]

        self.y_train = y[:train_test_split]
        self.y_test = y[train_test_split:]

    def build_model(self):
        np.random.seed(0)
        model = Sequential()
        for _ in range(NUM_LAYERS - 1):
            model.add(LSTM(NUM_CELLS_LSTM, input_shape=(WINDOW_SIZE, FEATURE_DIMENSION), dropout=LSTM_DROPOUT,
                           recurrent_dropout=LSTM_RECURRENT_DROPOUT, return_sequences=True))
        model.add(LSTM(NUM_CELLS_LSTM, input_shape=(WINDOW_SIZE, FEATURE_DIMENSION), dropout=LSTM_DROPOUT,
                       recurrent_dropout=LSTM_RECURRENT_DROPOUT))
        model.add(Dense(OUTPUT_DIMENSION))
        model.compile(loss=LSTM_LOSS_FUNCTION, optimizer=LSTM_OPTIMIZER)
        self.model = model

    def train_model(self):
        early_stopping = EarlyStopping(monitor=EARLY_STOP_METRIC, patience=LSTM_PATIENCE)
        K_b.set_session(K_b.tf.Session(config=K_b.tf.ConfigProto(intra_op_parallelism_threads=INTRA_OP_PARALLELISM_THREADS,
                                                           inter_op_parallelism_threads = INTER_OP_PARALLELISM_THREADS)))
        self.model.fit(np.asarray(self.X_train), np.asarray(self.y_train),
                       epochs=LSTM_EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       callbacks=[early_stopping, ],
                       validation_split=LSTM_VALIDATION_SPLIT)

    def forecast_model(self, start_datetime, end_datetime, freq):
        if freq is None:
            freq = str(self.data_handler.infer_sampling_frequency()) + 'Min'
        output_list = []
        input_list = self.X_test[-1]
        input_list = np.reshape(input_list, (1, input_list.shape[0], input_list.shape[1]))
        data_end_datetime = self.data_handler.data.index[-1]
        if Timestamp(start_datetime) < data_end_datetime:
            raise DateError("Start datetime is of present or past. Please enter a future datetime.")
        if end_datetime < start_datetime:
            raise DateWarning("End time is before start time. No predictions will be made.")
        date_prediction = pd.date_range(start=start_datetime,
                                        end=end_datetime, freq=freq)
        for i in range(len(date_prediction)):
            predicted_data = self.model.predict(input_list)
            predicted_data = np.append(predicted_data[0], [date_prediction[i].hour, date_prediction[i].minute])
            input_list = np.delete(input_list[0], obj=0, axis=0)
            input_list = np.append(input_list, np.reshape(predicted_data, (1, FEATURE_DIMENSION)), axis=0)
            input_list = np.asarray(np.reshape(input_list, (1, WINDOW_SIZE, FEATURE_DIMENSION)))
            output_list.append(predicted_data[0])
        output_list = destandardize(output_list, self.data_handler.mean_val, self.data_handler.std_val)
        output_list = pd.DataFrame(index=date_prediction, data=output_list)
        forecasted_list = pd.DataFrame(index=pd.date_range(start=start_datetime, end=end_datetime, freq=freq))
        desired_list = forecasted_list.join(output_list)
        return desired_list

    def run_model(self, weight_filename,  start_datetime, end_datetime, forecast_freq=None, train=False,
                  evaluate=False):
        """

        :param weight_filename: Filename containing the required weights for neural network
        :param start_datetime: For forecasting; Start date time of forecast. Standard datetime format has been used.
        :param end_datetime: For forecasting; End date time of forecast. Standard datetime format has been used.
        :param forecast_freq: For forecasting; Sampling frequency for forecasting. If None, then it is inferred from
        the past data, by taking mode of time difference.
        :param train: If true, then train the network. If false, then forecast using given weights.
        :param evaluate: If true, then show the evaluation on test set.

        :return: Pandas dataframe having datetime as index and corresponding forecasted result as values.
        """
        self.preprocess_data()
        self.set_train_test_split()
        self.build_model()
        if train:
            self.train_model()
            self.save_model(weight_filename)
            print(self.model.evaluate())
        else:
            self.load_model(weight_filename)
        if evaluate:
            s = self.evaluate_model()
            print(s)  # TODO: SET A LOGGER
        out = self.forecast_model(start_datetime=start_datetime, end_datetime=end_datetime, freq=forecast_freq)
        data = self.data_handler.data
        out = add_minmax_bound(dataframe=out, feature=self.data_handler.rel_features[0],
                               grp_std_dev=data.groupby([data.index.hour, data.index.minute]).std())
        return out
