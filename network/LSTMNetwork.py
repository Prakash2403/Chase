from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

from network import Network
from utils import *


class LSTMNetwork(Network):
    def __init__(self, data_handler):
        super(LSTMNetwork, self).__init__()
        self.data_handler = data_handler

    def set_train_test_split(self):
        x, y = window_transform_series(self.data_handler.data, FEATURE_TO_PREDICT)
        train_test_split = int(np.ceil(LSTM_TRAIN_TEST_SPLIT * len(y)))

        self.X_train = x[:train_test_split]
        self.X_test = x[train_test_split:]

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
        self.model.fit(self.X_train, self.y_train,
                       epochs=LSTM_EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       callbacks=[early_stopping, ],
                       validation_split=LSTM_VALIDATION_SPLIT)

    def forecast_model(self, start_datetime, end_datetime, freq):
        # if freq is None:
        #     freq = str(self.data_handler.infer_sampling_frequency()) + 'Min'
        output_list = []
        input_list = self.X_test[-1]
        input_list = np.reshape(input_list, (1, input_list.shape[0], input_list.shape[1]))
        # data_end_datetime = self.data_handler.data.index[-1]
        # if Timestamp(start_datetime) < data_end_datetime:
        #     raise DateError("Start datetime is of present or past. Please enter a future datetime.")
        # if end_datetime < start_datetime:
        #     raise DateWarning("End time is before start time. No predictions will be made.")
        # date_prediction = pd.date_range(start=start_datetime,
        #                                 end=end_datetime, freq=freq)
        # print(date_prediction)
        # for i in range(len(date_prediction)):
        predicted_data = self.model.predict(input_list)
        output_list.append(predicted_data[0])
        mean_val = self.data_handler.rel_stats[FEATURE_TO_PREDICT].loc['mean'].values
        std_val = self.data_handler.rel_stats[FEATURE_TO_PREDICT].loc['std'].values
        output_list = destandardize(output_list, mean_val, std_val)
        return output_list
        #     predicted_data = np.append(predicted_data[0], [date_prediction[i].hour, date_prediction[i].minute])
        #     input_list = np.delete(input_list[0], obj=0, axis=0)
        #     input_list = np.append(input_list, np.reshape(predicted_data, (1, FEATURE_DIMENSION)), axis=0)
        #     input_list = np.asarray(np.reshape(input_list, (1, WINDOW_SIZE, FEATURE_DIMENSION)))
        # output_list = pd.DataFrame(index=date_prediction, data=output_list)
        # forecasted_list = pd.DataFrame(index=pd.date_range(start=start_datetime, end=end_datetime, freq=freq))
        # desired_list = forecasted_list.join(output_list)
        # return desired_list

    def visualize_output(self):
        _, y = window_transform_series(self.data_handler.data, FEATURE_TO_PREDICT)
        train_test_split = int(np.ceil(LSTM_TRAIN_TEST_SPLIT * len(y)))
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        mean_val = self.data_handler.rel_stats[FEATURE_TO_PREDICT].loc['mean'].values
        std_val = self.data_handler.rel_stats[FEATURE_TO_PREDICT].loc['std'].values
        plt.plot(destandardize(self.data_handler.data[FEATURE_TO_PREDICT].values,
                               mean_val, std_val), color='k')
        split_pt = train_test_split + WINDOW_SIZE
        plt.plot(np.arange(WINDOW_SIZE, split_pt, 1), destandardize(train_predict, mean_val, std_val), color='b')
        plt.plot(np.arange(split_pt, split_pt + len(test_predict), 1), destandardize(test_predict, mean_val, std_val)
                 , color='r')
        plt.xlabel(self.data_handler.name)
        plt.ylabel("Price")
        plt.legend(['original series', 'training fit', 'testing fit'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def run_model(self, weight_filename, start_datetime=None, end_datetime=None, forecast_freq=None, train=False,
                  evaluate=False, visualize=False):
        """

        :param weight_filename: Filename containing the required weights for neural network
        :param start_datetime: For forecasting; Start date time of forecast. Standard datetime format has been used.
        :param end_datetime: For forecasting; End date time of forecast. Standard datetime format has been used.
        :param forecast_freq: For forecasting; Sampling frequency for forecasting. If None, then it is inferred from
        the past data, by taking mode of time difference.
        :param train: If true, then train the network. If false, then forecast using given weights.
        :param evaluate: If true, then show the evaluation on test set.
        :param visualize: If True then plot the train test prediction and actual data on graph.

        :return: Pandas dataframe having datetime as index and corresponding forecasted result as values.
        """
        self.data_handler.preprocess_data()
        self.set_train_test_split()
        self.build_model()
        if train:
            self.train_model()
            self.save_model(weight_filename)
        else:
            self.load_model(weight_filename)
        if evaluate:
            s = self.evaluate_model()
            print(s)
        if visualize:
            self.visualize_output()
        out = self.forecast_model(start_datetime=start_datetime, end_datetime=end_datetime, freq=forecast_freq)
        return out
