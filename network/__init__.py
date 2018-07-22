import abc

from config import MODEL_DIR


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

    @abc.abstractmethod
    def visualize_output(self):
        pass

    def evaluate_model(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def save_model(self, filename):
        self.model.save_weights(MODEL_DIR + filename)

    def load_model(self, filename):
        self.model.load_weights(MODEL_DIR + filename)
