import abc


class DataHandler:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.data = None
        self.name = None
        self.rel_stats = None

    @abc.abstractmethod
    def add(self, data_frame, name):
        pass

    @abc.abstractmethod
    def preprocess_data(self):
        pass

    def __len__(self):
        return self.data.shape[0]
