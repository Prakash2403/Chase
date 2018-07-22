from datahandler import DataHandler
from utils import *
from utils.compute_stock_features import *


class StockDataHandler(DataHandler):

    def __init__(self):
        super(StockDataHandler, self).__init__()

    def add(self, data: pd.DataFrame, name: str):
        self.data = data
        self.rel_stats = {}
        for col in REL_DATA_COLUMNS:
            self.rel_stats[col] = {
                              'mean': self.data[col].mean(),
                              'std': self.data[col].std(),
                              'min': self.data[col].min(),
                              'max': self.data[col].max()
                              }
        self.rel_stats = pd.DataFrame(self.rel_stats)
        self.name = name

    def preprocess_data(self):
        for feature in REL_PREDEFINED_FEATURES:
            self.data[feature] = eval(feature.lower())(self.data, FEATURE_TO_PREDICT)
        for key in EXTRA_FEATURES:
            self.data[key] = EXTRA_FEATURES[key](self.data, FEATURE_TO_PREDICT)
        for col in COLUMNS_TO_STANDARDIZE:
            self.data[col] = standardize(self.data[col])
        for col in COLUMNS_TO_NORMALIZE:
            self.data[col] = normalize(self.data[col])
        for col in CUSTOM_PREPROCESSOR_COLUMNS:
            for fp in CUSTOM_PREPROCESSOR_FP:
                self.data[col] = fp(self.data[col])
        self.data = self.data.dropna(axis='index')

    def __len__(self):
        return self.data.shape[0]
