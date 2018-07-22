import os

from compute_features import get_sma, get_daily_returns

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = PROJECT_DIR + "/data/"
MODEL_DIR = PROJECT_DIR + "/models/"


# DATA EXTRACTION, TRANSFORMATION, LOADING

STOCKS = ['EOD/DIS']  # Currently, only one stock is supported, will add support for multiple stock prediction soon :):)
MODE = 'LOCAL'  # Either local or quandl. Tell the program from where to extract the data.
SAVE = True  # If mode is "quandl", then save the extracted datasets
RETRIEVAL_START_DATE = '2016-06-06'  # YYYY-MM-DD
RETRIEVAL_END_DATE = '2018-07-20'  # YYYY-MM-DD
INDEX_COLUMN = 'Date'  # If None, then use default indexing.
REL_DATA_COLUMNS = ['Adj_Close', 'Volume']
EXTRA_FEATURES = {'SMA': get_sma, 'daily_returns': get_daily_returns}
COLUMNS_TO_STANDARDIZE = ['Adj_Close', 'Volume', 'SMA', 'daily_returns']
COLUMNS_TO_NORMALIZE = []  # PRO TIP: Standardization works better in most cases.
CUSTOM_PREPROCESSOR_FP = []  # If you want custom preprocessing(s) on data, add function pointer to this list.
CUSTOM_PREPROCESSOR_COLUMNS = []  # Columns on which you want to do custom preprocessing.
FEATURE_TO_PREDICT = ['Adj_Close']

# DEFAULT NEURAL NETWORK CONFIGURATION

LSTM_OPTIMIZER = 'adam'
LSTM_EPOCHS = 50
LSTM_PATIENCE = 15  # Early Stopping
LSTM_LOSS_FUNCTION = 'mean_squared_error'
WINDOW_SIZE = 5
NUM_CELLS_LSTM = 100
BATCH_SIZE = 256
VERBOSE = 0
NUM_LAYERS = 2
LSTM_VALIDATION_SPLIT = 0
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.1
LSTM_TRAIN_TEST_SPLIT = 0.85
FEATURE_DIMENSION = len(REL_DATA_COLUMNS) + len(EXTRA_FEATURES)
OUTPUT_DIMENSION = len(FEATURE_TO_PREDICT)
EARLY_STOP_METRIC = 'loss'


# TENSORFLOW CPU OPTIMIZATION
# PARALLELISM CONTROLLER
USE_OPTIMIZER = False  # Set it True if you observe regukar crashes in program.
INTRA_OP_PARALLELISM_THREADS = 1  # RECOMMENDED VALUE: SET IT EQUAL TO NUMBER OF PHYSICAL CORES
INTER_OP_PARALLELISM_THREADS = 1  # RECOMMENDED VALUE: SET IT EQUAL TO NUMBER OF PHYSICAL CORES

# QUANDL SETTINGS
with open('secrets.txt', 'r') as f:
    keys = f.read()
QUANDL_KEY = eval(keys)['quandl_key']

# TODO: Set logging configurations
