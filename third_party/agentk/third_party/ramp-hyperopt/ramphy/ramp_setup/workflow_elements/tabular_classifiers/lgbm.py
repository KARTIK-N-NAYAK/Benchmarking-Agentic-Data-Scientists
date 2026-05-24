import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
colsample_bytree = Hyperparameter(dtype='float', default=0.5, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
colsample_bynode = Hyperparameter(dtype='float', default=0.5, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
min_split_gain = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
learning_rate = Hyperparameter(dtype='float', default=0.05, values=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
max_depth = Hyperparameter(dtype='int', default=5, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
min_child_weight = Hyperparameter(dtype='int', default=9, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_estimators = Hyperparameter(dtype='int', default=400, values=[10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 1000])#, 2000, 3000, 5000, 7000, 10000])
reg_alpha = Hyperparameter(dtype='float', default=2.0, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
reg_lambda = Hyperparameter(dtype='float', default=5.0, values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
subsample = Hyperparameter(dtype='float', default=0.9, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
max_bin = Hyperparameter(dtype='int', default=256, values=[256, 512, 1024, 2048, 4096, 8192])
min_data_in_leaf = Hyperparameter(dtype='int', default=1, values=[1, 5, 10, 20, 50, 100, 200, 500, 700])
boosting_type = Hyperparameter(dtype='str', default='gbdt_5', values=['gbdt_0', 'gbdt_1', 'gbdt_5', 'gbdt_10', 'dart_0', 'dart_1', 'dart_5', 'dart_10', 'goss'])
drop_rate = Hyperparameter(dtype='float', default=0.1, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# RAMP END HYPERPARAMETERS

COLSAMPLE_BYTREE = float(colsample_bytree)
COLSAMPLE_BYNODE = float(colsample_bynode)
MIN_SPLIT_GAIN = float(min_split_gain)  # Using min_split_gain as a similar concept to gamma
LEARNING_RATE = float(learning_rate)
MAX_DEPTH = int(max_depth)
MIN_CHILD_SAMPLES = int(min_child_weight)  # min_child_samples in LightGBM
N_ESTIMATORS = int(n_estimators)
REG_ALPHA = float(reg_alpha)
REG_LAMBDA = float(reg_lambda)
SUBSAMPLE = float(subsample)
MAX_BIN = int(max_bin)
MIN_DATA_IN_LEAF = int(min_data_in_leaf)
BOOSTING_TYPE = str(boosting_type)
if BOOSTING_TYPE[:4] == 'dart':
    BAGGING_FREQ = int(BOOSTING_TYPE[5:])
    BOOSTING_TYPE = 'dart'
elif BOOSTING_TYPE[:4] == 'gbdt':
    BAGGING_FREQ = int(BOOSTING_TYPE[5:])
    BOOSTING_TYPE = 'gbdt'
else:
    BAGGING_FREQ = None
DROP_RATE = float(drop_rate)


class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        target_cols = metadata["data_description"]["target_cols"]
        if len(target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")            
        target_value_dict = metadata["data_description"]["target_values"]
        target_values = target_value_dict[target_cols[0]]
        if len(target_values) == 2:
            self.objective = 'binary'
        else:
            self.objective = 'multiclass'

    def fit(self, X, y):
        self.clf = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            min_split_gain=MIN_SPLIT_GAIN,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            colsample_bynode=COLSAMPLE_BYNODE,
            reg_alpha=REG_ALPHA,
            reg_lambda=REG_LAMBDA,
            min_child_samples=MIN_CHILD_SAMPLES,
            max_bin=MAX_BIN,
            bagging_freq=BAGGING_FREQ,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
            boosting_type=BOOSTING_TYPE,
            drop_rate=DROP_RATE,
            objective=self.objective,
            verbose=-1,
        )
        self.clf.fit(X, y.ravel())

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba
