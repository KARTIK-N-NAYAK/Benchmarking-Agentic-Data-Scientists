import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xb
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
colsample_bytree = Hyperparameter(dtype='float', default=0.5, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
colsample_bylevel = Hyperparameter(dtype='float', default=0.5, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
colsample_bynode = Hyperparameter(dtype='float', default=0.5, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gamma = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
learning_rate = Hyperparameter(dtype='float', default=0.1, values=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
max_depth = Hyperparameter(dtype='int', default=2, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
min_child_weight = Hyperparameter(dtype='int', default=1, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_estimators = Hyperparameter(dtype='int', default=700, values=[10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 1000])#, 2000, 3000, 5000, 7000, 10000])
reg_alpha = Hyperparameter(dtype='float', default=0.1, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
reg_lambda = Hyperparameter(dtype='float', default=0.5, values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
subsample = Hyperparameter(dtype='float', default=1.0, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# RAMP END HYPERPARAMETERS

N_ESTIMATORS = int(n_estimators)
MAX_DEPTH = int(max_depth)
LEARNING_RATE = float(learning_rate)
MIN_CHILD_WEIGHT = int(min_child_weight)
GAMMA = float(gamma)
SUBSAMPLE = float(subsample)
COLSAMPLE_BYTREE = float(colsample_bytree)
COLSAMPLE_BYLEVEL = float(colsample_bylevel)
COLSAMPLE_BYNODE = float(colsample_bynode)
REG_ALPHA = float(reg_alpha)
REG_LAMBDA = float(reg_lambda)

class Regressor(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        score_name = metadata["score_name"]
        if score_name in ["mse", "rmse", "rmsle", "r2"]:
            self.objective = "reg:squarederror"
        elif score_name in ["rmsle"]:
            self.objective = "reg:squaredlogerror"
        elif score_name in ["mae", "medae", "smape"]:
            self.objective = "reg:absoluteerror"
        else:
            raise ValueError(f"Unknown score_name {score_name}")

    def fit(self, X, y):
        if self.metadata["score_name"] == "rmsle":
            y = np.log1p(y)
        self.reg = xb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            min_child_weight=MIN_CHILD_WEIGHT,
            gamma=GAMMA,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            colsample_bylevel=COLSAMPLE_BYLEVEL,
            colsample_bynode=COLSAMPLE_BYNODE,
            reg_alpha=REG_ALPHA,
            reg_lambda=REG_LAMBDA,
            objective=self.objective,
        )
        self.reg.fit(X, y)

    def predict(self, X):
        y_pred = self.reg.predict(X)
        if self.metadata["score_name"] == "rmsle":
            y_pred = np.expm1(y_pred)
        return y_pred
