import numpy as np
import catboost as cb
from _catboost import CatBoostError
from sklearn.base import BaseEstimator
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
n_estimators = Hyperparameter(dtype='int', default=400, values=[10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 1000])#, 2000, 3000, 5000, 7000, 10000])
learning_rate = Hyperparameter(dtype='float', default=0.05, values=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
max_depth = Hyperparameter(dtype='int', default=5, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
l2_leaf_reg = Hyperparameter(dtype='float', default=3.0, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
border_count = Hyperparameter(dtype='int', default=254, values=[32, 64, 128, 254, 512, 1024])
grow_policy = Hyperparameter(dtype='str', default='SymmetricTree', values=['SymmetricTree', 'Depthwise', 'Lossguide'])
min_data_in_leaf = Hyperparameter(dtype='int', default=1, values=[1, 5, 10, 20, 50, 100, 200, 500, 700])
bootstrap_type = Hyperparameter(dtype='str', default='No', values=['No', 'Bernoulli', 'MVS', 'Bayesian_0', 'Bayesian_1', 'Bayesian_5', 'Bayesian_10', 'Bayesian_20', 'Bayesian_50'])
random_strength = Hyperparameter(dtype='float', default=1, values=[0, 1, 5, 10, 20, 50, 100])
# RAMP END HYPERPARAMETERS

ITERATIONS = int(n_estimators)
LEARNING_RATE = float(learning_rate)
DEPTH = int(max_depth)
L2_LEAF_REG = float(l2_leaf_reg)
BORDER_COUNT = int(border_count)
GROW_POLICY = str(grow_policy)
MIN_DATA_IN_LEAF = int(min_data_in_leaf)
BOOTSTRAP_TYPE = str(bootstrap_type)
if BOOTSTRAP_TYPE[:8] == 'Bayesian':
    BAGGING_TEMPERATURE = float(BOOTSTRAP_TYPE[9:])
    BOOTSTRAP_TYPE = "Bayesian"
else:
    BAGGING_TEMPERATURE = None
RANDOM_STRENGTH = float(random_strength)

class Regressor(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        self.cat_features = [
            col for col, type in
            metadata["data_description"]["feature_types"].items()
            if type == "cat"
        ]
        score_name = metadata["score_name"]
        if score_name in ["mse", "rmse", "rmsle", "r2"]:
            self.objective = "RMSE"
        elif score_name in ["mae", "medae", "smape"]:
            self.objective = "MAE"
        elif score_name == "mape":
            self.objective = "MAPE"
        else:
            raise ValueError(f"Unknown score_name {score_name}")

    def fit(self, X, y):
        if self.metadata["score_name"] == "rmsle":
            y = np.log1p(y)
        if y.ndim > 1 and y.shape[1] > 1:
            # only loss function available for multi output regression
            self.objective = 'MultiRMSE'

        self.reg = cb.CatBoostRegressor(
            cat_features=self.cat_features,
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            l2_leaf_reg=L2_LEAF_REG,
            border_count=BORDER_COUNT,
            bagging_temperature=BAGGING_TEMPERATURE,
            grow_policy=GROW_POLICY,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
            bootstrap_type=BOOTSTRAP_TYPE,
            random_strength=RANDOM_STRENGTH,
            loss_function=self.objective,
            verbose=False,
        )
        try:
            self.reg.fit(X, y)
            return
        except CatBoostError:
            # Catching mysterious _catboost.CatBoostError:
            # /src/catboost/catboost/private/libs/lapack/linear_system.cpp:31:
            # System of linear equations is not positive definite
            for i in range(2, 10):
                print(f"Catboost crashed, attempt no {{i}}")
                try:
                    self.reg = cb.CatBoostRegressor(
                        cat_features=self.cat_features,
                        iterations=ITERATIONS,
                        learning_rate=LEARNING_RATE,
                        depth=DEPTH,
                        l2_leaf_reg=L2_LEAF_REG,
                        border_count=BORDER_COUNT,
                        bagging_temperature=BAGGING_TEMPERATURE,
                        grow_policy=GROW_POLICY,
                        min_data_in_leaf=MIN_DATA_IN_LEAF,
                        bootstrap_type=BOOTSTRAP_TYPE,
                        random_strength=RANDOM_STRENGTH,
                        loss_function=self.objective,
                        verbose=False,
                        random_seed=i,
                    )
                    self.reg.fit(X, y)
                    return
                except CatBoostError:
                    pass
        print("Falling back to default")
        self.reg = cb.CatBoostRegressor(
            cat_features=self.cat_features,
            loss_function=self.objective,
            verbose=False,
        )
        self.reg.fit(X, y)

    def predict(self, X):
        y_pred = self.reg.predict(X)
        if self.metadata["score_name"] == "rmsle":
            y_pred = np.expm1(y_pred)
        return y_pred
