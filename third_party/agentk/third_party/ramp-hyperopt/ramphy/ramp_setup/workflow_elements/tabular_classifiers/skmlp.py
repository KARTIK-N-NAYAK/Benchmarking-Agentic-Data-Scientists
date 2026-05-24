import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import SplineTransformer
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
layers = Hyperparameter(dtype='str', default='512-256', values=['32', '64', '128', '256', '256-128', '512-256'])#, '1024-512', '1024-512-512'])
activation = Hyperparameter(dtype='str', default='tanh', values=['relu', 'tanh', 'logistic'])
alpha = Hyperparameter(dtype='float', default=0.1, values=[0.0001, 0.001, 0.01, 0.1])
learning_rate_init = Hyperparameter(dtype='float', default=0.01, values=[0.001, 0.01, 0.1])
max_iter = Hyperparameter(dtype='int', default=5000, values=[5000, 10000, 20000])
n_iter_no_change = Hyperparameter(dtype='int', default=10, values=[5, 10, 20])
beta_1 = Hyperparameter(dtype='float', default=0.8, values=[0.8, 0.9, 0.95])
beta_2 = Hyperparameter(dtype='float', default=0.999, values=[0.99, 0.999, 0.9999])
epsilon = Hyperparameter(dtype='float', default=1e-7, values=[1e-8, 1e-7, 1e-6])
n_knots = Hyperparameter(dtype='int', default=5, values=[3, 5, 10, 20])
# RAMP END HYPERPARAMETERS

LAYERS = list(map(int, str(layers).split("-")))
MAX_ITER = int(max_iter)
N_ITER_NO_CHANGE = int(n_iter_no_change)
ACTIVATION = str(activation)
SOLVER = 'adam'
ALPHA = float(alpha)
LEARNING_RATE_INIT = float(learning_rate_init)
BETA_1 = float(beta_1)
BETA_2 = float(beta_2)
EPSILON = float(epsilon)
N_KNOTS = int(n_knots)

class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        target_cols = metadata["data_description"]["target_cols"]
        if len(target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")            

    def fit(self, X, y):
        X_new = X.copy()
        self.spline_transformers = {{}}
        for col in X_new.columns:
            if self.metadata["data_description"]["feature_types"][col] == "num" and len(X_new[col].unique()) > 2:
                transformer = SplineTransformer(n_knots=N_KNOTS)
                X_tr = transformer.fit_transform(X_new[[col]])
                self.spline_transformers[col] = transformer
                n_new_cols = X_tr.shape[1]
                new_cols = [f"{{col}}_spline_{{i}}" for i in range(n_new_cols)]
                X_new[new_cols] = X_tr
                X_new = X_new.drop(columns=[col])
        self.clf = MLPClassifier(
            hidden_layer_sizes=LAYERS,
            early_stopping=True,
            max_iter=MAX_ITER,
            n_iter_no_change=N_ITER_NO_CHANGE,
            activation=ACTIVATION,
            solver=SOLVER,
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE_INIT,
            beta_1=BETA_1,
            beta_2=BETA_2,
            epsilon=EPSILON,
        )
        self.clf.fit(X_new, y)

    def predict_proba(self, X):
        X_new = X.copy()
        for col in X_new.columns:
            if col in self.spline_transformers.keys():
                transformer = self.spline_transformers[col]
                X_tr = transformer.transform(X_new[[col]])
                n_new_cols = X_tr.shape[1]
                new_cols = [f"{{col}}_spline_{{i}}" for i in range(n_new_cols)]
                X_new[new_cols] = X_tr
                X_new = X_new.drop(columns=[col])
        n_chunks = X_new.shape[0] * X_new.shape[1] / 10 ** 7
        n_rows = int(X_new.shape[0] / n_chunks)
        return np.concatenate([
            self.clf.predict_proba(X_new[i * n_rows:(i + 1) * n_rows])
            for i in range(int(X_new.shape[0] / n_rows) + 1)
        ], axis=0)