import faiss
import numpy as np
from ramphy import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y

# RAMP START HYPERPARAMETERS
n_neighbors = Hyperparameter(
    dtype="int",
    default=100,
    values=[
        100,
        400,
    ],
)
weights = Hyperparameter(
    dtype="str",
    default="uniform",
    values=["uniform", "distance"],
)
# RAMP END HYPERPARAMETERS

N_NEIGHBORS = int(n_neighbors)
WEIGHTS = str(weights)


class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, device="gpu"):
        self.n_neighbors = N_NEIGHBORS
        self.weights = WEIGHTS
        self.device = device

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.index = faiss.IndexFlatL2(X.shape[1])
        if self.device == "gpu":
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(X.astype(np.float32))
        self.y = y
        return self

    def predict(self, X):
        X = check_array(X).astype(np.float32)
        dist, I = self.index.search(X, self.n_neighbors)
        with np.errstate(divide="ignore"):
            if self.weights == "uniform":
                dist = np.ones(dist.shape)
            elif self.weights == "distance":
                dist = 1 / dist
            else:
                assert hasattr(self.weights, "__call__")
                dist = self.weights(dist)
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return np.average(self.y[I], axis=1, weights=dist)
