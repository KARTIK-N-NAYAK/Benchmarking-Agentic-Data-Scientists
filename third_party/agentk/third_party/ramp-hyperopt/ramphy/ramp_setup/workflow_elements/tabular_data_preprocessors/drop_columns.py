import os
from typing import Optional, Tuple
import json
import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
# RAMP END HYPERPARAMETERS

class DataPreprocessor(rs.BaseDataPreprocessor):
    """Drops columns."""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        cols_to_drop = []
        for col in X_train.columns:
            if bool(eval(f"{{col}}_{hyper_suffix}")):
                cols_to_drop.append(col)
        print(f"Dropping {{cols_to_drop}}")
        X_train = X_train.drop(cols_to_drop, axis=1)
        X_test = X_test.drop(cols_to_drop, axis=1)
        for col in cols_to_drop:
            metadata["data_description"]["feature_types"].pop(col)
        return X_train, y_train, X_test, metadata
