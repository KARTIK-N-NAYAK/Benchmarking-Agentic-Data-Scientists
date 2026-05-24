import re
import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Modifies LGBM invalid column names"""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        feature_types = metadata["data_description"]["feature_types"]
        feature_values = metadata["data_description"]["feature_values"]
        new_cols = {{col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in feature_types}}
        X_train = X_train.rename(columns=new_cols)
        X_test = X_test.rename(columns=new_cols)
        metadata["data_description"]["feature_types"] = {{
            new_cols[col]: col_type for col, col_type in feature_types.items()
        }}
        if feature_values is not None:
            metadata["data_description"]["feature_values"] = {{
                new_cols[col]: col_type for col, col_type in feature_values.items()
            }}
        return X_train, y_train, X_test, metadata
