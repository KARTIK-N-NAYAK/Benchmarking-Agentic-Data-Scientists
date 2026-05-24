import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Removes columns that are only in training dataset.

    This should be done differently as these columns could be used in some other
    way to build a better model.
    """

    def preprocess(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:

        col_sym_diff = set(X_train.columns).symmetric_difference(
            set(X_test.columns))
        try:
            X_train = X_train.drop(col_sym_diff, axis=1)
        except KeyError:
            pass
        try:
            X_test = X_test.drop(col_sym_diff, axis=1)
        except KeyError:
            pass

        for col in col_sym_diff:
            metadata["data_description"]["feature_types"].pop(col)

        return X_train, y_train, X_test, metadata
