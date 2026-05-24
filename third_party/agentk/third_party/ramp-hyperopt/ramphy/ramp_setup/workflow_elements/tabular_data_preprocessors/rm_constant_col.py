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

        cols_to_drop = []

        for col in X_train.columns:
            if X_train[col].nunique() == 1 and X_test[col].nunique() == 1:
                if X_train[col].iloc[0] == X_test[col].iloc[0]:
                    cols_to_drop.append(col)

        # Drop these constant columns from both dataframes
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        for col in cols_to_drop:
            metadata["data_description"]["feature_types"].pop(col)

        return X_train, y_train, X_test, metadata
