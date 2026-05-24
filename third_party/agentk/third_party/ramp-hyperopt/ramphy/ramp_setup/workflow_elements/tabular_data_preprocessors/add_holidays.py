from typing import Tuple
import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter

import holidays


def is_holiday(date, country):
    country_holidays = holidays.CountryHoliday(country)
    return date in country_holidays


# RAMP START HYPERPARAMETERS
# RAMP END HYPERPARAMETERS



class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes categorical feature"""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:

        for df in [X_train, X_test]:
            df["generated_is_holiday"] = df.apply(
                lambda row: is_holiday(row[{date_col}], row[{location_col}]), axis=1
            ).astype(int)

        metadata["data_description"]["feature_types"]["generated_is_holiday"] = "num"

        return X_train, y_train, X_test, metadata

