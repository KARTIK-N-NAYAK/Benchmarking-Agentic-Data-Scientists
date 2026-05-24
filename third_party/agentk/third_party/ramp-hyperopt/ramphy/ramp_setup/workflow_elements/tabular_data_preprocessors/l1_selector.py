from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# RAMP START HYPERPARAMETERS
selector_C = Hyperparameter(dtype="float", default=0.001, values=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
# RAMP END HYPERPARAMETERS

SELECTOR_C = float(selector_C)


class DataPreprocessor(rs.TransformerBaseDataPreprocessor):
    """Uses Logistic regression with L1 to select the best features"""

    def fit(
        self,
        X: pd.DataFrame,
        metadata: dict,
        y: Optional[np.ndarray],
    ) -> None:
        self.selector = SelectFromModel(LogisticRegression(penalty="l1", C=SELECTOR_C, solver="liblinear"))

        if y is None:
            raise ValueError("y must be provided for feature selection")
        self.selector.fit(X, y)
        # Saves the features
        self.features_set = set(self.selector.get_feature_names_out())
        self.dropped_features = list(set(X.columns) - self.features_set)

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[dict]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[dict]]:
        selected_features = list(self.selector.get_feature_names_out())
        X_prepr = self.selector.transform(X)
        X_prepr = pd.DataFrame(X_prepr, columns=selected_features)
        if metadata is not None:
            metadata = self.drop_metadata_features(metadata=metadata, features=self.dropped_features)
        return X_prepr, y, metadata
