from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# RAMP START HYPERPARAMETERS
removed_features = Hyperparameter(dtype="int", default=-1, values=[-1, -2, -3, -5, -10])
# RAMP END HYPERPARAMETERS

REMOVED_FEATURES = int(removed_features)


class DataPreprocessor(rs.TransformerBaseDataPreprocessor):
    """Uses random forest to select hte best features"""

    def fit(
        self,
        X: pd.DataFrame,
        metadata: dict,
        y: Optional[np.ndarray],
    ) -> None:
        """Fit preprocessing parameters on data

        Args:
            X (pd.DataFrame): _description_
            metadata (MetaData): _description_
            y (Optional[np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        K = max(X.shape[1] + REMOVED_FEATURES, 1)
        self.selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=K)
        if y is None:
            raise ValueError("y must be provided for feature selection")
        self.selector.fit(X, y)
        self.selected_features = self.selector.get_feature_names_out()
        self.dropped_features = list(set(X.columns) - set(self.selected_features))

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[dict]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[dict]]:
        selected_features = list(self.selector.get_feature_names_out())
        X_prepr = self.selector.transform(X)
        X_prepr = pd.DataFrame(X_prepr, columns=selected_features)
        if metadata is not None:
            metadata = self.drop_metadata_features(metadata=metadata, features=self.dropped_features)
        return X_prepr, y, metadata
