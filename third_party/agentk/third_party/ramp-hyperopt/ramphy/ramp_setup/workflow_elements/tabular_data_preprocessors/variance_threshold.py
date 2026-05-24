from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from sklearn.feature_selection import VarianceThreshold

# RAMP START HYPERPARAMETERS
threshold = Hyperparameter(
    dtype="float",
    default=0.0,
    values=[0.0, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
)
# RAMP END HYPERPARAMETERS

THRESHOLD = int(threshold)


class DataPreprocessor(rs.TransformerBaseDataPreprocessor):
    """Uses random forest to select hte best features"""

    def fit(
        self,
        X: pd.DataFrame,
        metadata: dict,
        y: np.ndarray,
    ) -> None:
        """Fit preprocessing parameters on data

        Args:
            X (pd.DataFrame): _description_
            metadata (MetaData): _description_
            y (Optional[np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        self.selector = VarianceThreshold(threshold=THRESHOLD)
        self.selector.fit(X, y)
        # Saves the features
        self.features_set = set(self.selector.get_feature_names_out())
        self.dropped_features = list(set(X.columns) - self.features_set)
        print(f"Variance Preprocessor - Dropping: {{self.dropped_features}}")

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[dict]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[dict]]:
        selected_features = list(self.selector.get_feature_names_out())
        X_prepr = self.selector.transform(X)
        X_prepr = pd.DataFrame(X_prepr, columns=selected_features)
        if metadata is not None:
            metadata = self.drop_metadata_features(metadata=metadata, features=self.dropped_features)
        return X_prepr, y, metadata
