import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from abc import ABC
from abc import abstractmethod
import ramphy.ramp_setup as rs
from copy import deepcopy


class TransformerBaseDataPreprocessor(rs.BaseDataPreprocessor):
    """Feature Preprocessor (apply to all models)."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        metadata: dict,
        y: np.ndarray,
    ) -> None:
        """Fit the transformer"""

    @abstractmethod
    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[dict]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[dict]]:
        """Transforms the data"""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        """Preprocess X_train and X_test.

        The number of rows should not be changed.

        Args:
            X_train (pd.DataFrame): train data
            y_train (np.ndarray): train labels
            X_test (pd.DataFrame): test data
            metadata (dict): metadata

        Returns:
            Tuple[ pd.DataFrame, np.ndarray, pd.DataFrame, dict]: X_train, y_train, X_test, metadata
        """
        self.fit(X=X_train, y=y_train, metadata=metadata)
        print(f"X train columns {len(X_train.columns)} before transform: {list(X_train.columns)}")
        X_train, y_train, metadata = self.transform(X=X_train, y=y_train, metadata=metadata)
        X_test, _, _ = self.transform(X=X_test, y=None, metadata=None)
        print(f"X train columns {len(X_train.columns)} after transform: {list(X_train.columns)}")
        return X_train, y_train, X_test, metadata

    def drop_metadata_features(self, metadata: dict, features: List[str]) -> dict:
        """Makes a copy of the metadata without the features to drop

        Args:
            metadata (dict): metadata
            features (List[str]): features to drop

        Returns:
            dict: copy of the metadata
        """
        new_metadata = deepcopy(metadata)
        for feat in features:
            new_metadata["data_description"]["feature_types"].pop(feat, None)
        return new_metadata
