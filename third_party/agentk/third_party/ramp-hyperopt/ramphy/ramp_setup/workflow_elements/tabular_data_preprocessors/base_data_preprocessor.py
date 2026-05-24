import numpy as np
import pandas as pd
from typing import Tuple
from abc import ABC
from abc import abstractmethod


class BaseDataPreprocessor(ABC):
    """Feature Preprocessor (apply to all models)."""

    @abstractmethod
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
