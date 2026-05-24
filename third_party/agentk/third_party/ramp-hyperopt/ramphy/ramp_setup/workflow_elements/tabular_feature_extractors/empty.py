from typing import Optional

import numpy as np
import pandas as pd


class FeatureExtractor:
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
