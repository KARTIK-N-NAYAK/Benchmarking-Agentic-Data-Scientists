from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Does nothing"""

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: dict
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], dict]:
        return X, y, metadata
