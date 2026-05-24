import re
import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from sklearn.compose import ColumnTransformer
from skrub import MinHashEncoder


# RAMP START HYPERPARAMETERS
n_components{col} = Hyperparameter(
    dtype="int", default=30, values=[10, 30, 50, 100, 200]
)
ngram_range{col} = Hyperparameter(
    dtype="str", default='2-4', values=['2-4', '2-3', '3-5', '4-6']
)
hashing{col} = Hyperparameter(
    dtype="str", default="fast", values=["fast", "murmur"]
)
minmax_hash{col} = Hyperparameter(
    dtype="bool", default=False, values=[True, False]
)
# RAMP END HYPERPARAMETERS
N_COMPONENTS = int(n_components{col})
NGRAM_RANGE = tuple(map(int, str(ngram_range{col}).split("-")))
HASHING = str(hashing{col})
MINMAX_HASH = bool(minmax_hash{col})


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes text feature"""

    def __init__(self, col={str_col}):
        self.to_cache = True
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        X = pd.concat((X_train, X_test))
        transformer = MinHashEncoder(
            n_components=N_COMPONENTS,
            ngram_range=NGRAM_RANGE,
            hashing=HASHING,
            minmax_hash=MINMAX_HASH,
        )
        if self.col in X_train.columns:
            X[self.col] = X[self.col].astype(str)
            transformer.fit(X[self.col])
            new_columns = transformer.get_feature_names_out()
            converted_columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) + f"_{{i}}" for i, col in enumerate(new_columns)]
            X_transformed_df = transformer.transform(X[self.col])
    #        col_rename = {{
    #            name: new_name
    #            for name, new_name in zip(X_transformed_df.columns, converted_columns)}}
    #        X_transformed_df = X_transformed_df.rename(columns=col_rename, errors='raise')
            X_transformed_df = pd.DataFrame(
                X_transformed_df.to_numpy(), columns=converted_columns, index=X.index)
            X = pd.concat((X, X_transformed_df), axis=1)
    
            metadata["data_description"]["feature_types"].pop(self.col)
            for col in converted_columns:
                metadata["data_description"]["feature_types"][col] = "num"
    
            X = X.drop(columns=[self.col])
            X_train = X.iloc[:len(X_train)]
            X_test = X.iloc[len(X_train):]

        return X_train, y_train, X_test, metadata
