import re
import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs
from category_encoders import BinaryEncoder
from category_encoders import CountEncoder
from category_encoders import HashingEncoder
from category_encoders import TargetEncoder
from ramphy import Hyperparameter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# RAMP START HYPERPARAMETERS
encoding_strategy{col} = Hyperparameter(
    dtype="str", default="OneHot", values=["OneHot", "Count", "Target", "Binary", "Hashing_01", "Hashing_05", "Hashing_10", "Hashing_20", "Hashing_30", "Hashing_40", "Hashing_50", "Hashing_70"]
)
r_features_for_hashing{col} = Hyperparameter(dtype="float", default=0.3, values=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
# RAMP END HYPERPARAMETERS

ENCODING_STRATEGY = str(encoding_strategy{col})
if ENCODING_STRATEGY[:7] == "Hashing":
    R_FEATURES_FOR_HASHING = int(str(ENCODING_STRATEGY[8:]))
    ENCODING_STRATEGY = "Hashing"


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes categorical feature"""

    def __init__(self, col={str_col}):
        self.to_cache = ENCODING_STRATEGY in ["Hashing", "Count", "Target", "Binary"]
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        global ENCODING_STRATEGY
        global R_FEATURES_FOR_HASHING
        if self.col in X_train.columns:
            # For now drop high-cardinality columns, later maybe dirty_cat
            if len(X_train[self.col].unique()) > 20:
#                X_train = X_train.drop(columns=self.col)
#                X_test = X_test.drop(columns=self.col)
#                metadata["data_description"]["feature_types"].pop(self.col)
                ENCODING_STRATEGY = "Hashing"
                N_FEATURES_FOR_HASHING = 20
#            else:
            if ENCODING_STRATEGY == "OneHot":
                # to avoid non authorized characters in column names
                def feature_name_combiner(input_feature, category):
                    output_feature = input_feature + '_' + str(category)
                    output_feature = re.sub(r'[^a-zA-Z0-9_]', '_', output_feature)
                    return output_feature
    
                transformer = OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    drop='if_binary',
                    feature_name_combiner=feature_name_combiner)
            elif ENCODING_STRATEGY == "Count":
                transformer = CountEncoder(handle_unknown=0, min_group_size=1, cols=[self.col])
            elif ENCODING_STRATEGY == "Target":
                transformer = TargetEncoder(handle_unknown="value", min_samples_leaf=1, cols=[self.col])
            elif ENCODING_STRATEGY == "Binary":
                transformer = BinaryEncoder(handle_unknown="value", cols=[self.col])
            elif ENCODING_STRATEGY == "Hashing":
                n_unique_values = np.sum([len(v) for v in metadata["data_description"]["feature_values"]])
#                n_features_for_hashing = max(1, int(round(R_FEATURES_FOR_HASHING * n_unique_values)))
                n_features_for_hashing = N_FEATURES_FOR_HASHING
                transformer = HashingEncoder(drop_invariant=True, n_components=n_features_for_hashing, cols=[self.col])
    
            if ENCODING_STRATEGY == "Target":
                transformer.fit(X_train[[self.col]], y_train)
            else:
                transformer.fit(pd.concat([X_train, X_test])[[self.col]])
    
            new_columns = transformer.get_feature_names_out([self.col])
            if ENCODING_STRATEGY == "Hashing":
                new_columns = [f"{{self.col}}_{{c}}" for c in new_columns]
    
            X_transformed = transformer.transform(X_train[[self.col]])
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            else:
                X_transformed = X_transformed.values
            X_train = X_train.drop(columns=[self.col])
            X_transformed_df = pd.DataFrame(X_transformed, columns=new_columns, index=X_train.index)
            X_train = pd.concat((X_train, X_transformed_df), axis=1)
            for col in new_columns:
                X_train[col] = pd.to_numeric(X_train[col], downcast="integer")
    
            X_transformed = transformer.transform(X_test[[self.col]])
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            else:
                X_transformed = X_transformed.values
            X_test = X_test.drop(columns=[self.col])
            X_transformed_df = pd.DataFrame(X_transformed, columns=new_columns, index=X_test.index)
            X_test = pd.concat((X_test, X_transformed_df), axis=1)
            # X_test[new_columns] = X_transformed
            for col in new_columns:
                X_test[col] = pd.to_numeric(X_test[col], downcast="integer")
    
            metadata["data_description"]["feature_types"].pop(self.col)
            for col in new_columns:
                metadata["data_description"]["feature_types"][col] = "num"
        
        return X_train, y_train, X_test, metadata
