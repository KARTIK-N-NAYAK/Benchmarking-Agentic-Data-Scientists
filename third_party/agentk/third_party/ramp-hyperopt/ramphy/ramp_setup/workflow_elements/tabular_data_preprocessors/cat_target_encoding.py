import os
from typing import Optional, Tuple
import json
import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from category_encoders import TargetEncoder

# RAMP START HYPERPARAMETERS
# RAMP END HYPERPARAMETERS

class DataPreprocessor(rs.BaseDataPreprocessor):
    """Target encodes columns."""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        cols_to_encode = []
        for col in X_train.columns:
#            if metadata["data_description"]["feature_types"][col] == "cat":
            try:
                if bool(eval(f"{{col}}_{hyper_suffix}")):
                    cols_to_encode.append(col)
            except NameError:
                pass
    
        print(f"Encoding {{cols_to_encode}} (does not work properly for multiclass yet)")
        transformer = TargetEncoder(handle_unknown="value", min_samples_leaf=1, cols=cols_to_encode)
        transformer.fit(X_train[cols_to_encode], y_train)
        new_columns_orig = transformer.get_feature_names_out(cols_to_encode)
        new_columns = [f"{{col}}_target_encode" for col in new_columns_orig]
        if len(new_columns) > 0:
            X_transformed = transformer.transform(X_train[cols_to_encode])
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            X_transformed_df = pd.DataFrame(X_transformed, columns=new_columns, index=X_train.index)
            X_train = pd.concat((X_train, X_transformed_df), axis=1)
#            X_train[new_columns] = pd.to_numeric(X_train[new_columns], downcast="integer")
    
            X_transformed = transformer.transform(X_test[cols_to_encode])
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            X_transformed_df = pd.DataFrame(X_transformed, columns=new_columns, index=X_test.index)
            X_test = pd.concat((X_test, X_transformed_df), axis=1)
#            X_test[new_columns] = pd.to_numeric(X_test[new_columns], downcast="integer")
    
            for col in new_columns:
                metadata["data_description"]["feature_types"][col] = "num"

        return X_train, y_train, X_test, metadata
