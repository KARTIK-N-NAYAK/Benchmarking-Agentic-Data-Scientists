# <｜fim▁begin｜>
import numpy as np
import pandas as pd
import os

# --- Design classe: TabRegressionTargetScaler implementing methods __init__, fit, transform and inverse_transform
#
# class TabRegressionTargetScaler:
# 
#    def __init__(self):  # to initialize parameters for scaling (e.g., mean, std for normalization)
#       ...
# 
#    def fit(data: pd.DataFrame) -> None:  # to set some internal states of TabRegressionTargetScaler based on the data
#        ...
# 
#    def transform(data: pd.DataFrame) -> pd.DataFrame:  # rescale the data based on the scaling parameters
#       ...
# 
#    def inverse_transform(transf_data: pd.DataFrame) -> pd.DataFrame:  # revert the scaling applied by the `transform`
#       ...
# <｜fim▁hole｜>

# @NO_MEMORY_START@
if __name__ == "__main__":
    # --- Test the implementation
    # Initialize the scaler

    tab_regression_target_scaler = TabRegressionTargetScaler()

    # Load regression targets to scale
    root_path = "@ROOT_DS_DATA_PATH@"
    train_data_path = os.path.join(root_path, "train_tab_target_map.csv")
    tab_target = pd.read_csv(train_data_path, index_col="id")
    tab_regression_target = tab_target[[c for c in tab_target.columns if c.endswith("_regression")]]

    # fit, transform and inverse transform
    tab_regression_target_scaler.fit(tab_regression_target)
    transformed_tab_regression_target = tab_regression_target_scaler.transform(tab_regression_target)
    inv_transformed_tab_regression_target = tab_regression_target_scaler.inverse_transform(transformed_tab_regression_target)

    # Check if original is equal to inverse transform
    error = np.abs(tab_regression_target - inv_transformed_tab_regression_target).sum(0)

    threshold = 1e-3
    error_message = ""
    for i, colname in enumerate(error.index[error.argsort(-1)]):
        if i > 3:
            break
        if error[colname] > threshold:
            if len(error_message) == 0:
                error_message += f"Mismatch between original and inverse column:\n"
            error_message += f"    - {colname}: {error[colname]:.4f}\n"

    if len(error_message) > 0:
        raise ValueError(error_message)
    # <｜fim▁end｜>

    # @NO_MEMORY_START@
    print(f"Could perform fit and transform, and inverse transform without error.")
# @NO_MEMORY_END@

