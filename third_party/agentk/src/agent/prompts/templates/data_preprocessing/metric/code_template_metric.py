"""
This script creates a function that takes the predicted output `y_pred` and the true output `y_true`
and returns the value of the metric corresponding to the task.
"""
# useful imports
import os
import numpy as np
import pandas as pd
from torch import Tensor

# metric definition
def metric_function(
        y_pred: pd.DataFrame | Tensor,
        y_true: pd.DataFrame | Tensor,
) -> float:
    """
    Computes the metric on (a batch of) inputs and returns the result.
    Args:
        y_pred: the predicted  target
        y_true: the true target
    """
    # <to complete>
    return score
