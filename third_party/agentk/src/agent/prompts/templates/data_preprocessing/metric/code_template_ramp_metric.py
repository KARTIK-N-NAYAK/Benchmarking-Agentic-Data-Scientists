"""
This script creates a function that takes the predicted output `y_pred` and the true output `y_true`
and returns the value of the metric corresponding to the task.
"""
# useful imports
import os
import numpy as np
import pandas as pd
from typing import Iterable, Tuple
from torch import Tensor

# <｜fim▁begin｜>


class Score:

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
           Computes the score on and returns the result.
           Args:
               y_pred: the predicted  target
               y_true: the true target
           Outputs:
               score: the metric score
        """
        # <｜fim▁hole｜>

        return score

    @property
    def minimum(self) -> float:
        # <｜fim▁hole｜>

    @property
    def maximum(self) -> float:
        # <｜fim▁hole｜>

    @property
    def lower_is_better(self) -> bool:
         # <｜fim▁hole｜>

# <｜fim▁end｜>
