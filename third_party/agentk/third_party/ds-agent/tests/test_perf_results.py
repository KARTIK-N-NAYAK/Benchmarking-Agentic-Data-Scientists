import numpy as np
import pandas as pd

from ds_agent.results_processing.performance_results import hash_pandas_df


def test_hash_pandas_df() -> int:
    df_ref = pd.DataFrame(columns=["A", "id", "B"], data=np.zeros((20, 3)))
    df_ref["A"] = [True, False] * 10
    df_ref["id"] = np.arange(20, 0, -1)
    df_ref["B"] = ["a", "b"] * 10

    df_pos = pd.DataFrame(columns=["A", "id", "B"], data=np.zeros((20, 3)))
    df_pos["A"] = [0., True] * 10
    df_pos["id"] = np.arange(1, 21)
    df_pos["B"] = ["b", "a"] * 10

    df_neg = pd.DataFrame(columns=["A", "id", "B"], data=np.zeros((20, 3)))
    df_neg["A"] = [True, False] * 10
    df_neg["id"] = np.arange(1, 21)
    df_neg["B"] = ["a", "b"] * 10

    assert hash_pandas_df(df=df_ref) == hash_pandas_df(df=df_pos)
    assert hash_pandas_df(df=df_ref) != hash_pandas_df(df=df_neg)

    return 0
