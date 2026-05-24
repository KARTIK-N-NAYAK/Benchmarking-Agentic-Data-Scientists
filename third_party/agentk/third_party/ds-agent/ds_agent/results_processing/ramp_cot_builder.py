import glob
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ds_agent.utils import ListableEnum


class RampColumnNames(ListableEnum, Enum):
    CONTRIBUTIVITY = "contributivity"
    HYP_SUB = "hyperopt_submission"


class MissingRampSummaryError(RuntimeError):
    pass


def _get_summary_df(root_dir: Path) -> pd.DataFrame:
    """
    Traverse each top-level child of `root_dir` named without underscores, locate all
    `summary.csv` files in their `hyperopt_output` subdirectories, and return them
    as a single concatenated DataFrame.

    Args:
       root_dir: Path to the root directory of ramp kit

    Returns:
       pd.DataFrame: Concatenated contents of all found `summary.csv` files (indices reset)
    """
    summary_dfs = []
    for p in glob.iglob(str(root_dir / "submissions/*/")):
        if "_hyperopt" in p[:-1]:
            continue
        csv_path = Path(p) / "hyperopt_output" / "summary.csv"
        if not csv_path.exists():
            continue
        summary_dfs.append(pd.read_csv(csv_path, index_col=0))

    if len(summary_dfs) > 0:
        return pd.concat(summary_dfs, ignore_index=True)
    else:
        raise MissingRampSummaryError(f"No summary.csv files found in: {root_dir}")


def _get_blend_actions_df(ramp_kit_dir: Path):
    """
    Retrieve the final 'blend' action across folds 0–29 for a given RAMP kit.

    Args:
       ramp_kit_dir: Path to the RAMP kit directory containing fold result subfolders.

    Returns:
       A DataFrame (or similar object) representing the last 'blend' action for each fold.
    """
    import ramphy.ramp_setup as rs

    last_blend_action = rs.scripts.orchestration.last_action(
        ramp_kit_dir=ramp_kit_dir, name="blend", fold_idxs=range(30)
    )
    return last_blend_action


# Function to get the last blend actions
def _get_hyperopt_cv_df(ramp_kit_dir: Path, k: int):
    """
    Aggregate hyperopt CV results, merge with blend contributivity for top-k submissions,
    and return a Markdown-formatted summary of their key metrics.

    Args:
       ramp_kit_dir: Path to the RAMP kit directory for retrieving blend actions.
       k: Number of top submissions (by contributivity) to include.

    Returns:
       A Markdown-formatted string summarizing key CV metrics and contributivities
       for the top-k submissions, or a message if no summary exists.
    """
    base_model1 = 'lgbm'
    base_model2 = 'xgboost'
    base_model3 = 'catboost'

    final_summary_df = _get_summary_df(root_dir=ramp_kit_dir)

    contrib_colname = RampColumnNames.CONTRIBUTIVITY.value

    last_blend_action = _get_blend_actions_df(ramp_kit_dir=ramp_kit_dir)

    columns = [col for col in final_summary_df.columns if not col.endswith("_i") and not col.endswith("_drop")]
    final_summary_df = final_summary_df[columns]

    aggregator = lambda x: x.mean() if x.dtype.kind in ["b", "i", "f", "c"] else x.iloc[0]
    hyperopt_cv_df = final_summary_df.groupby(RampColumnNames.HYP_SUB.value).agg(aggregator)
    hyperopt_cv_df[contrib_colname] = 0

    for s, c in tqdm(last_blend_action.contributivities.items(), desc="last_blend_action", leave=False):
        if s.startswith(base_model1) or s.startswith(base_model2) or s.startswith(base_model3):
            hyperopt_cv_df.loc[s, contrib_colname] = c

    new_hyperopt_cv_df = hyperopt_cv_df.sort_values(RampColumnNames.CONTRIBUTIVITY.value, ascending=False)

    new_hyperopt_cv_df = new_hyperopt_cv_df.dropna(axis=1, how='all')

    non_zero_count = (new_hyperopt_cv_df[contrib_colname] > 0).sum()
    k = min(k, non_zero_count)

    top_k_entries = new_hyperopt_cv_df.head(k)
    top_k_entries = top_k_entries.reset_index()
    top_k_entries[RampColumnNames.HYP_SUB.value] = top_k_entries[RampColumnNames.HYP_SUB.value].str.split('_').str[0]
    top_k_entries[contrib_colname] = top_k_entries[contrib_colname] / 10

    text_result = ""
    for model_name, group in top_k_entries.groupby(RampColumnNames.HYP_SUB.value):
        # Remove the 'hyperopt_submission' column
        group = group.drop(columns=[RampColumnNames.HYP_SUB.value])

        # Remove columns where all values are NaN
        group = group.dropna(axis=1, how='all')

        group = group.loc[:, group.columns.str.startswith('hyper') | group.columns.str.startswith(contrib_colname)]

        # Check for columns with the same value in all rows
        columns_to_remove = []
        for col in group.columns:
            if col != RampColumnNames.CONTRIBUTIVITY.value and group[col].nunique() == 1:
                columns_to_remove.append((col, group[col].iloc[0]))

        # Remove columns with the same value in all rows
        group = group.drop(columns=[col[0] for col in columns_to_remove])

        # Add column name and value to text result 1e-05
        text_result += f"### {model_name}\n"
        if len(columns_to_remove) > 0:
            text_result += "\n**Shared hyperparameters**:\n"
            for col_name, value in columns_to_remove:
                if isinstance(value, float):
                    value = f"{value:.2g}"
                assert col_name.startswith("hyper_"), col_name
                text_result += f"- {col_name[len('hyper_'):]}: {value}\n"
            text_result += "\n"

        # Round all floating point columns
        for col in group.columns:
            if group[col].dtype.kind in ['f', 'i']:  # Check if column is float
                group[col] = group[col].map(lambda x: f"{x:.2g}")

        # Convert dataframe to text and add to the result
        group = group.rename(
            columns={RampColumnNames.CONTRIBUTIVITY.value: f"{RampColumnNames.CONTRIBUTIVITY.value} (%)"}
        )
        text_result += group.to_markdown(index=False) + "\n\n"

    return text_result


def build_ramp_cot(ramp_kit_dir: Path) -> str:
    """

    Args:
        ramp_kit_dir: root directory of ramp run

    Returns:

    """
    text_to_prepend = (
        "We test multiple ensemble models and combined the best-performing ones to make final predictions. "
        "Each model has a different contribution based on its performance. Below are some models "
        "and their contributions in the final prediction, along with their hyperparameters."
        " Use this information to come up with a solution for the competition. "
        "There may be better models and sets of hyperparameter that can be used so you are free "
        "to explore and come up with a better solution"
    )

    contributions = _get_hyperopt_cv_df(ramp_kit_dir=ramp_kit_dir, k=10)

    return text_to_prepend + "\n\n" + contributions
