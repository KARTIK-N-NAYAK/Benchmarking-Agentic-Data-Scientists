"""
This script creates a function that takes the transformed target
and returns a formatted DataFrame following sample_submission format
"""
import os.path

# useful imports

import pandas as pd

try:
    from code_submission_format import df_to_submission_format
except ImportError as e:
    print(e)
    raise e

try:
    if os.path.exists('df_tab_target_inv_transform.csv'):
        sample_submission = pd.read_csv('data/sample_submission.csv', index_col=False)
        sample_submission.drop(
            sample_submission.columns[sample_submission.columns.str.contains('unnamed', case=False)], axis=1, inplace=True
        )

        # formatted_submission = submission_format(inverse_transform(targets))
        inv_transform_df = pd.read_csv('df_tab_target_inv_transform.csv', index_col=False)

        formatted_submission = df_to_submission_format(inv_transform_df)
        formatted_submission.drop(
            formatted_submission.columns[formatted_submission.columns.str.contains('unnamed', case=False)],
            axis=1,
            inplace=True
        )

        # check all columns are filled, or at least not all filled with NaN
        for c in formatted_submission.columns:
            if formatted_submission[c].isna().sum() == len(formatted_submission):
                raise ValueError(f"Column {c} is empty! It is only filled with NaN, make sure "
                                 f"your submission format function is correct.")

        # check that df_to_submission_format() did not alter the number of rows of its input, inv_transform_df
        assert formatted_submission.shape[0] == inv_transform_df.shape[0], \
            (f"The dataframe `df_tab_target_inv_transform.csv` has {inv_transform_df.shape[0]} rows, but "
             f"after applying the function `df_to_submission_format` to it we get a dataframe that has "
             f"{formatted_submission.shape[0]} rows. However both should have the same number of rows.")

        n_cols_sample_sub = sample_submission.shape[1]
        n_cols_formatted_sub = formatted_submission.shape[1]
        different_sample_sub_cols = set(sample_submission.columns).difference(set(formatted_submission.columns))
        different_formatted_sub_cols = set(formatted_submission.columns).difference(set(sample_submission.columns))

        # Check that sample_submission and formatted_submission have same number of columns
        assert len(sample_submission.columns) == len(formatted_submission.columns), \
            (f"sample_submission has {n_cols_sample_sub} target columns and "
             f"your formatted submission has {n_cols_formatted_sub} columns, "
             f"but they should have the same number of columns.\n"
             f"`sample_submission` has the following ({n_cols_sample_sub}) columns: {sample_submission.columns}\n"
             f"`df_to_submission_format(df_tab_target_inv_transform.csv)` "
             f"has the following ({n_cols_formatted_sub}) columns: {formatted_submission.columns}\n")

        # Check that sample_submission and formatted_submission have same column names in same order
        assert (sample_submission.columns == formatted_submission.columns).all(), \
            (f"Not all columns names match between `sample_submission.csv` and `df_tab_target_inv_transform.csv` "
             f"after applying the format submission function.\n"
             f"Columns in `sample_submission.csv` that are not in common: {different_sample_sub_cols}\n"
             f"Columns in `df_tab_target_inv_transform.csv` (after applying `df_to_submission_format()`) "
             f" that are not in common: {different_formatted_sub_cols}\n"
             f"All columns in both should match.")

except Exception as e:
    print(e)
    raise e
