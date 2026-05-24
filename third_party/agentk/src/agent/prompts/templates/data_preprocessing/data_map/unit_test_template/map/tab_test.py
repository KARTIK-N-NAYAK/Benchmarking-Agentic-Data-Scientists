import os

import pandas as pd

# check that all columns in the map correspond to the ones in sample_submission file, if it exists
if (
        os.path.exists("./train_tab_target_map.csv")
        and os.path.exists("./test_tab_input_map.csv")
        and os.path.exists("./data/sample_submission.csv")
):
    train_tab_target_map = pd.read_csv("./train_tab_target_map.csv")
    test_tab_input_map = pd.read_csv("./test_tab_input_map.csv")
    sample_submission = pd.read_csv("./data/sample_submission.csv")

    submission_target_names = set([c for c in sample_submission.columns if c not in test_tab_input_map.columns])
    train_tab_target_names = set([c.split("_")[0] for c in train_tab_target_map.columns if c != "id"])

    _in_submission_only = submission_target_names.difference(train_tab_target_names)
    _in_train_targets_only = train_tab_target_names.difference(submission_target_names)
    if len(_in_submission_only) > 0:
        raise ValueError(
            f"Columns {_in_submission_only} are in sample_submission.csv but not in train_tab_target_map.csv\n"
            f"However, the target columns of train_tab_target_map.csv and sample_submission.csv should be the same."
        )
    if len(_in_train_targets_only) > 0:
        raise ValueError(
            f"Columns {_in_train_targets_only} are in train_tab_target_map.csv but not in sample_submission.csv\n"
            f"However, the target columns of train_tab_target_map.csv and sample_submission.csv should be the same."
        )

print("Unit test passed.")
