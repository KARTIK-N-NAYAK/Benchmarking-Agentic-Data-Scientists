# unit test for training tabular data
import pandas as pd

tab_name = "train_tab_target_map.csv"

train_tab_target_map = pd.read_csv(f"./{tab_name}")

# check that all columns correspond to a regression or a classification
columns = [column for column in train_tab_target_map.columns if column != "id"]

endings = ["_regression", "_classification"]
for column in columns:
    invalid_end = True
    for ending in endings:
        if column.endswith(ending):
            invalid_end = False
            break
    if invalid_end:
        raise ValueError(
            f"Column name `{column}` in {tab_name} is not valid as id does not end with any of {' / '.join(endings)} "
        )

assert len(train_tab_target_map) > 0, f"len(train_tab_target_map) is 0, but it should be >0."
assert "id" in train_tab_target_map.columns, f"column name 'id' not in train_tab_target_map.columns"
n_id_appears = sum([1 for c in train_tab_target_map.columns if c == 'id'])
assert n_id_appears == 1, (f"column name 'id' appears {n_id_appears} times in train_tab_target_map.columns "
                           f"but should only appear once!")
assert not train_tab_target_map[[c for c in train_tab_target_map.columns if c != "id"]].isna().any().all(), \
    "NaN detected in train_tab_target_map"

print("Unit test passed.")
