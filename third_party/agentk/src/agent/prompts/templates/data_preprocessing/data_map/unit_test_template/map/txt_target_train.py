# unit test for training text data
import os
import pandas as pd

train_txt_target_map = pd.read_csv("./train_txt_target_map.csv")

assert len(train_txt_target_map) > 0, f"len(train_txt_target_map) is 0, but it should be >0."
assert "id" in train_txt_target_map.columns, f"column name 'id' not in train_txt_target_map.columns"
n_id_appears = sum([1 for c in train_txt_target_map.columns if c == 'id'])
assert n_id_appears == 1, (f"column name 'id' appears {n_id_appears} times in train_txt_target_map.columns "
                           f"but should only appear once!")
for c in train_txt_target_map.columns:
    if c != "id":
        assert isinstance(train_txt_target_map.iloc[0][c], str), \
            f"path {train_txt_target_map.iloc[0][c]} in column {c} is not a string!"

print("Unit test passed.")
