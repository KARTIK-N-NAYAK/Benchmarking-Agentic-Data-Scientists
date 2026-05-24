# unit test for training tabular data
import pandas as pd

train_tab_input_map = pd.read_csv("./train_tab_input_map.csv")

assert len(train_tab_input_map) > 0, f"len(train_tab_input_map) is 0, but it should be >0."
assert "id" in train_tab_input_map.columns, f"column name 'id' not in train_tab_input_map.columns"
n_id_appears = sum([1 for c in train_tab_input_map.columns if c == 'id'])
assert n_id_appears == 1, (f"column name 'id' appears {n_id_appears} times in train_tab_input_map.columns "
                           f"but should only appear once!")
assert len(train_tab_input_map.columns) > 1, \
    "the only column in `train_tab_input_map` is \'id\' but there should be at least one tabular input column in the map"

print("Unit test passed.")
