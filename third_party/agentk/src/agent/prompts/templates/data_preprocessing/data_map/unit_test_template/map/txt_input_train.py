# unit test for training text data
import pandas as pd

train_txt_input_map = pd.read_csv("./train_txt_input_map.csv")

assert len(train_txt_input_map) > 0, f"len(train_txt_input_map) is 0, but it should be >0."
assert "id" in train_txt_input_map.columns, f"column name 'id' not in train_txt_input_map.columns"
n_id_appears = sum([1 for c in train_txt_input_map.columns if c == 'id'])
assert n_id_appears == 1, (f"column name 'id' appears {n_id_appears} times in train_txt_input_map.columns "
                           f"but should only appear once!")
assert len(train_txt_input_map.columns) > 1, \
    "the only column in `train_txt_input_map` is \'id\' but there should be at least one image text column in the map"
for c in train_txt_input_map.columns:
    if c != "id":
        assert isinstance(train_txt_input_map.iloc[0][c], str), \
            f"{train_txt_input_map.iloc[0][c]} in column {c} is not a string!"

print("Unit test passed.")
