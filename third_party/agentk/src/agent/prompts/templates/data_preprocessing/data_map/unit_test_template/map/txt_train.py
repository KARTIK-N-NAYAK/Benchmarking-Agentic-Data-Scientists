# unit test for training text data
import pandas as pd
import numpy as np

try:
    train_txt_input_map = pd.read_csv("./train_txt_input_map.csv")
except FileNotFoundError:
    train_txt_input_map = None
try:
    train_txt_target_map = pd.read_csv("./train_txt_target_map.csv")
except FileNotFoundError:
    train_txt_target_map = None

if train_txt_input_map is None and train_txt_target_map is None:
    print(f"Both maps are None! If Text inputs and Text targets are not needed for the task, this is normal.")
else:
    if train_txt_input_map is None:
        train_txt_input_map = train_txt_target_map[["id"]]

    if train_txt_target_map is None:
        train_txt_target_map = train_txt_input_map[["id"]]

    # check sizes
    assert len(train_txt_input_map) > 0, f"len(train_txt_input_map) is 0, but it should be >0."
    assert len(train_txt_target_map) > 0, f"len(train_txt_target_map) is 0, but it should be >0."
    assert len(train_txt_input_map) == len(train_txt_target_map), \
        (f"Error: len(train_txt_input_map) is {len(train_txt_input_map)} and len(train_txt_target_map) is "
         f"{len(train_txt_target_map)} but they should be equal.")

    # check "id" columns are the same
    assert (np.array(train_txt_input_map["id"]).flatten() == np.array(train_txt_target_map["id"]).flatten()).all(), \
        "Error: Values in 'id' column of `train_txt_input_map` and `train_txt_target_map` do not match."

print("Unit test passed.")
