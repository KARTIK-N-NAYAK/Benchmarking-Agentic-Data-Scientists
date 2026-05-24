# unit test
import os
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

if os.path.exists("./root_path_to_agent.txt"):
    with open("./root_path_to_agent.txt", "r") as f:
        agent_root_path = f.read()
    sys.path.insert(0, agent_root_path)
else:
    sys.path.insert(0, os.environ.get("AGENT_PATH", str(Path(__file__).parent.parent.parent.parent)))

from agent.tools.data_map.map_dataset import MapDataset, map_dataset_collate_function

# check that train input and train target maps have the same columns
for train_file, test_file in zip(
        ["./train_img_input_map.csv", "./train_tab_input_map.csv", "./train_txt_input_map.csv"],
        ["./test_img_input_map.csv", "./test_tab_input_map.csv", "./test_txt_input_map.csv"]
):
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_map = pd.read_csv(train_file)
        test_map = pd.read_csv(test_file)
        train_test_cols_intersect = set(train_map.columns).intersection(set(test_map.columns))
        train_test_cols_intersect.remove('id')
        train_test_cols_intersect_err_msg = (
            f"Error: The maps {train_file} and {test_file} do not have the same columns but they should!\n"
            f"Columns of {train_file}: {list(train_map.columns)}\n"
            f"Columns of {test_file}: {list(test_map.columns)}\n"
            f"So the columns in common are:\n{train_test_cols_intersect}\n"
            f"Make sure both dataframes have the same columns!"
        )
        assert len(train_map.columns) == len(test_map.columns), train_test_cols_intersect_err_msg

try:
    _, test_dataset = MapDataset.create_train_test_datasets(
        train_tab_input_map_path=None,
        train_img_input_map_path=None,
        train_txt_input_map_path=None,
        train_tab_target_map_path=None,
        train_img_target_map_path=None,
        train_txt_target_map_path=None,
        test_tab_input_map_path="./test_tab_input_map.csv" if os.path.exists("./test_tab_input_map.csv") else None,
        test_img_input_map_path="./test_img_input_map.csv" if os.path.exists("./test_img_input_map.csv") else None,
        test_txt_input_map_path="./test_txt_input_map.csv" if os.path.exists("./test_txt_input_map.csv") else None,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=map_dataset_collate_function)
    for batch in test_dataloader:
        (
            indices,
            (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
            (tab_targets_batch, img_targets_batch, txt_targets_batch),
        ) = batch
        break
    print("Test batch loaded correctly")
except Exception as e:
    print(f"Error while loading a batch from the test_dataloader:\n{e}")
    raise e
