# unit test
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

if os.path.exists("./root_path_to_agent.txt"):
    with open("./root_path_to_agent.txt", "r") as f:
        agent_root_path = f.read()
    sys.path.insert(0, agent_root_path)
else:
    sys.path.insert(0, os.environ.get("AGENT_PATH", str(Path(__file__).parent.parent.parent.parent)))

train_input_maps_column_names = set()
train_input_maps_num_rows = {}
train_input_maps_rows = {}
for file in ["train_img_input_map.csv", "train_tab_input_map.csv", "train_txt_input_map.csv"]:
    if os.path.exists(file):
        _input_map = pd.read_csv(file)
        train_input_maps_column_names = train_input_maps_column_names.union(_input_map.columns)
        if _input_map.shape[0] not in train_input_maps_num_rows:
            train_input_maps_num_rows[_input_map.shape[0]] = [file]
        else:
            train_input_maps_num_rows[_input_map.shape[0]].append(file)
        train_input_maps_rows[file] = set(_input_map['id'].values)

train_target_maps_column_names = set()
train_target_maps_num_rows = {}
train_target_maps_rows = {}
for file in ["train_img_target_map.csv", "train_tab_target_map.csv", "train_txt_target_map.csv"]:
    if os.path.exists(file):
        _target_map = pd.read_csv(file)
        train_target_maps_column_names = train_target_maps_column_names.union(_target_map.columns)
        if _target_map.shape[0] not in train_target_maps_num_rows:
            train_target_maps_num_rows[_target_map.shape[0]] = [file]
        else:
            train_target_maps_num_rows[_target_map.shape[0]].append(file)
        train_target_maps_rows[file] = set(_target_map['id'].values)

# check all input maps have same id column
if len(train_input_maps_rows) > 1:
    for file1, file2 in combinations(train_input_maps_rows.keys(), 2):
        diff_ids_12 = train_input_maps_rows[file1] - train_input_maps_rows[file2]
        diff_ids_21 = train_input_maps_rows[file2] - train_input_maps_rows[file1]
        assert len(diff_ids_12) == 0, f"These ids are in {file1} but not in {file2}: {list(diff_ids_12)[:10]}..."
        assert len(diff_ids_21) == 0, f"These ids are in {file2} but not in {file1}: {list(diff_ids_21)[:10]}..."

# check all target maps have same id column
if len(train_target_maps_rows) > 1:
    for file1, file2 in combinations(train_target_maps_rows.keys(), 2):
        diff_ids_12 = train_target_maps_rows[file1] - train_target_maps_rows[file2]
        diff_ids_21 = train_target_maps_rows[file2] - train_target_maps_rows[file1]
        assert len(diff_ids_12) == 0, f"These ids are in {file1} but not in {file2}: {list(diff_ids_12)[:10]}..."
        assert len(diff_ids_21) == 0, f"These ids are in {file2} but not in {file1}: {list(diff_ids_21)[:10]}..."

# check that apart from the "id" column, the other columns are not called the same between input and target maps
input_target_cols_intersect = train_input_maps_column_names.intersection(train_target_maps_column_names)
assert 'id' in input_target_cols_intersect, f"Column 'id' should be present in both input and target maps!"
input_target_cols_intersect.remove('id')
input_target_cols_intersect_err_msg = (f"Error: There are {len(input_target_cols_intersect)} columns in common between "
                                       f"input and target maps but they should belong to one or the other:\n"
                                       f"[{','.join([c for c in input_target_cols_intersect])}]")
assert len(input_target_cols_intersect) == 0, input_target_cols_intersect_err_msg

# check that the id column for input and target maps has the same number of rows
rows = set(train_input_maps_num_rows).union(train_target_maps_num_rows)
n_rows_error_str = None
if len(rows) > 1:
    n_rows_error_str = "Make sure all train input and target maps have the same number of rows!\n"
    for row in rows:
        if row in train_input_maps_num_rows:
            n_rows_error_str += f'- {train_input_maps_num_rows[row]} has {row} rows\n'
        if row in train_target_maps_num_rows:
            n_rows_error_str += f'- {train_target_maps_num_rows[row]} has {row} rows\n'

# check that the id column is the same for input and target maps:
# print the rows that are present in input maps but not in target maps and vice versa
id_error_str = ""
input_ids = train_input_maps_rows[list(train_input_maps_rows.keys())[0]]
target_ids = train_target_maps_rows[list(train_target_maps_rows.keys())[0]]
diff_input_target = input_ids - target_ids
diff_target_input = target_ids - input_ids
if len(diff_input_target) > 0 or len(diff_target_input) > 0:
    id_error_str = "Make sure all train input and target maps have exactly the same values in their 'id' column!\n"
if len(diff_input_target) > 0:
    id_error_str += f"\nThese ids are in input maps but not in target maps: {', '.join(list(map(str, diff_input_target))[:10])}...\n"
if len(diff_target_input) > 0:
    id_error_str += f"\nThese ids are in target maps but not in input maps: {', '.join(list(map(str, diff_target_input))[:10])}...\n"

error_str = ""
if n_rows_error_str is not None and len(n_rows_error_str) > 0:
    error_str += n_rows_error_str
if id_error_str is not None and len(id_error_str) > 0:
    error_str += id_error_str
if len(error_str) > 0:
    raise AssertionError(error_str)

print("Unit test passed.")
