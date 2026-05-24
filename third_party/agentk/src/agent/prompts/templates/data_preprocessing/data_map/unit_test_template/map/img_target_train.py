# unit test for training image data
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

train_img_target_map = pd.read_csv("./train_img_target_map.csv")

assert len(train_img_target_map) > 0, f"len(train_img_target_map) is 0, but it should be >0."
assert "id" in train_img_target_map.columns, f"column name 'id' not in train_img_target_map.columns"
n_id_appears = sum([1 for c in train_img_target_map.columns if c == 'id'])
assert n_id_appears == 1, (f"column name 'id' appears {n_id_appears} times in train_img_target_map.columns "
                           f"but should only appear once!")


# check that all image paths exist
def init_worker(shared_event):
    global event
    event = shared_event


# check that all image paths exist
def check_file_exists(file_path):
    """Check if a file exists and return the file path if it does."""
    if event.is_set():
        return

    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        if not file_path.exists():
            event.set()
            raise FileNotFoundError(f"File not found: {file_path}")
    else:
        event.set()
        raise ValueError(f"file_path must be a string or Path object but found {file_path} with type {type(file_path)}")
    return file_path


def check_files(file_list):
    """Check a list of files for existence using multiprocessing."""
    with mp.Manager() as manager:
        shared_event = manager.Event()
        with mp.Pool(processes=5, initializer=init_worker, initargs=(shared_event,)) as pool:
            # Use pool.map to apply the check_file_exists function to each file
            results = pool.map(check_file_exists, file_list)
    return results


for c in train_img_target_map.columns:
    if c != "id":
        files_to_check = train_img_target_map[c].tolist()
        check_files(files_to_check)

for c in train_img_target_map.columns:
    if c != "id":
        assert os.path.isfile(train_img_target_map.iloc[0][c]), \
            f"path {train_img_target_map.iloc[0][c]} in column {c} is not a file!"
        assert os.path.exists(train_img_target_map.iloc[0][c]), \
            f"path {train_img_target_map.iloc[0][c]} in column {c} does not exist!"

print("Unit test passed.")
