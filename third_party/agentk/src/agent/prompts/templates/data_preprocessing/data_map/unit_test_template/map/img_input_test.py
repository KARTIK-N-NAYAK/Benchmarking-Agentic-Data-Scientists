# unit test for test text data
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

test_img_input_map = pd.read_csv("./test_img_input_map.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")

assert len(test_img_input_map) > 0, f"len(test_img_input_map) is 0, but it should be >0."
# assert len(test_img_input_map) >= len(sample_submission), \
#     f"len(test_img_input_map) is {len(test_img_input_map)} but it should be at least {len(sample_submission)}."

assert "id" in test_img_input_map.columns, "Column 'id' needs to be in the 'test_img_input_map.csv'"
assert len(test_img_input_map.columns) > 1, ("Error: only 1 column in the test text inputs but needs at least 2:"
                                             " the 'id' and the text test input features.")


# check that all image paths exist
def init_worker(shared_event):
    global event
    event = shared_event


# check that all image paths exist
def check_abspath_file_exists(file_path):
    """
    Check if file_path is absolute and if the file at that address exists.
    If both are true, return the file path.
    """
    if event.is_set():
        return

    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # check path is absolute and not relative
        if not file_path.is_absolute():
            event.set()
            raise ValueError(f"Paths must be absolute but found at least one relative path: {file_path}")

        # check file exists
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
            results = pool.map(check_abspath_file_exists, file_list)
    return results


for c in test_img_input_map.columns:
    if c != "id":
        files_to_check = test_img_input_map[c].tolist()
        check_files(files_to_check)

if len(test_img_input_map) == len(sample_submission):
    # ---------- WARNING ----------
    # in general, we cannot know a priori if the test ids and the sample_submission ids should be the same, but
    # we consider it highly unlikely.
    # Example:
    # consider a competition where there are N test inputs with a set of ids IDs_1
    # and consider that this competition also has a sample_submission with N test inputs and a set of ids IDs_2
    # In this case, we assume that IDs_1 == IDs_2 and the present unit test (assertion below) verifies that
    # but in general it could be that IDs_1 != IDs_2 and the true ids we need in the test map are IDs_1.
    # However, we assume that if there are the same number of elements in the test inputs and in the sample_submission,
    # it is because they are the same inputs so their ids must match
    submission_id_col_name = sample_submission.columns[0]
    ids_submission = np.array(sample_submission[submission_id_col_name])
    ids_test_img_map = np.array(test_img_input_map["id"])
    sorted_submission_ids = np.sort(ids_submission.flat)
    sorted_test_img_map_ids = np.sort(ids_test_img_map.flat)
    if not np.array_equal(sorted_submission_ids, sorted_test_img_map_ids):
        error_str = (f"Values in 'id' column of `test_img_input_map` and "
                     f"'{submission_id_col_name}' column of `sample_submission` do not match:")
        # find first element that doesn't match
        for j, (c1, c2) in enumerate(zip(sorted_submission_ids, sorted_test_img_map_ids)):
            if c1 != c2:
                error_str += f"\nat row {j}, {c1} (in `sample_submission`) != {c2} (in `test_img_input_map`)"
                break
        raise AssertionError(error_str)

print("Unit test passed.")
