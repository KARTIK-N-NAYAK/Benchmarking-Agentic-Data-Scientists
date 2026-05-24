# unit test for test text data
import pandas as pd
import numpy as np

test_txt_input_map = pd.read_csv("./test_txt_input_map.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")

assert len(test_txt_input_map) > 0, f"len(test_txt_input_map) is 0, but it should be >0."
# assert len(test_txt_input_map) >= len(sample_submission), \
#     f"len(test_txt_input_map) is {len(test_txt_input_map)} but it should be at least {len(sample_submission)}."

assert "id" in test_txt_input_map.columns, "Column 'id' needs to be in the 'test_txt_input_map.csv'"
assert len(test_txt_input_map.columns) > 1, ("Error: only 1 column in the test text inputs but needs at least 2:"
                                             " the 'id' and the text test input features.")
for c in test_txt_input_map.columns:
    if c != "id":
        assert isinstance(test_txt_input_map.iloc[0][c], str), \
            f"{test_txt_input_map.iloc[0][c]} in column {c} is not a string!"
if len(test_txt_input_map) == len(sample_submission):
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
    ids_test_txt_map = np.array(test_txt_input_map["id"])
    sorted_submission_ids = np.sort(ids_submission.flat)
    sorted_test_txt_map_ids = np.sort(ids_test_txt_map.flat)
    if not np.array_equal(sorted_submission_ids, sorted_test_txt_map_ids):
        error_str = (f"Values in 'id' column of `test_txt_input_map` and "
                     f"'{submission_id_col_name}' column of `sample_submission` do not match:")
        # find first element that doesn't match
        for j, (c1, c2) in enumerate(zip(sorted_submission_ids, sorted_test_txt_map_ids)):
            if c1 != c2:
                error_str += f"\nat row {j}, {c1} (in `sample_submission`) != {c2} (in `test_txt_input_map`)"
                break
        raise AssertionError(error_str)

print("Unit test passed.")
