"""
This unit test checks that we can go from the original training target to the transformed one and back without any differences.
"""
import pandas as pd
import numpy as np
from code_transform_txt_target_train import txt_target_transform, txt_target_inverse_transform

# Load train target
original_train_targets = pd.read_csv('./train_txt_target_map.csv')

# if no text train target, check that transform and inverse transform are the identity
if len(original_train_targets.columns) == 1:
    print("No text training targets for this task.")
    transformed_train_targets = txt_target_transform(original_train_targets)
    # check the transform is the identity
    assert (np.array(transformed_train_targets["id"]) == np.array(original_train_targets["id"])).all(), \
        ("`original_targets` and `txt_target_transform(original_targets)` are exactly the same "
         "but the transform should be the identity.")

    # check the inverse transform is the identity
    inverse_transformed_train_targets = txt_target_inverse_transform(original_train_targets)
    assert (np.array(inverse_transformed_train_targets["id"]) == np.array(original_train_targets["id"])).all(), \
        ("`original_targets` and `txt_target_inverse_transform(original_train_targets)` are not the same "
         "but the inverse transform should be the identity.")

else:
    # apply transform and inverse transform
    columns_without_id = [c for c in original_train_targets.columns if c != "id"]
    transformed_train_targets = txt_target_transform(original_train_targets[columns_without_id])
    inverse_transformed_train_targets = txt_target_inverse_transform(transformed_train_targets)

    # compare both and check they are exactly the same
    assert (np.array(inverse_transformed_train_targets).flatten() == np.array(
        original_train_targets[columns_without_id]).flatten()).all(), \
        ("`original_targets` and `txt_target_inverse_transform(txt_target_transform(original_targets))` are not "
         "the same.\nBut the composition of the transform and the inverse transform should define a bijection.")
