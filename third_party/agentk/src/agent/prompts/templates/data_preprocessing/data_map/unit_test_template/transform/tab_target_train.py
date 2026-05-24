"""
This unit test checks that we can go from the original training target to the transformed one and back without any differences.
"""
import torch
import pandas as pd
from code_transform_tab_target_train import tab_target_transform, tab_target_inverse_transform, enc, \
        class_names_columns_classification

# Load train target
original_train_targets = pd.read_csv('./train_tab_target_map.csv')

# apply transform and inverse transform
transformed_train_targets = tab_target_transform(original_train_targets)
num_target_columns = transformed_train_targets.shape[1] - 1  # exclude "id" column
random_target_values = torch.randn((transformed_train_targets.shape[0], num_target_columns))

if enc is not None:
    target_values = []
    groups_label = enc.categories_
    idx_start = 0
    for group in groups_label:
        group_target = random_target_values[:, idx_start:idx_start + len(group)].softmax(dim=-1)
        target_values.append(group_target)
        idx_start += len(group)
    target_values = torch.cat(target_values, dim=-1)
    inverse_transformed_train_targets = tab_target_inverse_transform(
        target_values=target_values.cpu().numpy(), ids=original_train_targets["id"].values
    )
else:
    inverse_transformed_train_targets = tab_target_inverse_transform(
        target_values=random_target_values.cpu().numpy(), ids=original_train_targets["id"].values
    )
inverse_transformed_train_targets.to_csv('df_tab_target_inv_transform.csv', index=False)
