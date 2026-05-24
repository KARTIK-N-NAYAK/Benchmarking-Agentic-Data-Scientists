# <｜fim▁begin｜>
import os

import numpy as np
import pandas as pd
import torch
from agent.tools.data_map.map_dataset import MapDataset, map_dataset_collate_function
from torch.utils.data import WeightedRandomSampler, DataLoader


# Complete the function calculate_class_weights without changing the structure of the function outlined
# def calculate_class_weights(target_df: pd.DataFrame, target_columns: list) -> np.ndarray:
#     """
#     Args:
#         target_df: contains targets
#         target_columns: list of target columns available in target_df
#     Returns: numpy array of class weights
#     """
#
#     ...
#
#     # Normalize the weight vector to ensure it sums up to 1
#     weight_vector *= len(weight_vector) / sum(weight_vector)
#     return weight_vector


# <｜fim▁hole｜>
#
# <｜fim▁end｜>


# @NO_MEMORY_START@
if __name__ == "__main__":
    root_path = "@ROOT_DS_DATA_PATH@"

    tab_target_map_path = os.path.join(root_path, "train_tab_target_map.csv")
    train_tab_input_path = os.path.join(root_path, "train_tab_input_map.csv")
    train_img_input_path = os.path.join(root_path, "train_img_input_map.csv")
    train_txt_input_path = os.path.join(root_path, "train_txt_input_map.csv")

    dataset, _ = MapDataset.create_train_test_datasets(
        train_tab_input_map_path=train_tab_input_path if os.path.exists(train_tab_input_path) else None,
        train_img_input_map_path=train_img_input_path if os.path.exists(train_img_input_path) else None,
        train_txt_input_map_path=train_txt_input_path if os.path.exists(train_txt_input_path) else None,
        train_tab_target_map_path=tab_target_map_path if os.path.exists(tab_target_map_path) else None,
    )

    val_proportion = 0.25
    train_dataset, validation_dataset = dataset.split(frac=val_proportion)
    target_df = train_dataset.tab_target_map
    target_df.drop(columns='id', errors='ignore', inplace=True)
    target_columns = target_df.columns.tolist()

    weight_vector = calculate_class_weights(target_df=target_df, target_columns=target_columns)
    weight_vector = torch.from_numpy(weight_vector.astype(np.float32))
    sampler = WeightedRandomSampler(weight_vector, len(weight_vector))

    train_dl = DataLoader(train_dataset, batch_size=256, pin_memory=True, prefetch_factor=2,
                          persistent_workers=True, collate_fn=map_dataset_collate_function, num_workers=20,
                          sampler=sampler)

    print(f"Weight vector code generation is successfully done.")
# @NO_MEMORY_END@
