# <｜fim▁begin｜>
import os
import pandas as pd
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from agent.tools.data_map.map_dataset import MapDataset, map_dataset_collate_function

# --- Create CustomTrainImageInputTransform, a composition of torchvision.transforms using torchvision.transforms.Compose
# <｜fim▁hole｜>
# --- [End]

# <｜fim▁end｜>

# @NO_MEMORY_START@
if __name__ == "__main__":
    # Test that the transform works
    root_path = "@ROOT_DS_DATA_PATH@"
    tab_target_map_path = os.path.join(root_path, "train_tab_target_map.csv")
    img_target_map_path = os.path.join(root_path, "train_img_target_map.csv")
    train_dataset, _ = MapDataset.create_train_test_datasets(
        train_img_input_map_path=os.path.join(root_path, "train_img_input_map.csv"),
        train_tab_target_map_path=tab_target_map_path if os.path.exists(tab_target_map_path) else None,
        train_img_target_map_path=img_target_map_path if os.path.exists(img_target_map_path) else None,
        custom_img_train_input_transform=CustomTrainImageInputTransform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=map_dataset_collate_function, num_workers=4
    )

    img_shape = None
    for batch in tqdm(train_dataloader):
        (indices, (_, img_inputs_batch, _), (tab_targets_batch, _, _)) = batch
        img_shape_curr = img_inputs_batch.shape[1:]
        if img_shape != img_shape_curr and img_shape is not None:
            raise RuntimeError(
                f"Multiple shapes detected! After applying CustomTrainImageInputTransform, some images have shapes "
                f"{img_shape} and {img_shape_curr} "
                f"but should have the same shape for all images after the transform. "
                f"(Note that the first dimension is the batch size {train_dataloader.batch_size}).\n"
                f"Hint: if needed add a lambda transform that ensures the number of channels is, e.g. 3 in this example:\n"
                f"`T.Lambda(lambda x: x.repeat([3 if (i - x.ndim == -3 and d == 1) else 1 for i, d in enumerate(x.shape)])),`"
                f"\nand\n`T.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x)`"
            )
        img_shape = img_shape_curr
        if os.getenv('AGENT_DEBUG', False):
            break

    total_pixel_limit = 16 * 10e9
    if img_shape[-1] * img_shape[-2] * len(train_dataset) > total_pixel_limit:  # Heuristic computation limit on total pixels
        max_image_pixels = total_pixel_limit / len(train_dataset)
        scaling_ratio = torch.sqrt(max_image_pixels / (img_shape[-1] * img_shape[-2]))
        max_size = (scaling_ratio * img_shape[-2:]).int()
        raise RuntimeError(
            "Transform produces images that will be too big to finish a training epoch in reasonable time. "
            f"Please use a transform that results in smaller images (i.e. smaller than {max_size})"
        )


    print(f"Transformed image tensor shape: {img_shape}")
# @NO_MEMORY_END@
