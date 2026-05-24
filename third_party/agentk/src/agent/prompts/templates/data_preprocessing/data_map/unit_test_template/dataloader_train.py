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

from agent.tools.data_map.map_dataset import MapDataset, map_dataset_collate_function, Identity, \
    DefaultImageInputTransform

try:
    from code_transform_tab_target_train import tab_target_transform, tab_target_inverse_transform
except ImportError as e:
    tab_target_transform = Identity()
    tab_target_inverse_transform = Identity()

try:
    from code_transform_img_target_train import img_target_transform, img_target_inverse_transform
except ImportError as e:
    img_target_transform = Identity()
    img_target_inverse_transform = Identity()

try:
    from code_transform_txt_target_train import txt_target_transform, txt_target_inverse_transform
except ImportError as e:
    txt_target_transform = Identity()
    txt_target_inverse_transform = Identity()

try:
    train_dataset, _ = MapDataset.create_train_test_datasets(
        train_tab_input_map_path="./train_tab_input_map.csv" if os.path.exists("./train_tab_input_map.csv") else None,
        train_img_input_map_path="./train_img_input_map.csv" if os.path.exists("./train_img_input_map.csv") else None,
        train_txt_input_map_path="./train_txt_input_map.csv" if os.path.exists("./train_txt_input_map.csv") else None,
        train_tab_target_map_path="./train_tab_target_map.csv" if os.path.exists(
            "./train_tab_target_map.csv") else None,
        train_img_target_map_path="./train_img_target_map.csv" if os.path.exists(
            "./train_img_target_map.csv") else None,
        train_txt_target_map_path="./train_txt_target_map.csv" if os.path.exists(
            "./train_txt_target_map.csv") else None,
        test_tab_input_map_path=None,
        test_img_input_map_path=None,
        test_txt_input_map_path=None,
        tab_input_transform=Identity(),
        img_input_transform=DefaultImageInputTransform,
        txt_input_transform=Identity(),
        tab_target_transform=tab_target_transform,
        img_target_transform=img_target_transform,
        txt_target_transform=txt_target_transform,
        tab_target_inverse_transform=tab_target_inverse_transform,
        img_target_inverse_transform=img_target_inverse_transform,
        txt_target_inverse_transform=txt_target_inverse_transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=map_dataset_collate_function)
    for batch in train_dataloader:
        (
            indices,
            (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
            (tab_targets_batch, img_targets_batch, txt_targets_batch),
        ) = batch
        break
    print("Training batch loaded correctly")
except Exception as e:
    print(f"Error while loading a batch from the train_dataloader:\n{e}")
    raise e

# check that all images can be loaded correctly, otherwise, remove these indices altogether from maps
batch_size = max(32, len(train_dataloader) // 100)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, collate_fn=map_dataset_collate_function
)
failed_batches = []
iterable_train_loader = iter(train_dataloader)
for batch_idx in range(len(train_dataloader) + 1):
    try:
        batch = next(iterable_train_loader)
        (
            indices,
            (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
            (tab_targets_batch, img_targets_batch, txt_targets_batch),
        ) = batch
    except OSError as e:
        failed_batches.extend(list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size)))
    except StopIteration as e:
        break

filtered_failed_indices = []
for idx in failed_batches:
    try:
        element = train_dataset.__getitem__(idx)
    except OSError as e:
        filtered_failed_indices.append(idx)
if len(filtered_failed_indices) > 0:
    print(f"Dropping indices {filtered_failed_indices} from all training maps as there is a loading issue")

# remove filtered indices from all maps
train_tab_input_map = pd.read_csv("./train_tab_input_map.csv") if os.path.exists("./train_tab_input_map.csv") else None
train_img_input_map = pd.read_csv("./train_img_input_map.csv") if os.path.exists("./train_img_input_map.csv") else None
train_txt_input_map = pd.read_csv("./train_txt_input_map.csv") if os.path.exists("./train_txt_input_map.csv") else None
train_tab_target_map = pd.read_csv("./train_tab_target_map.csv") if os.path.exists(
    "./train_tab_target_map.csv") else None
train_img_target_map = pd.read_csv("./train_img_target_map.csv") if os.path.exists(
    "./train_img_target_map.csv") else None
train_txt_target_map = pd.read_csv("./train_txt_target_map.csv") if os.path.exists(
    "./train_txt_target_map.csv") else None
for path in [
    "./train_tab_input_map.csv",
    "./train_img_input_map.csv",
    "./train_txt_input_map.csv",
    "./train_tab_target_map.csv",
    "./train_img_target_map.csv",
    "./train_txt_target_map.csv",
]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.drop(filtered_failed_indices, axis=0)
        df.to_csv(path, index=False)
