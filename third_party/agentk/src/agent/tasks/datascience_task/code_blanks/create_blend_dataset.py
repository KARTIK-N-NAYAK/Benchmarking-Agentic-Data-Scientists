import os
import pathlib
import pickle

import torch
import torch.optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

try:
    from map_dataset import MapDataset, map_dataset_collate_function, Identity
except ImportError:
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    print(f"map_dataset should be in {current_dir}")
    raise

from blend_params import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, NUM_WORKERS

from solve_common_utils import (
    tab_fe_preprocess,
    tab_target_transform,
    tab_target_inverse_transform,
    custom_tab_regression_scaler,
    img_target_transform,
    img_target_inverse_transform,
    txt_target_transform,
    txt_target_inverse_transform,
    CustomTestImageInputTransform
)
from train_utils import SubmissionModel

torch.manual_seed(123)
random_state = 123


# --- Build dataset
def get_optional_path(path: str) -> str | None:
    """ Return path if it exists, else return None """
    if os.path.exists(path):
        return path


train_tab_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_tab_input_map.csv")
train_img_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_img_input_map.csv")
train_txt_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_txt_input_map.csv")
train_tab_target_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_tab_target_map.csv")
train_img_target_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_img_target_map.csv")
train_txt_target_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/train_txt_target_map.csv")

test_tab_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/test_tab_input_map.csv")
test_img_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/test_img_input_map.csv")
test_txt_input_map_path = get_optional_path("@ROOT_DS_DATA_PATH@/test_txt_input_map.csv")

train_dataset, test_dataset = MapDataset.create_train_test_datasets(
    train_tab_input_map_path=train_tab_input_map_path,
    train_img_input_map_path=train_img_input_map_path,
    train_txt_input_map_path=train_txt_input_map_path,
    train_tab_target_map_path=train_tab_target_map_path,
    train_img_target_map_path=train_img_target_map_path,
    train_txt_target_map_path=train_txt_target_map_path,
    test_tab_input_map_path=test_tab_input_map_path,
    test_img_input_map_path=test_img_input_map_path,
    test_txt_input_map_path=test_txt_input_map_path,
    tab_input_transform=Identity(),
    img_input_transform=Identity(),
    txt_input_transform=Identity(),
    tab_target_transform=tab_target_transform,
    img_target_transform=img_target_transform,
    txt_target_transform=txt_target_transform,
    tab_target_inverse_transform=tab_target_inverse_transform,
    img_target_inverse_transform=img_target_inverse_transform,
    txt_target_inverse_transform=txt_target_inverse_transform,
    custom_tab_regression_scaler=custom_tab_regression_scaler,
    custom_img_train_input_transform=CustomTestImageInputTransform,
    custom_img_test_input_transform=CustomTestImageInputTransform,
    tab_fe=tab_fe_preprocess.preprocess if tab_fe_preprocess else None
)

val_proportion = 0.25
train_dataset, validation_dataset = train_dataset.split(frac=val_proportion, random_state=random_state)


def get_data_loaders() -> tuple[DataLoader, DataLoader]:
    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(random_state))
    test_sampler = RandomSampler(test_dataset, generator=torch.Generator().manual_seed(random_state))

    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler,
                          collate_fn=map_dataset_collate_function, num_workers=NUM_WORKERS)

    test_dl = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, sampler=test_sampler,
                         collate_fn=map_dataset_collate_function, num_workers=NUM_WORKERS)

    return train_dl, test_dl


train_dataloader, test_dataloader = get_data_loaders()


def get_mlp_dataloaders_train() -> TensorDataset:
    model = SubmissionModel.load_from_checkpoint(checkpoint_path=f"./best_model.ckpt")
    blend_inputs, blend_targets, ind = model.get_blend_submissions(dataloader=train_dataloader, blend_embeddings=False)
    train_x = torch.cat(blend_inputs, dim=0).to(torch.device('cpu'))
    train_y = torch.cat(blend_targets, dim=0).to(torch.device('cpu'))
    train_dataset = TensorDataset(train_x, train_y)
    return train_dataset


def get_mlp_dataloaders_test() -> tuple[TensorDataset, list[list[int]]]:
    indices = []
    model = SubmissionModel.load_from_checkpoint(checkpoint_path=f"./best_model.ckpt")
    blend_inputs, _, ind = model.get_blend_submissions(dataloader=test_dataloader, blend_embeddings=False)
    test_x = torch.cat(blend_inputs, dim=0).to(torch.device('cpu'))
    indices.append(ind)
    test_dataset = TensorDataset(test_x)
    return test_dataset, indices


if __name__ == "__main__":
    train_dataset = get_mlp_dataloaders_train()
    torch.save(train_dataset, './train_probabilities.pt')
    test_dataset, indices = get_mlp_dataloaders_test()
    torch.save(test_dataset, './test_probabilities.pt')
    with open('./indices.pkl', 'wb') as f:
        pickle.dump(indices, f)
