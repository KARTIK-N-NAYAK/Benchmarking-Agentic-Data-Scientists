from __future__ import annotations

import os
import copy
import dataclasses
import json
import os.path
import re
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from ds_agent.utils import SubmissionFormatError

# This is used also in third_party/img_embed_code_template.py,
# so if this changes, remember to change also the unit test there.
DefaultImageInputTransform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.RGB(),
        v2.ToTensor(),
    ]
)

DefaultImageTargetTransform: v2.Compose = v2.Compose([v2.ToTensor()])


class ImageExtensionType(str, Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    TIF = "tif"

    @classmethod
    def list(cls) -> list[str]:
        return [k for k in cls]


class Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


def open_img_and_transform(transform, p):
    return transform(Image.open(p), p)[0] if isinstance(transform, v2.Transform) else transform(Image.open(p))


def translate_benchmark_path(output_dir, p):
    # We must save as png here to avoid saving-related compression artefacts
    path_from_root = Path(p).relative_to("/")
    return output_dir / path_from_root.with_suffix(".png")


def transform_and_save(transform, output_dir, p) -> str:
    save_path = translate_benchmark_path(output_dir, p)
    if not save_path.exists():
        transformed_img = open_img_and_transform(transform, p)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        transformed_img.save(save_path)
    return str(save_path)


def split_img_transform(img_transform: T.Compose) -> tuple[T.Compose | Identity, T.Compose | Identity]:
    """Heuristically separates early resize/cropping (deterministic) transforms to cache the results in shm"""
    RESIZE_TYPES = {"Resize", "CenterCrop"}

    # Take a slice to preserve the original img_transform
    remaining_transforms = img_transform.transforms[:]
    resize_transforms = []
    while type(remaining_transforms[0]).__name__ in RESIZE_TYPES:
        # This is a string comparison to include transforms.v2
        resize_transforms.append(remaining_transforms.pop(0))

    def compose_or_identity(transforms):
        return T.Compose(transforms) if transforms else Identity()

    return compose_or_identity(resize_transforms), compose_or_identity(remaining_transforms)


class MapDataset(Dataset):
    SHARED_MEM = Path("/dev/shm/dsagent_img_data")
    img_shm_dir = None
    IMG_THREADS = 2

    def __init__(
            self,
            tab_input_map: pd.DataFrame | None = None,
            img_input_map: pd.DataFrame | None = None,
            txt_input_map: pd.DataFrame | None = None,
            tab_target_map: pd.DataFrame | None = None,
            img_target_map: pd.DataFrame | None = None,
            txt_target_map: pd.DataFrame | None = None,
            tab_input_transform=Identity(),
            img_input_transform=DefaultImageInputTransform,
            txt_input_transform=Identity(),
            tab_target_transform=Identity(),
            img_target_transform=DefaultImageTargetTransform,
            txt_target_transform=Identity(),
            tab_target_inverse_transform=Identity(),
            img_target_inverse_transform=Identity(),
            txt_target_inverse_transform=Identity(),
            custom_tab_regression_scaler: Any | None = None,
            custom_img_input_transform: T.Compose = None,
            custom_img_target_transform: T.Compose = None,
            reindex_maps: bool = True,
            enable_img_shm=False,
            reuse_img_shm=True,
    ):
        """
        Input Maps are dataframes, each has the IDs as the first column:
         - Tabular input map is for the tabular modality. After the ID column, it has the other tabular features.
         - Image input map is for the Image modality. After the ID column, it has the paths to image inputs.
         - Text input map is for the Text modality. After the ID column, it has the text inputs.
        Target Maps are dataframes, each has the IDs as the first column:
         - Tabular target map is for the tabular modality. After the ID column, it has the other tabular targets.
         - Image target map is for the Image modality. After the ID column, it has the paths to image targets.
         - Text target map is for the Text modality. After the ID column, it has the text targets.
        Note: If one modality is not needed in the task, there should still be an input map and a target map dataframes
         with the ID column and no other columns.

        Args:
            tab_input_map: tabular input map
            img_input_map: image input map
            txt_input_map: text input map
            tab_target_map: tabular target map
            img_target_map: image target map
            txt_target_map: text target map
            tab_input_transform: maps the input to a format used by the model
            img_input_transform: maps the input to a format used by the model
            txt_input_transform: maps the input to a format used by the model
            tab_target_transform: maps the submission target to the format of the model's output
            img_target_transform: maps the submission target to the format of the model's output
            txt_target_transform: maps the submission target to the format of the model's output
            tab_target_inverse_transform: maps the output of the model back to the submission format
            img_target_inverse_transform: maps the output of the model back to the submission format
            txt_target_inverse_transform: maps the output of the model back to the submission format
            custom_tab_regression_scaler: a transform specifically applied to the regression targets (should be a
                class with a fit, a transform and an inverse_transform method, e.g. applying some rescaling)
        """
        self.tab_input_map = tab_input_map
        self.img_input_map = img_input_map
        self.txt_input_map = txt_input_map
        self.tab_target_map = tab_target_map
        self.img_target_map = img_target_map
        self.txt_target_map = txt_target_map

        self.tab_input_transform = tab_input_transform
        self.img_input_transform = img_input_transform
        self.txt_input_transform = txt_input_transform
        self.tab_target_transform = tab_target_transform
        self.img_target_transform = img_target_transform
        self.txt_target_transform = txt_target_transform
        self.triggered_img_transform_warning = False

        self.tab_target_inverse_transform = tab_target_inverse_transform
        self.img_target_inverse_transform = img_target_inverse_transform
        self.txt_target_inverse_transform = txt_target_inverse_transform

        self.enable_img_shm = enable_img_shm
        self.reuse_img_shm = reuse_img_shm

        # fetch index column
        if self.tab_input_map is not None:
            self.index_col = self.tab_input_map[["id"]]
        elif self.img_input_map is not None:
            self.index_col = self.img_input_map[["id"]]
        elif self.txt_input_map is not None:
            self.index_col = self.txt_input_map[["id"]]
        else:
            raise ValueError(
                f"MapDataset.index_col is None, this means that all input maps of all"
                f" modalities are not created. Please create at least one of them."
            )

        # now that we have self.index_col, we fill other None maps with the ids
        if self.tab_input_map is None:
            self.tab_input_map = self.index_col
        if self.tab_target_map is None:
            self.tab_target_map = self.index_col
            self.tab_regression_target_cols = []
        else:
            self.tab_regression_target_cols = [c for c in self.tab_target_map if c.endswith("_regression")]
        if self.img_input_map is None:
            self.img_input_map = self.index_col
        if self.img_target_map is None:
            self.img_target_map = self.index_col
        if self.txt_input_map is None:
            self.txt_input_map = self.index_col
        if self.txt_target_map is None:
            self.txt_target_map = self.index_col

        if reindex_maps:
            self.tab_input_map = self.tab_input_map.sort_values(by='id', ignore_index=True)
            self.img_input_map = self.img_input_map.sort_values(by='id', ignore_index=True)
            self.txt_input_map = self.txt_input_map.sort_values(by='id', ignore_index=True)
            self.tab_target_map = self.tab_target_map.sort_values(by='id', ignore_index=True)
            self.img_target_map = self.img_target_map.sort_values(by='id', ignore_index=True)
            self.txt_target_map = self.txt_target_map.sort_values(by='id', ignore_index=True)
            self.index_col = self.tab_input_map[['id']]

        if custom_img_input_transform is not None:
            self.img_input_transform = custom_img_input_transform
        if self.img_input_transform is None or isinstance(self.img_input_transform, Identity):
            warnings.warn(f"Using default image transform! Add custom if necessary.")
            self.img_input_transform = DefaultImageInputTransform

        if custom_img_target_transform is not None:
            self.img_target_transform = custom_img_target_transform
        if self.img_target_transform is None or isinstance(self.img_target_transform, Identity):
            warnings.warn(f"Using default image transform! Add custom if necessary.")
            self.img_target_transform = DefaultImageTargetTransform

        if len(self.tab_regression_target_cols) == 0:
            custom_tab_regression_scaler = None
        self.custom_tab_regression_scaler = custom_tab_regression_scaler
        if self.custom_tab_regression_scaler is not None and len(self.tab_regression_target_cols) > 0:
            self.custom_tab_regression_scaler.fit(self.tab_target_map[self.tab_regression_target_cols])
            self.tab_target_map[self.tab_regression_target_cols] = self.custom_tab_regression_scaler.transform(
                self.tab_target_map[self.tab_regression_target_cols])

        self.load_img_input_transform()

        self.classification_indices = None
        self.regression_indices = None

    def load_img_input_transform(self) -> pd.DataFrame:
        """Collection of operations to post-process self.img_input_transform"""
        if self.enable_img_shm:
            self.img_resize_transform, self.img_remaining_transform = split_img_transform(self.img_input_transform)
        else:
            self.img_resize_transform, self.img_remaining_transform = Identity(), self.img_input_transform

        self.img_input_map_resized = self.preload_and_resize_img_data(reuse_existing=self.reuse_img_shm)

    def preload_and_resize_img_data(self, max_workers=64, reuse_existing=True):
        """ Copies all img data into shm
        Set reuse_existing to true if you are sure the existing data is good (e.g. in a dataloader process)
        """
        if isinstance(self.img_resize_transform, Identity) or len(self.img_input_map.columns) <= 1:
            # Don't bother if resize is Identity or there are no images
            return self.img_input_map

        # Create a path-safe name by removing special characters
        os.umask(0o000)
        img_resize_transform_name = re.sub(r"\W+", "", repr(self.img_resize_transform))
        self.img_shm_dir = self.SHARED_MEM / img_resize_transform_name

        img_input_map_resized = pd.DataFrame()
        img_input_map_resized["id"] = self.img_input_map["id"]
        if not reuse_existing:
            self.img_shm_dir.mkdir(parents=True, exist_ok=True)
            if max_workers == 0:  # Sequential map for debugging/profiling
                for column_key in self.img_input_map.drop(columns=["id"]):
                    image_paths = self.img_input_map[column_key]
                    img_input_map_resized[column_key] = image_paths.map(
                        partial(transform_and_save, self.img_resize_transform, self.img_shm_dir)
                    )
            else:
                # with ProcessPoolExecutor(max_workers=max_workers) as executor:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for column_key in self.img_input_map.drop(columns=["id"]):
                        image_paths = self.img_input_map[column_key]
                        img_input_map_resized[column_key] = list(tqdm(
                            executor.map(
                                partial(transform_and_save, self.img_resize_transform, self.img_shm_dir),
                                image_paths
                            ),
                            total=len(image_paths),
                            desc=f"Preloading/resizing images: {column_key}"
                        ))
        else:  # If reuse_existing is set (e.g. in dataloader workers), just translate the paths assuming the files are there
            for column_key in self.img_input_map.drop(columns=["id"]):
                img_input_map_resized[column_key] = self.img_input_map[column_key].map(
                    partial(translate_benchmark_path, self.img_shm_dir)
                )

        return img_input_map_resized

    def unload_img_data(self):
        if hasattr(self, "img_shm_dir") and self.img_shm_dir is not None and os.path.exists(self.img_shm_dir):
            shutil.rmtree(self.img_shm_dir, ignore_errors=True)

    def split(self, indices: list[int] | None = None, frac: float | None = None, **split_kwargs) -> tuple[
        MapDataset, MapDataset]:
        """
        Creates two different datasets split according to the indices or a proportion
        specified using frac
        """
        map_attrs = [
            "tab_input_map", "img_input_map", "txt_input_map", "tab_target_map", "img_target_map", "txt_target_map"
        ]
        self_map_attrs = [getattr(self, a) for a in map_attrs]

        # Common transforms to all outputs
        common_transforms = dict(
            tab_input_transform=self.tab_input_transform,
            img_input_transform=self.img_input_transform,
            txt_input_transform=self.txt_input_transform,
            tab_target_transform=self.tab_target_transform,
            img_target_transform=self.img_target_transform,
            txt_target_transform=self.txt_target_transform,
            tab_target_inverse_transform=self.tab_target_inverse_transform,
            img_target_inverse_transform=self.img_target_inverse_transform,
            txt_target_inverse_transform=self.txt_target_inverse_transform,
            enable_img_shm=self.enable_img_shm,
            reuse_img_shm=self.reuse_img_shm,
        )

        # sample the indices of the split
        if indices is not None:
            if split_kwargs or frac:
                raise ValueError("Use either indices or frac, not both.")

            all_indices = list(range(len(self.tab_input_map)))
            indices_1 = indices
            indices_2 = [index for index in range(len(all_indices)) if index not in indices_1]

            map_attrs_1 = (a.iloc[indices_1] for a in self_map_attrs)
            map_attrs_2 = (a.iloc[indices_2] for a in self_map_attrs)
        else:
            split_maps_collated = train_test_split(*self_map_attrs, test_size=frac, **split_kwargs)
            # (train_tab_input_map, val_tab_input_map, train_img_input_map,
            # val_img_input_map, train_txt_input_map, val_txt_input_map,
            # train_tab_target_map, val_tab_target_map, train_img_target_map,
            # val_img_target_map, train_txt_target_map, val_txt_target_map)

            map_attrs_1 = split_maps_collated[::2]
            map_attrs_2 = split_maps_collated[1::2]

        dataset_1 = MapDataset(**dict(zip(map_attrs, map_attrs_1)), **common_transforms)
        dataset_2 = MapDataset(**dict(zip(map_attrs, map_attrs_2)), **common_transforms)

        return dataset_1, dataset_2

    def __len__(self) -> int:
        if self.tab_input_map is not None:
            return len(self.tab_input_map)
        if self.img_input_map is not None:
            return len(self.img_input_map)
        if self.txt_input_map is not None:
            return len(self.txt_input_map)
        raise ValueError()

    @staticmethod
    def _load_img(path) -> Image:
        return Image.open(path)

    @staticmethod
    def _load_txt(path):
        pass

    @staticmethod
    def _load_json(path):
        return json.load(open(path, 'r'))

    @staticmethod
    def _load_npy(path):
        img_as_arr = torch.from_numpy(np.load(path))
        if img_as_arr.ndim < 3:
            img_as_arr.unsqueeze_(0)
        return img_as_arr

    def get_targets_loss_indices(self) -> Tuple[np.array, np.array]:
        """
        Retrieves the indices of classification and regression columns from the transformed tabular target.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two NumPy arrays:
                - The first array contains indices of classification columns.
                - The second array contains indices of regression columns.
        """
        classification_indices = []
        regression_indices = []
        if self.tab_target_map is not None:
            transformed_tab_target = self.tab_target_transform(self.tab_target_map)
            transformed_tab_target_cols = transformed_tab_target.columns.values.tolist()[1:]  # Skipping `id` column

            classification_regression_mask = [1 if "_classification" in c else 0 for c in transformed_tab_target_cols]
            classification_indices = [i for i, val in enumerate(classification_regression_mask) if val == 1]
            regression_indices = [i for i, val in enumerate(classification_regression_mask) if val == 0]

        return np.array(classification_indices), np.array(regression_indices)

    def load(self, path, file_extension):
        file_extension = file_extension.lower()
        if file_extension in ImageExtensionType.list():
            return self._load_img(path)
        elif file_extension == "txt":
            return self._load_txt(path)
        elif file_extension == 'json':
            return self._load_json(path)
        elif file_extension == 'npy':
            return self._load_npy(path)
        else:
            print(f"image at path {path} was not loaded properly, please double-check.", flush=True)
            return None

    def __getitems__(self, idx: Iterable[int]) -> Tuple[
        np.ndarray,
        Tuple[pd.DataFrame, Iterable[Tensor], pd.DataFrame],
        Tuple[pd.DataFrame, Iterable[Tensor], pd.DataFrame],
    ]:
        """
        Args:
            idx: iterable of indices

        Returns:
            a tuple consisting of (in that order):
             - a batch of IDs corresponding to the indices
             - a batch of tabular elements, if any, corresponding to the indices
             - a batch of image elements, if any, corresponding to the indices
             - a batch of text elements, if any, corresponding to the indices
             - a batch of tabular targets, if any, corresponding to the indices
             - a batch of image targets, if any, corresponding to the indices
             - a batch of text targets, if any, corresponding to the indices
        """
        index_type = self.tab_input_map.iloc[idx]["id"].values.dtype
        indices = np.array(self.tab_input_map.iloc[idx]["id"]).astype(index_type)

        tab_inputs_batch = self.tab_input_map.iloc[idx]
        txt_inputs_batch = self.txt_input_map.iloc[idx]
        img_input_paths = self.img_input_map_resized.iloc[idx]
        img_inputs_batch = None
        if img_input_paths is not None:
            with ThreadPoolExecutor(max_workers=self.IMG_THREADS) as executor:
                images_df = (
                    img_input_paths.drop(columns="id")
                    .map(partial(executor.submit, partial(open_img_and_transform, self.img_remaining_transform)))
                )
                img_inputs_batches = [
                    torch.stack([f.result() for f in futures]) for futures in images_df.itertuples(index=False)
                ]

            if len(img_inputs_batches) >= 1:
                img_inputs_batch = torch.stack(img_inputs_batches)

        tab_targets_batch = self.tab_target_map.iloc[idx] if self.tab_target_map is not None else None
        txt_targets_batch = self.txt_target_map.iloc[idx] if self.txt_target_map is not None else None
        img_target_paths = self.img_target_map.iloc[idx] if self.img_target_map is not None else None
        img_targets_batch = None
        if img_target_paths is not None:
            images_df = img_target_paths.drop(columns="id").map(Image.open).map(self.img_target_transform)
            img_targets_batches = [torch.stack(images) for images in images_df.itertuples(index=False)]
            if len(img_targets_batches) >= 1:
                img_targets_batch = torch.stack(img_targets_batches)

        tab_inputs_batch = self.tab_input_transform(tab_inputs_batch)
        # img_inputs_batch = self.img_input_transform(img_inputs_batch) if img_inputs_batch is not None else None
        txt_inputs_batch = self.txt_input_transform(txt_inputs_batch)
        tab_targets_batch = self.tab_target_transform(tab_targets_batch) if tab_targets_batch is not None else None
        # img_targets_batch = self.img_target_transform(img_targets_batch) if img_targets_batch is not None else None
        txt_targets_batch = self.txt_target_transform(txt_targets_batch) if txt_targets_batch is not None else None
        return (
            indices,
            (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
            (tab_targets_batch, img_targets_batch, txt_targets_batch)
        )

    def __getitem__(self, idx: int):
        assert not isinstance(idx, Iterable)
        return self.__getitems__([idx])

    @staticmethod
    def create_train_test_datasets(
            train_tab_input_map_path: str = None,
            train_img_input_map_path: str = None,
            train_txt_input_map_path: str = None,
            train_tab_target_map_path: str = None,
            train_img_target_map_path: str = None,
            train_txt_target_map_path: str = None,
            test_tab_input_map_path: str = None,
            test_img_input_map_path: str = None,
            test_txt_input_map_path: str = None,
            tab_input_transform: Callable | Identity = Identity(),
            img_input_transform: Callable | DefaultImageInputTransform = DefaultImageInputTransform,
            txt_input_transform: Callable | Identity = Identity(),
            tab_target_transform: Callable | Identity = Identity(),
            img_target_transform: Callable | DefaultImageInputTransform = DefaultImageTargetTransform,
            txt_target_transform: Callable | Identity = Identity(),
            tab_target_inverse_transform: Callable | Identity = Identity(),
            img_target_inverse_transform: Callable | Identity = Identity(),
            txt_target_inverse_transform: Callable | Identity = Identity(),
            custom_tab_regression_scaler: Any | None = None,
            custom_img_train_input_transform: T.Compose = None,
            custom_img_test_input_transform: T.Compose = None,
            custom_img_target_transform: T.Compose = None,
            tab_fe: Callable | None = None,
            enable_img_shm: bool = False,
    ) -> tuple[MapDataset | None, MapDataset | None]:
        """
        Create (a train) or/and (a test) dataset(s)

        Args:
            train_tab_input_map_path: tabular input map path
            train_img_input_map_path: image input map path
            train_txt_input_map_path: text input map path
            train_tab_target_map_path: tabular target map path
            train_img_target_map_path: image target map path
            train_txt_target_map_path: text target map path
            test_tab_input_map_path: text target map path
            test_img_input_map_path: text target map path
            test_txt_input_map_path: text target map path
            tab_input_transform: maps the input to a format used by the model
            img_input_transform: maps the input to a format used by the model
            txt_input_transform: maps the input to a format used by the model
            tab_target_transform: maps the submission target to the format of the model's output
            img_target_transform: maps the submission target to the format of the model's output
            txt_target_transform: maps the submission target to the format of the model's output
            tab_target_inverse_transform: maps the output of the model back to the submission format
            img_target_inverse_transform: maps the output of the model back to the submission format
            txt_target_inverse_transform: maps the output of the model back to the submission format
            custom_tab_regression_scaler: a transform specifically applied to the regression targets (should be a
                class with a fit, a transform and an inverse_transform method, e.g. applying some rescaling)
            custom_img_train_input_transform: custom training input image transform function
            custom_img_test_input_transform: custom test input image transform function
            custom_img_target_transform: custom target image transform function
            tab_fe: preprocessing  function for tabular data
            enable_img_shm:
        """
        train_tab_input_map = pd.read_csv(train_tab_input_map_path) if train_tab_input_map_path is not None else None
        train_img_input_map = pd.read_csv(train_img_input_map_path) if train_img_input_map_path is not None else None
        if train_img_input_map is not None:
            train_img_input_map.attrs["path"] = Path(train_img_input_map_path)
        train_txt_input_map = pd.read_csv(train_txt_input_map_path) if train_txt_input_map_path is not None else None
        train_tab_target_map = pd.read_csv(train_tab_target_map_path) if train_tab_target_map_path is not None else None
        train_img_target_map = pd.read_csv(train_img_target_map_path) if train_img_target_map_path is not None else None
        train_txt_target_map = pd.read_csv(train_txt_target_map_path) if train_txt_target_map_path is not None else None

        test_tab_input_map = pd.read_csv(test_tab_input_map_path) if test_tab_input_map_path is not None else None
        test_img_input_map = pd.read_csv(test_img_input_map_path) if test_img_input_map_path is not None else None
        if test_img_input_map is not None:
            test_img_input_map.attrs["path"] = Path(test_img_input_map_path)
        test_txt_input_map = pd.read_csv(test_txt_input_map_path) if test_txt_input_map_path is not None else None
        if train_txt_input_map is not None:
            train_txt_input_map = train_txt_input_map.replace(np.nan, "")
        if test_txt_input_map is not None:
            test_txt_input_map = test_txt_input_map.replace(np.nan, "")

        # Apply feature engineering if provided
        if tab_fe is not None and train_tab_input_map is not None and test_tab_input_map is not None and train_tab_target_map is not None:
            tab_target_map_copy = copy.deepcopy(train_tab_target_map)
            train_tab_input_map, test_tab_input_map = tab_fe(
                train=train_tab_input_map, test=test_tab_input_map, target=tab_target_map_copy
            )

        train_dataset = test_dataset = None

        if train_tab_input_map is not None or train_img_input_map is not None or train_txt_input_map is not None:
            train_dataset = MapDataset(
                tab_input_map=train_tab_input_map,
                img_input_map=train_img_input_map,
                txt_input_map=train_txt_input_map,
                tab_target_map=train_tab_target_map,
                img_target_map=train_img_target_map,
                txt_target_map=train_txt_target_map,
                tab_input_transform=tab_input_transform,
                img_input_transform=img_input_transform,
                txt_input_transform=txt_input_transform,
                tab_target_transform=tab_target_transform,
                img_target_transform=img_target_transform,
                txt_target_transform=txt_target_transform,
                tab_target_inverse_transform=tab_target_inverse_transform,
                img_target_inverse_transform=img_target_inverse_transform,
                txt_target_inverse_transform=txt_target_inverse_transform,
                custom_tab_regression_scaler=custom_tab_regression_scaler,
                custom_img_input_transform=custom_img_train_input_transform,
                custom_img_target_transform=custom_img_target_transform,
                enable_img_shm=enable_img_shm,
                reuse_img_shm=False,  # This is only called once so force load the img data here
            )

        if test_tab_input_map is not None or test_img_input_map is not None or test_txt_input_map is not None:
            test_dataset = MapDataset(
                tab_input_map=test_tab_input_map,
                img_input_map=test_img_input_map,
                txt_input_map=test_txt_input_map,
                tab_target_map=None,
                img_target_map=None,
                txt_target_map=None,
                custom_img_input_transform=custom_img_test_input_transform,
                custom_tab_regression_scaler=None,
                custom_img_target_transform=None,
                enable_img_shm=enable_img_shm,
                reuse_img_shm=False,  # This is only called once so force load the img data here
            )

        if train_dataset is None and test_dataset is None:
            raise ValueError("At least one [train|test]_*_input_map must be provided.")

        return train_dataset, test_dataset


def map_dataset_collate_function(batch):
    indices, inputs, targets = batch

    def drop_id_and_use_none(df):
        if isinstance(df, pd.DataFrame):
            df = df.drop(columns="id", errors="ignore")
            return df if not df.empty else None
        return df

    return indices, tuple(map(drop_id_and_use_none, inputs)), tuple(map(drop_id_and_use_none, targets))


def map_dataset_collate_function_single(
        elements: Iterable[
            Tuple[
                np.ndarray,
                Tuple[pd.DataFrame, Tensor | list[Tensor], pd.DataFrame],
                Tuple[pd.DataFrame, Tensor | list[Tensor], pd.DataFrame]
            ]
        ]
) -> Tuple[
    np.ndarray,
    Tuple[pd.DataFrame, Iterable[Tensor], pd.DataFrame],
    Tuple[pd.DataFrame, Iterable[Tensor], pd.DataFrame]
]:
    """
    Args:
        elements: indices, (tab_inputs, img_inputs, txt_inputs), (tab_targets, img_targets, txt_targets). 
    """
    indices, inputs, targets = zip(*elements)
    inputs, targets = list(inputs), list(targets)

    collated_indices = np.array(list(chain.from_iterable(indices)))

    collated_tab_inputs = pd.concat(tab_inputs for tab_inputs, _, _ in inputs)
    collated_img_inputs = torch.stack([img_inputs for _, img_inputs, _ in inputs]) if inputs[0][1] is not None else None
    collated_txt_inputs = pd.concat(txt_inputs for _, _, txt_inputs in inputs)

    collated_tab_targets = pd.concat(tab_targets for tab_targets, _, _ in targets)
    collated_img_targets = torch.stack([img_targets for _, img_targets, _ in targets]) if targets[0][
                                                                                              1] is not None else None
    collated_txt_targets = pd.concat(txt_targets for _, _, txt_targets in targets)

    collated_tab_inputs = collated_tab_inputs.drop("id", axis=1, errors="ignore")
    collated_txt_inputs = collated_txt_inputs.drop("id", axis=1, errors="ignore")
    collated_tab_targets = collated_tab_targets.drop("id", axis=1, errors="ignore")
    collated_txt_targets = collated_txt_targets.drop("id", axis=1, errors="ignore")

    if collated_tab_inputs.empty:
        collated_tab_inputs = None
    if collated_tab_targets.empty:
        collated_tab_targets = None
    if collated_txt_inputs.empty:
        collated_txt_inputs = None
    if collated_txt_targets.empty:
        collated_txt_targets = None

    return (
        collated_indices,
        (collated_tab_inputs, collated_img_inputs, collated_txt_inputs),
        (collated_tab_targets, collated_img_targets, collated_txt_targets)
    )


def is_str_numeric(s: str) -> bool:
    """Checks if a string is in fact a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False


@dataclasses.dataclass
class TabularOnlyDataset:
    """
    This class only supports tabular-only tasks.
    The idea is to return the train.csv and test.csv necessary to run a RAMP kit.
    Starting from the metadata, data maps and transforms setup as the output of the data preprocessing pipeline,
     this class implements helper functions to create the classical CSV files used in RAMP.

    The functions are `get_train_dataset()` and `get_test_dataset()`
    """

    setup_dir: str
    original_target: bool = False

    def __post_init__(self):
        setup_dir = Path(self.setup_dir)
        self.train_tab_input_map = pd.read_csv(setup_dir / "train_tab_input_map.csv")
        self.train_tab_target_map = pd.read_csv(setup_dir / "train_tab_target_map.csv")
        self.train_tab_target_inv_tf_map = pd.read_csv(setup_dir / "df_tab_target_inv_transform.csv")
        self.test_tab_input_map = pd.read_csv(setup_dir / "test_tab_input_map.csv")
        self.sample_submission = pd.read_csv(setup_dir / "data/sample_submission.csv")
        self.column_types = json.load(open(setup_dir / "metadata/column_types.json", "r"))
        self.submission_names = json.load(open(setup_dir / "metadata/submission_names.json", "r"))
        self.task_category = json.load(open(setup_dir / "metadata/task_category.json", "r"))

        self.id_name = self.submission_names["id_name"]
        self.target_names = self.submission_names["target_names"]

        # get submission format function(s)
        plan = json.load(open(setup_dir / "plan.json", "r"))
        if (plan["submission_format"]["status"]["status_str"] == "DONE"
                and (setup_dir / "code_submission_format.py").exists()):
            spec = spec_from_file_location(
                name="df_to_submission_format", location=setup_dir / "code_submission_format.py"
            )
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            self.df_to_submission_format = getattr(module, "df_to_submission_format")
        elif (plan["submission_format_alt"]["status"]["status_str"] == "DONE"
              and (setup_dir / "code_submission_format.py").exists()):
            spec = spec_from_file_location(
                name="df_to_submission_format", location=setup_dir / "code_submission_format.py"
            )
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            self.df_to_submission_format = getattr(module, "df_to_submission_format")
        else:
            raise SubmissionFormatError("Both submission formats are FORCED passed!")

    @staticmethod
    def get_mixed_typed_columns(df: pd.DataFrame) -> list[str]:
        """
        Checks for each column if the values are of the same type and tries to cast the column to one type only if
        possible (i.e. str).
        """
        mixed_typed_columns = []
        for column in df.columns:
            inferred_type = pd.api.types.infer_dtype(df[column])
            if "mixed" in inferred_type:
                print(column, ':', inferred_type)
                mixed_typed_columns.append(column)
        return mixed_typed_columns

    def get_positive_class(self) -> dict[str, Any]:
        positive_class = json.load(open(Path(self.setup_dir) / "metadata/positive_class.json", "r"))
        try:
            colname_builder = {'_'.join(colname.split('_')[:-1]): colname for colname in
                               self.train_tab_target_map.columns}
            if len(set(self.target_names).intersection(colname_builder.keys())) == 0:
                raise KeyError(
                    f"target_names {self.target_names} and target columns from setup mapped back to submission format"
                    f" {colname_builder.keys()} do not share any name"
                )
            # rename column names in positive class dict
            for t in set(self.target_names).intersection(colname_builder.keys()):
                if is_str_numeric(str(positive_class[colname_builder[t]])):
                    positive_class[t] = positive_class[colname_builder[t]]
                else:
                    positive_class[t] = str(positive_class[colname_builder[t]])
                positive_class.pop(colname_builder[t])
        except KeyError:
            formatted_submission = self.df_to_submission_format(self.train_tab_target_inv_tf_map)  # returns a DataFrame
            class_values = set(self.train_tab_target_inv_tf_map[self.train_tab_target_inv_tf_map.columns[-1]].values)
            colname_builder = None
            for c in formatted_submission.columns:
                if len(set(formatted_submission[c].values).intersection(class_values)) > 1:
                    colname_builder = {c: list(positive_class.keys())[0]}
                    break
            assert colname_builder is not None
            if len(set(self.target_names).intersection(colname_builder.keys())) == 0:
                raise KeyError(
                    f"target_names {self.target_names} and target columns from setup mapped back to submission format"
                    f" {colname_builder.keys()} do not share any name"
                )
            for t in set(self.target_names).intersection(colname_builder.keys()):
                # rename column names in positive class dict
                if is_str_numeric(str(positive_class[colname_builder[t]])):
                    positive_class[t] = positive_class[colname_builder[t]]
                else:
                    positive_class[t] = str(positive_class[colname_builder[t]])
                positive_class.pop(colname_builder[t])
        return positive_class

    def get_train_dataset(self) -> pd.DataFrame:
        if self.original_target:
            return self._get_train_dataset_with_formatted_colnames()
        else:
            try:
                return self._get_train_dataset_with_colname_builder()
            except KeyError:
                return self._get_train_dataset_with_formatted_colnames()

    def _get_train_dataset_with_colname_builder(self) -> pd.DataFrame:
        train_data = self.train_tab_input_map.copy()
        # associate column name without the last '_classification'/'_regression' to the column in train_tab_target_map
        colname_builder = {'_'.join(colname.split('_')[:-1]): colname for colname in
                           self.train_tab_target_map.columns}
        for i in self.target_names:
            train_data[i] = self.train_tab_target_map[colname_builder[i]]
        train_data = train_data.rename({"id": self.id_name}, axis=1)
        return train_data

    def _get_train_dataset_with_formatted_colnames(self) -> pd.DataFrame:
        train_data = self.train_tab_input_map.copy()
        formatted_submission = self.df_to_submission_format(self.train_tab_target_inv_tf_map)
        # if len(formatted_submission.columns) == 2 and not self.original_target:
        #     warnings.warn(f"Use df_to_submission_format(train_tab_target_inv_tf_map) to create train data!")
        #     # use submission format column names if necessary for target columns
        #     for i in self.target_names:
        #         train_data[i] = formatted_submission[i]
        if len(formatted_submission.columns) == 2:
            # warnings.warn(f"Use tab_target_train_map.csv directly to create train data but"
            #               f" use target column name from df_to_submission_format(train_tab_target_inv_tf_map)")
            print(f"(#targets == 2) Use tab_target_train_map.csv directly to create train data!", flush=True)
            # simply use tab_target_train_map
            train_tab_target_colname = [
                self._clean_colname(c) for c in self.train_tab_target_map.columns if c != 'id'][0]
            train_tab_target_colname_orig = [c for c in self.train_tab_target_map.columns if c != 'id'][0]

            formatted_colname = formatted_submission.columns[-1]
            if formatted_colname != train_tab_target_colname and formatted_colname == self.target_names[0]:
                print(f"Target column name in formatted submission is {formatted_colname}.\n"
                      f"Target column name in training map is {train_tab_target_colname}.\n"
                      f"Please double-check but here we will be using the formatted one which is "
                      f"detected in sample_submission.csv.", flush=True)
                train_data[self.target_names[0]] = self.train_tab_target_map[train_tab_target_colname_orig]
            else:
                train_data[train_tab_target_colname] = self.train_tab_target_map[train_tab_target_colname_orig]
        else:
            # if we have more than just "id" and another target column name, it could be a multi-target task,
            # but it could also be that the submission format expects a proba for each class in different columns.
            # So for RAMP we simply use the tab_target_train_map.csv directly as it should contain only a single
            # column if it is indeed a multiclass classification task with a multi-column submission format
            # warnings.warn(f"Use tab_target_train_map.csv directly to create train data!")
            print(f"(#targets > 2) Use tab_target_train_map.csv directly to create train data!", flush=True)
            # simply use tab_target_train_map
            train_tab_target_colnames = [self._clean_colname(c) for c in self.train_tab_target_map if c != 'id']
            train_tab_target_colnames_orig = [c for c in self.train_tab_target_map.columns if c != 'id']
            for c, c_orig in zip(train_tab_target_colnames, train_tab_target_colnames_orig):
                train_data[c] = self.train_tab_target_map[c_orig]
        train_data = train_data.rename({"id": self.id_name}, axis=1)
        return train_data

    @staticmethod
    def _clean_colname(name: str) -> str:
        name = name.replace('_classification', '')
        name = name.replace('_regression', '')
        return name

    def get_test_dataset(self) -> pd.DataFrame:
        test_data = self.test_tab_input_map.copy()
        test_data = test_data.rename({"id": self.id_name}, axis=1)
        return test_data

    def get_sample_submission(self) -> pd.DataFrame:
        sample_submission = self.sample_submission.copy()
        sample_submission = sample_submission.rename({"id": self.id_name}, axis=1)
        return sample_submission
