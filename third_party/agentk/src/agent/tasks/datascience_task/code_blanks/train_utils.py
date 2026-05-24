import copy
import os
import pathlib
import traceback
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch.distributed as dist
import torch.optim
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

original_showwarning = warnings.showwarning

torch.manual_seed(123)
np.random.seed(123)
random_state = 123

from solve_params import MAX_EPOCHS

from solve_common_utils import tab_target_transform, tab_target_inverse_transform, enc, \
    class_names_columns_classification, get_data_loader, NanLossError, CustomTrainImageInputTransform, \
    CustomTestImageInputTransform, submission_format_functions, submission_names, tab_fe_preprocess, \
    img_target_transform, txt_target_transform, img_target_inverse_transform, txt_target_inverse_transform, \
    custom_tab_regression_scaler

try:
    from map_dataset import Identity, MapDataset, map_dataset_collate_function
except ImportError:
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    print(f"map_dataset should be in {current_dir}")
    raise

from code_metric import metric_function

# Some code_metric and code_submission_format modules patch warnings.showarning
# with a non-existent path, so we save the unpatched version and restore it
warnings.showwarning = original_showwarning


# --- Build dataset
def get_optional_path(path: str) -> str | None:
    """Return path if it exists, else return None"""
    if os.path.exists(path):
        return path


tab_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_tab_input_map.csv"
)
img_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_img_input_map.csv"
)
txt_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_txt_input_map.csv"
)
tab_target_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_tab_target_map.csv"
)
img_target_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_img_target_map.csv"
)
txt_target_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/train_txt_target_map.csv"
)

test_tab_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/test_tab_input_map.csv"
)
test_img_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/test_img_input_map.csv"
)
test_txt_input_map_path = get_optional_path(
    "@ROOT_DS_DATA_PATH@/test_txt_input_map.csv"
)

dataset, test_dataset = MapDataset.create_train_test_datasets(
    train_tab_input_map_path=tab_input_map_path,
    train_img_input_map_path=img_input_map_path,
    train_txt_input_map_path=txt_input_map_path,
    train_tab_target_map_path=tab_target_map_path,
    train_img_target_map_path=img_target_map_path,
    train_txt_target_map_path=txt_target_map_path,
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
    custom_img_train_input_transform=CustomTrainImageInputTransform,
    custom_img_test_input_transform=CustomTestImageInputTransform,
    tab_fe=tab_fe_preprocess.preprocess if tab_fe_preprocess else None,
    enable_img_shm=True  # Cache resized images in /dev/shm
)

if os.getenv("AGENT_DEBUG"):
    val_proportion = 0.05
else:
    val_proportion = 0.25
# Use the test transform pipeline for the validation dataset
train_dataset, validation_dataset = dataset.split(frac=val_proportion)
validation_dataset.img_input_transform = test_dataset.img_input_transform
validation_dataset.load_img_input_transform()


# Compute output_dim from inputs
train_input_sample = next(iter(get_data_loader(
    dataset=train_dataset, batch_size=2, is_sample_required=False, is_shuffle_required=False
)))

(
    indices,
    (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
    (tab_targets_batch, img_targets_batch, txt_targets_batch)
) = train_input_sample

if tab_targets_batch is not None:
    OUTPUT_DIM = tab_targets_batch.shape[1]
elif img_targets_batch is not None:
    OUTPUT_DIM = img_targets_batch.shape[1]
elif txt_targets_batch is not None:
    OUTPUT_DIM = txt_targets_batch.shape[1]


def get_tab_embedder():
    try:
        from tab_embed import TabularEmbedder, TAB_EMBED_DIM

        tab_input_dim = tab_inputs_batch.shape[-1]
        tab_embedder = TabularEmbedder(input_dim=tab_input_dim, embed_dim=TAB_EMBED_DIM)
        tab_embed_dim = TAB_EMBED_DIM
    except ImportError:
        tab_embedder = None
        tab_embed_dim = 0
    return tab_embedder, tab_embed_dim


def get_img_embedder():
    try:
        from img_embed import ImageEmbedder
        assert issubclass(ImageEmbedder, nn.Module)

        class AuxImageEmbedder(ImageEmbedder):
            """ Wrap ImageEmbedder to deal with the input dimension """

            def forward(self, x: torch.Tensor):
                """
                Args:
                    x: dimension (batch, n_images_per_id, n_channels, height, width)

                Returns:
                     y: dimension (batch, embed_dim)  --> the `n_images_per_id` are flatten
                """
                y = super().forward(x.view(-1, *x.shape[-3:]))
                return y.reshape(len(x), -1)

        img_embedder = AuxImageEmbedder()
        with torch.no_grad():
            img_embed_dim = img_embedder(img_inputs_batch[:2]).shape[-1]

    except ImportError:
        img_embedder = None
        img_embed_dim = 0

    return img_embedder, img_embed_dim


def get_txt_embedder():
    try:
        from txt_embed import TextEmbedder
        assert issubclass(TextEmbedder, nn.Module)

        class AuxTextEmbedder(TextEmbedder):
            """ Wrap TextEmbedder to deal with the input dimension """

            def forward(self, x: pd.DataFrame | np.ndarray | torch.Tensor):
                """
                Args:
                    x: dimension (batch, n_texts_per_id)

                Returns:
                     y: dimension (batch, embed_dim)  --> the `n_texts_per_id` are flattened
                """
                bsz = x.shape[0]
                if isinstance(x, pd.DataFrame):
                    x = x.values
                    x = x.reshape(-1, *x.shape[1:])
                    x = x.flatten().tolist()
                # make sure input is on same device as model params
                y = super(AuxTextEmbedder, self).forward(x)
                return y.reshape(bsz, -1)

        txt_embedder = AuxTextEmbedder()
        with torch.no_grad():
            txt_embed_dim = txt_embedder(txt_inputs_batch).shape[-1]

    except ImportError:
        txt_embedder = None
        txt_embed_dim = 0

    return txt_embedder, txt_embed_dim


def get_tab_head():
    try:
        from tab_head import TabularHead, regression_loss, classification_loss
    except ImportError:
        TabularHead = None
        regression_loss = None
        classification_loss = None
    return TabularHead, regression_loss, classification_loss


try:
    from img_head import get_img_head, img_loss
except ImportError:
    get_img_head = None
    img_loss = None

try:
    from txt_head import get_txt_head, txt_loss
except ImportError:
    get_txt_head = None
    txt_loss = None


class SubmissionModel(L.LightningModule):
    def __init__(self, learning_rate=1e-5, optimizer_choice='adam'):
        super().__init__()
        tab_embedder, tab_embed_dim = get_tab_embedder()
        img_embedder, img_embed_dim = get_img_embedder()
        txt_embedder, txt_embed_dim = get_txt_embedder()
        self.tab_embedder = tab_embedder
        self.img_embedder = img_embedder
        self.txt_embedder = txt_embedder
        self.embed_dim = tab_embed_dim + img_embed_dim + txt_embed_dim

        TabularHead, regression_loss, classification_loss = get_tab_head()
        self.tab_head = TabularHead(embed_dim=self.embed_dim, output_dim=OUTPUT_DIM)
        self.learning_rate = learning_rate
        self.optimizer_choice = optimizer_choice

        if get_img_head is None:
            self.img_head = None
        else:
            self.img_head = get_img_head()
        if get_txt_head is None:
            self.txt_head = None
        else:
            self.txt_head = get_txt_head()

        self.tab_regression_loss = regression_loss
        self.tab_classification_loss = classification_loss
        self.img_loss = img_loss
        self.txt_loss = txt_loss

        self.unfreeze_epoch = MAX_EPOCHS // 2

    def tab_loss(self, pred: torch.Tensor, target: pd.DataFrame) -> torch.Tensor:
        assert pred.shape == target.shape, (pred.shape, target.shape)
        target = torch.from_numpy(target.values).to(pred)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        loss = 0.0

        # If enc is not None that means we have classification target
        if enc:
            columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))

            classification_pred = pred[:, :len(columns_class_name)]
            classification_target = target[:, :len(columns_class_name)]

            groups_label = enc.categories_
            idx_start = 0
            for group in groups_label:
                group_pred = classification_pred[:, idx_start:idx_start + len(group)]
                group_target = classification_target[:, idx_start:idx_start + len(group)]
                idx_start += len(group)

                loss += self.tab_classification_loss(group_pred, group_target).mean()

            if pred.shape[-1] > len(columns_class_name):
                regression_pred = pred[:, len(columns_class_name):]
                regression_target = target[:, len(columns_class_name):]

                # If the target is nan, set the target to the predicted value so that there is actually no penalty
                regression_target[regression_target.isnan()] = regression_pred[regression_target.isnan()]

                loss += self.tab_regression_loss(regression_pred, regression_target).mean()

        # Full regression
        else:
            target[target.isnan()] = pred[target.isnan()]
            loss += self.tab_regression_loss(pred, target).mean()
        return loss

    def embed(self, tab, img, txt) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Get the embeddings for each modality"""

        if self.tab_embedder is None:
            tab_embed = None
        else:
            tab = torch.tensor(tab.values).to(device=self.device, dtype=self.dtype)
            tab_embed = self.tab_embedder(tab)
        if self.img_embedder is None:
            img_embed = None
        else:
            img = img.to(device=self.device, dtype=self.dtype)
            img_embed = self.img_embedder(img)
        if self.txt_embedder is None:
            txt_embed = None
        else:
            self.txt_embedder.to(self.device)
            txt_embed = self.txt_embedder(txt)
        return tab_embed, img_embed, txt_embed

    def decode(self, latent_embed: torch.Tensor
               ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Obtain logits / images / next tokens predictions given the latent embedding"""
        if self.tab_head is None:
            pred_tab = None
        else:
            pred_tab = self.tab_head(latent_embed)
        if self.img_head is None:
            pred_img = None
        else:
            pred_img = self.img_head(latent_embed)
        if self.txt_head is None:
            pred_txt = None
        else:
            pred_txt = self.txt_head(latent_embed)
        return pred_tab, pred_img, pred_txt

    def forward(
            self, tab: pd.DataFrame, img: torch.Tensor, txt: pd.DataFrame
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        tab_embed, img_embed, txt_embed = self.embed(tab=tab, img=img, txt=txt)

        to_fuse = [embed for embed in [tab_embed, img_embed, txt_embed] if embed is not None]
        assert len(to_fuse) > 0
        # Fuse the embeddings of each modality
        latent_embed = torch.cat(to_fuse, dim=1)
        return self.decode(latent_embed=latent_embed)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # training_step defines the train loop.
        (
            indices,
            (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
            (tab_targets_batch, img_targets_batch, txt_targets_batch),
        ) = batch

        pred_tab, pred_img, pred_txt = self.forward(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)

        loss = 0.0
        if tab_targets_batch is not None:
            loss += self.tab_loss(pred=pred_tab, target=tab_targets_batch).mean()
        if self.img_loss is not None:
            loss += self.img_loss(pred_img, img_targets_batch).mean()
        if self.txt_loss:
            loss += self.txt_loss(pred_txt, txt_targets_batch).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(indices))

        # Terminate training if loss is nan
        if torch.isnan(loss).item():
            raise NanLossError(f'Loss value is {loss}')
        return loss

    @staticmethod
    def get_param_groups(model: nn.Module | None, **kwargs) -> dict[str, ...] | None:
        if model is None:
            return None
        else:
            return {"params": model.parameters(), **kwargs}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = [
            self.get_param_groups(model=self.tab_embedder, lr=self.learning_rate),
            self.get_param_groups(model=self.tab_head, lr=self.learning_rate),
            self.get_param_groups(model=self.img_embedder, lr=self.learning_rate),
            self.get_param_groups(model=self.img_head, lr=self.learning_rate),
            self.get_param_groups(model=self.txt_embedder, lr=self.learning_rate),
            self.get_param_groups(model=self.txt_head, lr=self.learning_rate),
        ]

        params = [group for group in groups if group is not None]

        if self.optimizer_choice == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_choice == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_choice == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_choice}")

        return optimizer

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.unfreeze_epoch and self.img_embedder:
            print(f"Unfreezing layers at epoch {self.current_epoch}")
            self.img_embedder.unfreeze(n_last_layers=3)
        # if self.current_epoch % 2 == 0 and self.img_embedder:
        #     print(f"Unfreezing layers at epoch {self.current_epoch}")
        #     self.img_embedder.unfreeze(n_last_layers=self.unfreeze_layer_count)
        #     self.unfreeze_layer_count += 2

    def validation_step(self, batch, batch_idx):
        """Run"""
        # training_step defines the train loop.
        indices, inputs_batch, targets_batch = batch
        tab_inputs_batch, img_inputs_batch, txt_inputs_batch = inputs_batch
        tab_targets_batch, img_targets_batch, txt_targets_batch = targets_batch

        preds_batch = self.forward(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)
        tab_preds_batch, img_preds_batch, txt_preds_batch = preds_batch

        loss = 0.0
        if self.tab_loss is not None:
            loss += self.tab_loss(pred=tab_preds_batch, target=tab_targets_batch)
        if self.img_loss is not None:
            loss += self.img_loss(img_preds_batch, img_targets_batch).mean()
        if self.txt_loss:
            loss += self.txt_loss(txt_preds_batch, txt_targets_batch).mean()

        self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(indices))

    def get_submissions(
            self, dataloader: DataLoader, get_raw_preds: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """ Get submissions for each element of the dataloader

        Args:
            dataloader: dataloader on which
            get_raw_preds: whether to also return raw predictions
        """
        submission = pd.DataFrame([])
        raw_tab_preds = []
        target = pd.DataFrame([])

        img_preds = []
        txt_preds = []

        indices = []
        tab_targets_batch = None
        for batch in tqdm(dataloader):
            (indices_batch, (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
             (tab_targets_batch, img_targets_batch, txt_targets_batch)) = batch

            indices += list(indices_batch)

            with torch.no_grad():
                pred_tab, pred_img, pred_txt = self.forward(
                    tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch
                )

            if pred_tab is not None:

                # Classification target
                if enc:
                    softmax = nn.Softmax(dim=1)
                    columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))

                    classification_pred = pred_tab[:, :len(columns_class_name)].clone()

                    groups_label = enc.categories_
                    idx_start = 0
                    for group in groups_label:
                        group_pred = classification_pred[:, idx_start:idx_start + len(group)]
                        pred_tab[:, idx_start:idx_start + len(group)] = softmax(group_pred)
                        idx_start += len(group)

                pred_tab = pred_tab.cpu().numpy()
                if get_raw_preds:
                    raw_tab_preds.append(pred_tab)
                pred_tab = dataset.tab_target_inverse_transform(pred_tab, indices_batch)

                submission = pd.concat([submission, pred_tab])

            if pred_img is not None:
                assert not isinstance(dataset.img_target_inverse_transform,
                                      Identity), "Predicted images must be transformed"
                pred_img = dataset.img_target_inverse_transform(pred_img)
                img_preds.append(pred_img)

            if pred_txt is not None:
                pred_txt = dataset.txt_target_inverse_transform(pred_txt)
                txt_preds.append(pred_txt)

            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                tab_targets_batch = dataset.tab_target_inverse_transform(tab_targets_batch.values, indices_batch)
                target = pd.concat([target, tab_targets_batch])

            if img_targets_batch is not None and len(img_targets_batch) > 0:
                img_targets_batch = dataset.img_target_inverse_transform(img_targets_batch.values, indices_batch)
                target = pd.concat([target, img_targets_batch])

            if txt_targets_batch is not None and len(txt_targets_batch.columns) > 0:
                txt_targets_batch = dataset.txt_target_inverse_transform(txt_targets_batch.values, indices_batch)
                target = pd.concat([target, txt_targets_batch])

        # inverse transform of regression target
        if dataset.custom_tab_regression_scaler is not None:
            submission[dataset.tab_regression_target_cols] = dataset.custom_tab_regression_scaler.inverse_transform(
                submission[dataset.tab_regression_target_cols])
            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                target[dataset.tab_regression_target_cols] = dataset.custom_tab_regression_scaler.inverse_transform(
                    target[dataset.tab_regression_target_cols]
                )
        if get_raw_preds:
            raw_tab_preds = pd.DataFrame(np.concatenate(raw_tab_preds), index=indices)
        else:
            raw_tab_preds = None
        return submission, target, raw_tab_preds

    def get_blend_submissions(
            self, dataloader: DataLoader,
            blend_embeddings: bool = False,
    ) -> tuple[list[Tensor], list[Tensor], list[...]]:
        """Get submissions for each element of the dataloader"""
        blend_inputs = []
        blend_targets = []
        indices = []

        for batch in tqdm(dataloader):
            (
                indices_batch,
                (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
                (tab_targets_batch, img_targets_batch, txt_targets_batch),
            ) = batch

            indices += list(indices_batch)

            with torch.no_grad():
                if blend_embeddings:  # Blend from latent embedding
                    tab_embed, img_embed, txt_embed = self.embed(tab=tab_inputs_batch, img=img_inputs_batch,
                                                                 txt=txt_inputs_batch)
                    final_embedding = [embed for embed in [tab_embed, img_embed, txt_embed] if embed is not None]
                else:  # Blend from predictions
                    tab_pred, img_pred, txt_pred = self(tab=tab_inputs_batch, img=img_inputs_batch,
                                                           txt=txt_inputs_batch)
                    final_embedding = [embed for embed in [tab_pred, img_pred, txt_pred] if embed is not None]

                assert len(final_embedding) > 0
                latent_embed = torch.cat(final_embedding, dim=1)
                blend_inputs.append(latent_embed)

            batch_targets = []
            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                batch_targets.append(torch.tensor(tab_targets_batch.values, dtype=torch.float32))

            if img_targets_batch is not None and len(img_targets_batch) > 0:
                batch_targets.append(torch.tensor(img_targets_batch.values, dtype=torch.float32))

            if txt_targets_batch is not None and len(txt_targets_batch.columns) > 0:
                batch_targets.append(torch.tensor(txt_targets_batch.values, dtype=torch.float32))

            if batch_targets:
                blend_targets.append(torch.cat(batch_targets, dim=1))

        return blend_inputs, blend_targets, indices

    def get_score(self, dataloader: DataLoader) -> tuple[dict[str, float], pd.DataFrame]:
        """ Iterate through the dataloader to get predictions in submission format and compute the score
        Returns:
            score
            predictions in submission format
        """
        submissions, targets, _ = self.get_submissions(dataloader=dataloader)
        scores = {}
        for submission_format_func, submission_name in zip(submission_format_functions, submission_names):
            try:
                formatted_submission = submission_format_func(submissions)
                formatted_targets = submission_format_func(targets)
                try:
                    score = metric_function(y_pred=formatted_submission, y_true=formatted_targets)
                    score = float(score)
                except (IndexError, TypeError):
                    traceback.print_exc()
                    score = np.nan
                scores[submission_name] = score
            except Exception as e:
                print(f"Hit exception when trying to create {submission_name}: {e}")
        return scores, submissions


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def fast_train(
        training_config: dict[str, ...],
        accelerator: str,
        devices: str,
        max_training_time: str,
        max_exec_time: float
) -> None:
    print(f"BEFORE FAST TRAINING: max_exec_time: {max_exec_time}, max_training_time: {max_training_time}", flush=True)

    # Create a simple data module that explicitly manages batch size
    class FastDataModule(L.LightningDataModule):
        def __init__(self, train_dataset, val_dataset, train_batch_size, test_batch_size):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size

        def train_dataloader(self):
            return get_data_loader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size,
                is_sample_required=False,
                is_shuffle_required=False
            )

        def val_dataloader(self):
            return get_data_loader(
                dataset=self.val_dataset,
                batch_size=self.test_batch_size,
                is_sample_required=False,
                is_shuffle_required=False
            )

        def on_train_batch_size_update(self, new_size):
            """Method to update batch size after tuner runs"""
            print(f"Updating train batch size from {self.train_batch_size} to {new_size}", flush=True)
            self.train_batch_size = new_size
            # Keep the same ratio between train and test
            ratio = self.test_batch_size / self.train_batch_size if self.train_batch_size > 0 else 1
            self.test_batch_size = int(new_size * ratio)

    # Initial effective batch sizes from config
    effective_train_batch_size = training_config['train_batch_size']
    effective_test_batch_size = training_config['test_batch_size']
    accumulate_grad_batches = training_config['accumulate_grad_batches']
    train_batch_size = copy.copy(training_config['train_batch_size'])
    test_batch_size = copy.copy(training_config['test_batch_size'])

    data_module = FastDataModule(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size
    )

    # SubmissionModel with an overloaded `on_batch_size_update()` method
    class BatchSizeModel(type(SubmissionModel())):
        """Wrapper to handle batch size updates"""

        def __init__(self, data_module):
            super().__init__()
            self.data_module = data_module

        def on_batch_size_update(self, new_size):
            """Called by the batch size finder when a new size is found"""
            self.data_module.on_train_batch_size_update(new_size)

    fast_model = BatchSizeModel(data_module)
    fast_model.unfreeze_epoch = 0  # infer batch size with max nb. of layers unfrozen

    fast_trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=True
    )

    # start with micro batch size of 2 and multiply by 2 at most `max_trials` times until we OOM
    max_batch_size = find_max_batch_size(
        model=fast_model,
        data_module=data_module,
        accelerator=accelerator,
        devices=devices,
        max_trials=12
    )

    data_module.on_train_batch_size_update(max_batch_size)
    print(f"FOUND OPTIMAL BATCH SIZE: {data_module.train_batch_size}", flush=True)
    print("STARTING FAST TRAINING WITH OPTIMAL BATCH SIZE", flush=True)
    fast_trainer.fit(fast_model, datamodule=data_module)

    # Update the config with the final batch sizes
    data_module.test_batch_size = data_module.train_batch_size
    training_config["train_batch_size"] = data_module.train_batch_size
    training_config["test_batch_size"] = data_module.test_batch_size
    if max_batch_size >= effective_train_batch_size:
        training_config["accumulate_grad_batches"] = training_config.get("accumulate_grad_batches", 1)
    else:
        training_config["accumulate_grad_batches"] *= effective_train_batch_size // max_batch_size

    print(
        f"FINAL MICRO BATCH SIZES - "
        f"Train: {data_module.train_batch_size}, "
        f"Test: {data_module.test_batch_size}",
        flush=True
    )
    print(
        f"FINAL EFFECTIVE BATCH SIZES - "
        f"Train: {data_module.train_batch_size * training_config['accumulate_grad_batches']}, "
        f"Test: {data_module.test_batch_size * training_config['accumulate_grad_batches']}",
        flush=True
    )


def find_max_batch_size(model, data_module, accelerator, devices, max_trials=3):
    """Find the maximum batch size that fits in memory by starting with 2 and multiplying by 2 until we
    reach `max_trials` or we OOM."""
    print("Running manual batch size finder", flush=True)

    # Start with a small batch size and double it until we hit OOM
    current_batch_size = 4
    max_working_batch_size = 2

    for trial in range(max_trials):
        temp_trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0  # Skip validation
        )

        # Update batch size for this trial
        data_module.on_train_batch_size_update(current_batch_size)
        print(f"Trying batch size: {current_batch_size}", flush=True)

        try:
            temp_trainer.fit(model, datamodule=data_module)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower():
                print(f"Batch size {current_batch_size} caused OOM, reverting to {max_working_batch_size}", flush=True)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                break
            else:
                raise e

        max_working_batch_size = current_batch_size

        current_batch_size *= 2
        print(f"Batch size {max_working_batch_size} works, trying {current_batch_size} next", flush=True)

        # Clear memory
        with torch.no_grad():
            torch.cuda.empty_cache()


    return max_working_batch_size


def train(
        training_config: dict[str, ...],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        params: dict[str, ...],
        trial_num: int,
        accelerator: str,
        max_time: str,
        checkpoint_path: str,
        devices: list[int] | str = "auto",
) -> torch.Tensor:
    learning_rate = params['learning_rate'][0]
    optimizer_choice = params['optimizer'][0]

    logger = TensorBoardLogger("tb_logs", name=f"hebo_run")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        dirpath=checkpoint_path,
        filename=f"trial_{trial_num}",
    )
    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=5, verbose=True, mode="min")

    model = SubmissionModel(
        learning_rate=learning_rate,
        optimizer_choice=optimizer_choice,
    )

    extra_kwargs = dict(
        max_time=max_time,
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
    )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy="auto",
        **extra_kwargs
    )
    trainer.fit(model, train_dataloader, validation_dataloader)
    val_loss = trainer.checkpoint_callback.best_model_score

    return val_loss
