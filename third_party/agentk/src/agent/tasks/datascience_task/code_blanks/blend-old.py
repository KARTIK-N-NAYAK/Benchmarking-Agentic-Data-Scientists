import os
import pathlib
import pickle
from argparse import ArgumentParser
import shutil

import pytorch_lightning as L
import numpy as np
import pandas as pd
import torch.optim
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from hebo.optimizers.hebo import HEBO
import torch.distributed as dist
from torch.utils.data import Dataset

try:
    from tab_head import regression_loss, classification_loss
except ImportError as e:
    regression_loss = None
    classification_loss = None
from blend_params import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, HEBO_SPACE, N_TRIALS, TRIALS_DIR

from solve_common_utils import enc, class_names_columns_classification, submission_names, submission_format_functions
from train_utils import OUTPUT_DIM, dataset

torch.manual_seed(123)

num_gpus = torch.cuda.device_count()
distributed = num_gpus > 1 and dist.is_available() and dist.is_initialized()
print(f"STARTING DISTRIBUTED BLENDING ON {num_gpus} GPUS" if distributed else "STARTING SINGLE-GPU BLENDING")


class BlendConcatDataset(Dataset):
    def __init__(self, datasets, is_test: bool):
        self.is_test = is_test

        # Ensure all datasets have the same length
        assert len(set([dataset[:][0].shape[0] for dataset in datasets])) == 1, "Datasets must have the same length"

        # Prepare the combined dataset and target
        self.dataset, self.target = self.prepare_dataset(datasets)

    def prepare_dataset(self, datasets: list) -> tuple[torch.Tensor, torch.Tensor | None]:
        all_data = []
        reference_target = None

        for dataset in datasets:
            if self.is_test:
                data = dataset[:][0]  # Extract features
            else:
                data, target = dataset[:]  # Extract features and targets
                if reference_target is None:
                    reference_target = target
                assert torch.equal(target, reference_target), "Targets in datasets do not match!"

            # Append the data tensor to the all_data list
            all_data.append(data)

        # Concatenate all features along the feature dimension
        combined_data = torch.cat(all_data, dim=1)

        return combined_data, reference_target

    def __len__(self) -> int:
        # All datasets have the same length
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data_sample = self.dataset[idx]

        if self.is_test:
            return data_sample
        else:
            target_sample = self.target[idx] if self.target is not None else None
            return data_sample, target_sample


def get_mlp_dataloaders_train(lis_models: list):
    num_models = len(lis_models)
    datasets_lis = []
    for i in range(num_models):
        v = torch.load(f'{lis_models[i]}/train.pt', weights_only=False)
        datasets_lis.append(v)
    train_dataset = BlendConcatDataset(datasets_lis, is_test=False)
    blend_train_dataset, blend_val_dataset = torch.utils.data.random_split(train_dataset, [0.75, 0.25])
    blend_train_loader = DataLoader(blend_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    blend_val_loader = DataLoader(blend_val_dataset, batch_size=TRAIN_BATCH_SIZE)
    return blend_train_loader, blend_val_loader


def get_mlp_dataloaders_test(lis_models: list):
    indices = []
    num_models = len(lis_models)
    datasets_lis = []
    for i in range(num_models):
        v = torch.load(f'{lis_models[i]}/test.pt', weights_only=False)
        with open(f'{lis_models[i]}/indices.pkl', 'rb') as f:
            index = pickle.load(f)
        indices.append(index)
        datasets_lis.append(v)
    test_dataset = BlendConcatDataset(datasets_lis, is_test=True)
    blend_test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    return blend_test_loader, indices


class MLP(L.LightningModule):

    def __init__(self, embedding_dim: int, learning_rate=1e-5, optimizer_choice='adam', dropout=0.1):
        super().__init__()
        hidden_dim = 128
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUTPUT_DIM)
        )

        self.tab_regression_loss = regression_loss
        self.tab_classification_loss = classification_loss

        self.learning_rate = learning_rate
        self.optimizer_choice = optimizer_choice

    def tab_loss(self, pred: torch.Tensor, target: list) -> float:
        # Concatenate the target list if it's a list of tensors
        if isinstance(target, list):
            target = torch.cat(target, dim=1)

        assert pred.shape == target.shape, (pred.shape, target.shape)

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

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Use the forward method
        loss = self.tab_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = self.parameters()
        if self.optimizer_choice == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_choice == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_choice == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_choice}")

        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.tab_loss(y_hat, y)
        self.log('valid_loss', val_loss, prog_bar=True, on_epoch=True)
        return val_loss


def _format_submission_dtypes(formatted_submission: pd.DataFrame) -> pd.DataFrame:
    sample_submission = pd.read_csv("sample_submission.csv")
    for column in sample_submission:
        if sample_submission[column].isna().all():
            try:
                if np.array_equal(formatted_submission[column], formatted_submission[column].astype(int)):
                    formatted_submission[column] = formatted_submission[column].round().astype(int)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
        else:
            try:
                if (np.issubdtype(sample_submission[column].dtype, np.integer) and
                        np.array_equal(formatted_submission[column], formatted_submission[column].astype(int))):
                    formatted_submission[column] = formatted_submission[column].round().astype(int)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
            try:
                if (np.issubdtype(sample_submission[column].dtype, np.float32) and
                        np.array_equal(formatted_submission[column], formatted_submission[column].astype(float))):
                    formatted_submission[column] = formatted_submission[column].astype(float)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
            # else:
            #     formatted_submission[column] = formatted_submission[column].astype(sample_submission[column].dtypes)
    return formatted_submission


def objective(
        params: dict[str, ...],
        trial_num: int,
        accelerator: str,
        max_time: str,
        embedding_dim: int,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        max_trials: int = 10,
        devices: list[int] | str = "auto",
) -> torch.Tensor:
    learning_rate = params['learning_rate'][0]
    optimizer_choice = params['optimizer'][0]
    dropout = params['dropout'][0]

    logger = TensorBoardLogger("tb_logs", name=f"hebo_run")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        dirpath="./trials",
        filename=f"trial_{trial_num}",
    )
    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=5, verbose=True, mode="min")

    val_loss = None
    _trial = 0
    while val_loss is None and _trial < max_trials:
        model = MLP(
            learning_rate=learning_rate,
            optimizer_choice=optimizer_choice,
            dropout=dropout,
            embedding_dim=embedding_dim
        )

        try:
            extra_kwargs = dict(
                max_time=max_time,
                max_epochs=MAX_EPOCHS,
            )
            trainer = L.Trainer(
                accelerator=accelerator,
                devices=devices,
                logger=logger,
                callbacks=[checkpoint_callback, early_stop_callback],
                **extra_kwargs
            )
            trainer.fit(model, train_dataloader, validation_dataloader)
            val_loss = trainer.checkpoint_callback.best_model_score

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(e)

        finally:
            _trial += 1

    return val_loss


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def get_best_checkpoint(best_index: int, trials_dir: str = './trials') -> None:
    best_model_path = os.path.join(trials_dir, f"trial_{best_index}.ckpt")
    print("best_model_path", best_model_path)
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, "./blend_model.ckpt")
        print(f"Copied best model ")


def remove_trials(trials_dir: str) -> None:
    if os.path.exists(trials_dir):
        shutil.rmtree(trials_dir)
        print(f"Deleted existing {trials_dir} folder.")


def main(accelerator: str, devices: str, lis_models: list):
    logger = TensorBoardLogger("tb_logs")

    count_free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if
                       torch.cuda.utilization(device_id) == 0]
    accelerator = accelerator if len(count_free_gpus) else 'cpu'

    embedding_dim = None
    blend_test_dataloader, test_indices = get_mlp_dataloaders_test(lis_models=lis_models)
    train_dataloader, validation_dataloader = get_mlp_dataloaders_train(lis_models=lis_models)

    for x, y in train_dataloader:
        embedding_dim = x.shape[1]
        break

    opt = HEBO(HEBO_SPACE)
    remove_trials(TRIALS_DIR)
    for trial in range(N_TRIALS):

        if is_main_process():
            rec = opt.suggest(n_suggestions=1)
            params = rec.to_dict(orient='list')

        try:
            val_loss = objective(
                params=params,
                trial_num=trial,
                accelerator=accelerator,
                max_time=MAX_TIME,
                devices=devices,
                embedding_dim=embedding_dim,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader
            )
            if dist.is_initialized():
                # when using multi-node training, we need to gather the val_loss from each worker and
                # average it (or another reduce) and send it back to all workers, so they can share
                # the observations in their respective HEBO optimizers
                raise NotImplementedError(f"HEBO with DDP training not supported yet - "
                                          f"TODO: synchronize HEBO kernel params across ranks")
                val_loss = val_loss.reshape((1,)).to(f"cuda:{dist.get_rank()}")
                val_losses = [
                    torch.tensor([0.], device=f"cuda:{dist.get_rank()}") for _ in range(dist.get_world_size())
                ]
                dist.all_gather(val_losses, val_loss)
                val_loss_gathered = torch.cat(val_losses).mean().cpu().detach()
                opt.observe(rec, np.array([[val_loss_gathered.item()]]))
            else:
                opt.observe(rec, np.array([[val_loss.item()]]))
        except Exception as e:
            print(f"Trial {trial} failed: {e}")

        opt.X.to_csv("./hyperopt_x.csv")
        pd.DataFrame(opt.y).to_csv("./hyperopt_y.csv")

    # Best hyperparameters calculation only on rank 0
    if is_main_process() and opt.X.shape[0] >= 1:
        best_params = opt.X.iloc[opt.y.argmin()].to_dict()
        best_index = opt.y.argmin()
        print(f"Best hyper parameters: {best_params}")
        print(f"Best index: {best_index}")

    # Sync best index across processes
    best_index_list = [best_index]
    if distributed:
        dist.broadcast_object_list(best_index_list, src=0)
    best_index = best_index_list[0]

    get_best_checkpoint(best_index=best_index, trials_dir=trials_dir)

    trainer = L.Trainer(accelerator=accelerator, logger=logger, )
    blend_model = MLP.load_from_checkpoint(checkpoint_path=f"./blend_model.ckpt", embedding_dim=embedding_dim)
    blend_model.eval()
    blend_model.to(count_free_gpus[0] if len(count_free_gpus) else 'cpu')

    print(f"[START] Generate test predictions")
    test_submission = trainer.predict(blend_model, blend_test_dataloader, return_predictions=True)
    list_of_arrays = [t.numpy() for t in test_submission]
    test_submission = np.vstack(list_of_arrays)

    # Applying softmax as the transforms function expects probits (and not logits)
    if enc:
        softmax = nn.Softmax(dim=1)
        columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))
        classification_pred = test_submission[:, :len(columns_class_name)]
        groups_label = enc.categories_
        idx_start = 0
        for group in groups_label:
            group_pred = classification_pred[:, idx_start:idx_start + len(group)]

            test_submission[:, idx_start:idx_start + len(group)] = softmax(
                torch.from_numpy(group_pred)).detach().cpu().numpy()
            idx_start += len(group)

    test_submission = dataset.tab_target_inverse_transform(test_submission, test_indices[0][0])
    # inverse transform of the regression targets if needs be
    if dataset.custom_tab_regression_scaler is not None:
        test_submission[
            dataset.tab_regression_target_cols] = dataset.custom_tab_regression_scaler.inverse_transform(
            test_submission[dataset.tab_regression_target_cols]
        )
    for submission_format_func, submission_name in zip(submission_format_functions, submission_names):
        formatted_sub = submission_format_func(test_submission)
        formatted_sub = _format_submission_dtypes(formatted_sub)
        formatted_sub.to_csv(os.path.join(str(pathlib.Path(__file__).parent), submission_name), index=False)

    print(f"[END] Generate test predictions in {str(pathlib.Path(__file__).parent)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default='auto')
    parser.add_argument("--submissions", nargs='+')
    args = parser.parse_args()
    if os.getenv("AGENT_DEBUG", False):
        MAX_EPOCHS = 1
        TRAIN_BATCH_SIZE = 64
        TEST_BATCH_SIZE = 64
        MAX_TIME = "00:00:01:00"
    else:
        MAX_EPOCHS = 200
        MAX_TIME = "00:10:00:00"
    main(accelerator=args.accelerator, devices=args.devices, lis_models=args.submissions)
