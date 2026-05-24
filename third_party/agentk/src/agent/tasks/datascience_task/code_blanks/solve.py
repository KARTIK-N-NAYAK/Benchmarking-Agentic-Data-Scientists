import atexit
import json
import os
import pathlib
import shutil
import time
import warnings
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.optim
from datasets import Dataset
from hebo.optimizers.hebo import HEBO
from numpy import ndarray
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.model_selection import StratifiedKFold, KFold
from torch import Tensor
from tqdm import tqdm

from solve_common_utils import enc, get_data_loader, NanLossError, CustomTrainImageInputTransform, submission_names, \
    submission_format_functions, tab_target_inverse_transform
from solve_params import HEBO_SPACE, N_TRIALS, NUM_FOLDS, MAX_TIME, TRIALS_DIR, CV_TRIALS_DIR, TTA_ROUNDS, \
    EFFECTIVE_TRAIN_BATCH_SIZE, EFFECTIVE_TEST_BATCH_SIZE, ACCUMULATE_GRAD_BATCHES
from train_utils import train, get_img_embedder, SubmissionModel, dataset, train_dataset, validation_dataset, \
    fast_train, test_dataset

original_showwarning = warnings.showwarning

torch.manual_seed(123)
random_state = 123

# Set umask so that all files are created in mode 777
os.umask(0o000)

num_gpus = torch.cuda.device_count()
distributed = num_gpus > 1 and dist.is_available() and dist.is_initialized()
print(f"STARTING DISTRIBUTED TRAINING ON {num_gpus} GPUS" if distributed else "STARTING SINGLE-GPU TRAINING")

TRAINING_CONFIG = {
    "hebo_space": HEBO_SPACE,
    "n_trials": N_TRIALS,
    "num_folds": NUM_FOLDS,
    "max_time": MAX_TIME,
    "trials_dir": TRIALS_DIR,
    "cv_trials_dir": CV_TRIALS_DIR,
    "tta_rounds": TTA_ROUNDS,
    "train_batch_size": EFFECTIVE_TRAIN_BATCH_SIZE,
    "test_batch_size": EFFECTIVE_TEST_BATCH_SIZE,
    "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES
}


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


@rank_zero_only
def remove_trials(trials_dir: str) -> None:
    if os.path.exists(trials_dir):
        shutil.rmtree(trials_dir)
        print(f"Deleted existing {trials_dir} folder.")


@rank_zero_only
def get_best_checkpoint(best_index: int, trials_dir: str = './trials') -> None:
    best_model_path = os.path.join(trials_dir, f"trial_{best_index}.ckpt")
    print("best_model_path", best_model_path)
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, "./best_model.ckpt")
        print(f"Copied best model ")


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
                if np.issubdtype(sample_submission[column].dtype, np.integer) and np.array_equal(
                        formatted_submission[column], formatted_submission[column].astype(int)
                ):
                    formatted_submission[column] = formatted_submission[column].round().astype(int)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
            try:
                if np.issubdtype(sample_submission[column].dtype, np.float32) and np.array_equal(
                        formatted_submission[column], formatted_submission[column].astype(float)
                ):
                    formatted_submission[column] = formatted_submission[column].astype(float)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
            # else:
            #     formatted_submission[column] = formatted_submission[column].astype(sample_submission[column].dtypes)
    return formatted_submission


def time_to_seconds(time_str: str) -> float:
    days, hours, minutes, seconds = map(int, time_str.split(':'))
    total_seconds = (days * 3600 * 24) + (hours * 3600) + (minutes * 60) + seconds

    return total_seconds


def seconds_to_time_string(total_seconds: float) -> str:
    if total_seconds <= 0:
        return "00:00:00:10"  # leave 10 seconds

    days = int(total_seconds // (3600 * 24))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    time_string = f'{days:02}:{hours:02}:{minutes:02}:{seconds:02}'

    return time_string


@rank_zero_only
def get_max_time_per_bo_trial(max_total_runtime: float = None, margin_time: int = 900) -> str:
    """
    Find maximum time allowed to execute this script and per training round
    max_exec_time: max time for the whole submission (in seconds)
    margin_time: Margin time to accommodate test generation (in seconds)

    Returns:
        M
    """
    max_time_per_bo_trial = MAX_TIME
    if time_to_seconds(max_time_per_bo_trial) > (max_total_runtime - margin_time):
        max_time_per_bo_trial = seconds_to_time_string(total_seconds=max_total_runtime - margin_time)

    return max_time_per_bo_trial


@rank_zero_only
def manage_best_loss(best_loss_info: dict, trial_loss: float, trial_num: int) -> dict:
    """
    keep the best loss and trial info and delete trials those are not best
    """
    if not best_loss_info:
        best_loss_info['val_loss'] = trial_loss
        best_loss_info['trial_num'] = trial_num
    else:
        if best_loss_info['val_loss'] < trial_loss:
            trials_dir_ = os.path.join(TRIALS_DIR, f'trials_{trial_num}')
        else:
            old_trial_num = best_loss_info['trial_num']
            trials_dir_ = os.path.join(TRIALS_DIR, f'trials_{old_trial_num}')

        remove_trials(trials_dir_)

    print(f"Best trial so far is {best_loss_info['trial_num']} with average loss {best_loss_info['val_loss']}")
    return best_loss_info


def get_suitable_kfold(ref_target: pd.DataFrame, ref_target_column: str) -> StratifiedKFold | KFold:
    """
    Return suitable kfold strategy.
    """
    if enc:
        try:
            kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)
            kfold.split(np.zeros(len(ref_target[ref_target_column])), ref_target[ref_target_column])
        except Exception as e:
            """ Falls back to KFold"""
            print(e)
            print("Falls back to KFold")
            kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)

    return kfold


def get_best_params() -> dict:
    """
    provides best hyper parameter of HEBO run
    """
    hebo_x = os.path.join("hyperopt_x.csv")
    hebo_y = os.path.join("hyperopt_y.csv")
    if os.path.exists(hebo_x) and os.path.exists(hebo_y):
        try:
            hebo_x_df = pd.read_csv(hebo_x)
            hebo_y_df = pd.read_csv(hebo_y)
            best_index = hebo_y_df['0'].idxmin()
            best_params = hebo_x_df.iloc[best_index].to_dict()
            params = {"learning_rate": [best_params["learning_rate"]], "optimizer": [best_params["optimizer"]]}
        except Exception as e:
            print(e)
            params = {"learning_rate": [1e-4], "optimizer": ["adam"]}
    else:
        params = {"learning_rate": [1e-4], "optimizer": ["adam"]}

    return params


def start_train(
        training_config: dict[str, ...],
        params: dict[str, ...],
        train_ds: Dataset,
        validation_ds: Dataset,
        trial_num: int,
        accelerator: str,
        max_training_time: str,
        checkpoint_path: str,
        max_trials: int = 10,
        devices: list[int] | str = "auto"
) -> Tensor:
    train_batch_size = training_config["train_batch_size"]
    test_batch_size = training_config["test_batch_size"]
    accumulate_grad_batches = training_config["accumulate_grad_batches"]
    train_dataloader = get_data_loader(
        dataset=train_ds, batch_size=train_batch_size, is_sample_required=True, is_shuffle_required=False
    )
    validation_dataloader = get_data_loader(
        dataset=validation_ds, batch_size=train_batch_size, is_sample_required=False, is_shuffle_required=False
    )
    val_loss = None
    _trial = 0
    while val_loss is None and _trial < max_trials:
        try:
            val_loss = train(
                training_config=training_config,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                params=params,
                trial_num=trial_num,
                accelerator=accelerator,
                max_time=max_training_time,
                checkpoint_path=checkpoint_path,
                devices=devices,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower():
                print('| WARNING: ran out of memory, retrying with half the batch size')
                with torch.no_grad():
                    torch.cuda.empty_cache()

                train_batch_size //= 2
                test_batch_size //= 2
                accumulate_grad_batches *= 2
                if train_batch_size <= 0:
                    raise RuntimeError("No batch size seems to fit on the device")

                train_dataloader = get_data_loader(
                    dataset=train_ds, batch_size=train_batch_size, is_sample_required=True,
                    is_shuffle_required=False
                )
                validation_dataloader = get_data_loader(
                    dataset=validation_ds, batch_size=train_batch_size, is_sample_required=False,
                    is_shuffle_required=False
                )
            else:
                raise e
        except Exception as e:
            raise e
        finally:
            _trial += 1

    training_config["train_batch_size"] = train_batch_size
    training_config["test_batch_size"] = test_batch_size
    training_config["accumulate_grad_batches"] = accumulate_grad_batches

    return val_loss


def cross_validation_train(
        training_config: dict[str, ...],
        accelerator: str,
        max_training_time: str,
        devices: list[int] | str = "auto",
) -> Tensor:
    remove_trials(CV_TRIALS_DIR)
    params = get_best_params()

    if dataset.tab_target_map is not None:
        ref_input = dataset.tab_input_map
    elif dataset.img_target_map is not None:
        ref_input = dataset.img_input_map
    elif dataset.txt_target_map is not None:
        ref_input = dataset.txt_input_map
    else:
        raise RuntimeError

    if dataset.tab_target_map is not None:
        ref_target = dataset.tab_target_map
    elif dataset.img_target_map is not None:
        ref_target = dataset.img_target_map
    elif dataset.txt_target_map is not None:
        ref_target = dataset.txt_target_map
    else:
        raise RuntimeError
    ref_target_column = [col for col in ref_target.columns if col != 'id'][0]

    kfold = get_suitable_kfold(ref_target=ref_target, ref_target_column=ref_target_column)

    loss_values = []
    for fold, (train_idx, val_idx) in enumerate(
            kfold.split(ref_input, ref_target[ref_target_column])
    ):
        print(f"Training fold {fold + 1}/{NUM_FOLDS}")

        # Create subsets for train and validation
        train_subset, val_subset = dataset.split(indices=train_idx.tolist())

        val_loss = start_train(
            training_config=training_config,
            train_ds=train_subset,
            validation_ds=val_subset,
            params=params,
            trial_num=fold,
            accelerator=accelerator,
            max_training_time=max_training_time,
            checkpoint_path=CV_TRIALS_DIR,
            devices=devices,
        )
        loss_values.append(val_loss)

    stacked_losses = torch.stack(loss_values)
    print(f"Average loss of CV is : {stacked_losses.mean()}")

    return stacked_losses.mean()


def multi_trial_train(
        training_config: dict[str, ...],
        accelerator: str,
        devices: str,
        max_time_per_bo_trial: str,
        max_exec_time: float
) -> int | None:
    start_time: float = time.time()
    remove_trials(TRIALS_DIR)
    os.makedirs(TRIALS_DIR, exist_ok=True)
    best_index = None

    if N_TRIALS == 0:
        params = {"learning_rate": [1e-4], "optimizer": ["adam"]}
        _ = start_train(
            training_config=training_config,
            train_ds=train_dataset,
            validation_ds=validation_dataset,
            params=params,
            trial_num=0,
            accelerator=accelerator,
            max_training_time=max_time_per_bo_trial,
            checkpoint_path=TRIALS_DIR,
            devices=devices,
        )
        best_index = 0
    else:
        predefined_suggestions = [
            {"learning_rate": [1e-4], "optimizer": ["adam"]},
            {"learning_rate": [1e-4], "optimizer": ["sgd"]},
            {"learning_rate": [1e-4], "optimizer": ["adamw"]},
        ]

        # Only main process creates HEBO instance
        if is_main_process():
            print("Initializing HEBO on main process")
            opt = HEBO(HEBO_SPACE)
        else:
            opt = None

        if distributed:
            # Sync HEBO instance across all processes
            dist.barrier()

        for trial in range(N_TRIALS):
            print(f"Trial {trial + 1}/{N_TRIALS}")
            elapsed_time = time.time() - start_time
            if elapsed_time > max_exec_time:
                print("Total time exceeded. Stopping trials.")
                break

            if time_to_seconds(time_str=max_time_per_bo_trial) > (max_exec_time - elapsed_time):
                max_time_per_bo_trial = seconds_to_time_string(max_exec_time - elapsed_time)

            # SUGGEST PARAMS: Only main process suggests params using HEBO
            if is_main_process():
                if len(predefined_suggestions) > 0:
                    params = predefined_suggestions.pop(0)
                    rec = pd.DataFrame.from_dict(params, orient='columns')
                else:
                    rec = opt.suggest(n_suggestions=1)
                    params = rec.to_dict(orient='list')

                if time_to_seconds(max_time_per_bo_trial) > (max_exec_time - elapsed_time):
                    max_time_per_bo_trial = seconds_to_time_string(max_exec_time - elapsed_time)

            else:
                params = None
                rec = None
                max_time_per_bo_trial = None

            if distributed:
                # Broadcast params, rec and max_time_per_bo_trial to all ranks
                params_list = [params]
                dist.broadcast_object_list(params_list, src=0)
                params = params_list[0]
                max_time_per_bo_trial_list = [max_time_per_bo_trial]
                dist.broadcast_object_list(params_list, src=0)
                max_time_per_bo_trial = max_time_per_bo_trial_list[0]

            # Training happens on all GPUs using the same suggested params
            try:
                val_loss = start_train(
                    training_config=training_config,
                    train_ds=train_dataset,
                    validation_ds=validation_dataset,
                    params=params,
                    trial_num=trial,
                    accelerator=accelerator,
                    max_training_time=max_time_per_bo_trial,
                    checkpoint_path=TRIALS_DIR,
                    devices=devices,
                )

                if distributed:
                    # Reduce loss across all GPUs
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

                # OBSERVE: Only main process updates HEBO state
                if is_main_process():
                    opt.observe(rec, np.array([[val_loss.item()]]))
            except NanLossError as e:
                print(f"Trial {trial} failed: {e}, stopping the trials.")
                break
            except Exception as e:
                print(f"Trial {trial} failed: {e}")

            # Save HEBO state from rank 0 only
            if is_main_process():
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

    return best_index


def main(training_config: dict[str, ...], max_total_runtime: float, accelerator: str, devices: str) -> None:
    """
    Args:
        - max_total_runtime: maximum total runtime in seconds
    """
    count_free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if
                       torch.cuda.utilization(device_id) < 25]
    accelerator = accelerator if count_free_gpus else 'cpu'
    device = 'cuda' if count_free_gpus else 'cpu'

    if (os.getenv("RUN_TTA_ONLY", False) in ["1", "true", "True"]
            or os.getenv("RUN_INFERENCE_ONLY", False) in ["1", "True", "true"]):
        return generate_submissions(training_config=training_config, device=device)
    if os.getenv("RUN_CV_ONLY", False) in ["1", "true", "True"]:
        cross_validation_train(
            training_config=training_config,
            accelerator=accelerator,
            devices=devices,
            max_training_time=seconds_to_time_string(max_total_runtime)
        )
        return generate_cv_submissions(training_config=training_config, device=device)

    margin_time = 60 * 15

    start_time: float = time.time()

    max_time_per_bo_trial = get_max_time_per_bo_trial(
        max_total_runtime=max_total_runtime, margin_time=margin_time
    )

    # first perform a fast training loop to check that everything is ok
    fast_train(
        training_config=training_config,
        accelerator=accelerator,
        devices=devices,
        max_exec_time=max_total_runtime,
        max_training_time=max_time_per_bo_trial
    )

    max_total_runtime = max_total_runtime - (time.time() - start_time)
    max_time_per_bo_trial = get_max_time_per_bo_trial(
        max_total_runtime=max_total_runtime, margin_time=margin_time
    )

    # then do the real training
    best_index = multi_trial_train(
        training_config=training_config,
        accelerator=accelerator,
        devices=devices,
        max_exec_time=max_total_runtime - margin_time,
        max_time_per_bo_trial=max_time_per_bo_trial
    )

    # copy the best checkpoint and delete the trials dir
    if best_index is not None:
        get_best_checkpoint(best_index=best_index, trials_dir=TRIALS_DIR)
        remove_trials(TRIALS_DIR)
        # os.rmdir("./lightning_logs")

    max_total_runtime -= (time.time() - start_time)
    if time_to_seconds(max_time_per_bo_trial) > max_total_runtime:
        max_time_per_bo_trial = seconds_to_time_string(max_total_runtime)

    # Generate normal and tta submissions
    generate_submissions(training_config=training_config, device=device)

    cross_validation_train(
        training_config=training_config,
        accelerator=accelerator,
        devices=devices,
        max_training_time=max_time_per_bo_trial
    )

    # Generate CV and its tta based submissions
    generate_cv_submissions(training_config=training_config, device=device)


def generate_submission_transform(preds_mean: ndarray, pred_ref: pd.DataFrame) -> ndarray:
    submission_transform = tab_target_inverse_transform(preds_mean, pred_ref.index.values)
    # Note we are using the original dataset.custom_tab_regression_scaler as it needs to be fit
    # on targets, and obviously the test dataset doesn't have targets
    if dataset.custom_tab_regression_scaler is not None:
        submission_transform[
            dataset.tab_regression_target_cols
        ] = dataset.custom_tab_regression_scaler.inverse_transform(
            submission_transform[dataset.tab_regression_target_cols]
        )

    return submission_transform


def create_submission_files(submission, submission_tag: str, dst_dir: pathlib.Path) -> int:
    n_submissions = 0
    for submission_format_func, submission_name in zip(submission_format_functions, submission_names):
        try:
            formatted_sub = submission_format_func(submission)
            formatted_sub = _format_submission_dtypes(formatted_sub)
            submission_name = submission_tag + submission_name
            formatted_sub.to_csv(dst_dir / submission_name, index=False)
            n_submissions += 1
        except Exception as e:
            print(f"Hit exception when trying to create {submission_name}: {e}")

    return n_submissions


@rank_zero_only
def generate_submissions(training_config: dict[str, ...], device: str) -> None:
    validation_dataloader = get_data_loader(
        dataset=validation_dataset, batch_size=training_config["train_batch_size"],
        is_sample_required=False, is_shuffle_required=False
    )
    test_dataloader = get_data_loader(
        dataset=test_dataset, batch_size=training_config["test_batch_size"],
        is_sample_required=False, is_shuffle_required=False
    )

    model = SubmissionModel.load_from_checkpoint(checkpoint_path="./best_model.ckpt")
    model.eval()
    torch.cuda.empty_cache()

    device = torch.device(device)
    model.to(device)

    run_tta_only = os.getenv("RUN_TTA_ONLY", False) in ["1", "true", "True"]

    current_dir = pathlib.Path(__file__).parent

    if not run_tta_only:
        val_scores, validation_submission = model.get_score(validation_dataloader)
        save_path = str(current_dir / "val_scores.json")
        with open(save_path, 'w') as writer:
            writer.write(json.dumps(val_scores))

    print(f"[START] Generate test predictions")
    use_tta = os.getenv("TTA", False) in ["1", "true", "True"] or run_tta_only  # TODO: handle TTA not as env variable

    test_submission, _, no_tta_raw_preds = model.get_submissions(dataloader=test_dataloader, get_raw_preds=use_tta)
    n_submissions = create_submission_files(submission=test_submission, submission_tag="", dst_dir=current_dir)
    img_embed, _ = get_img_embedder()

    # TTA is only implemented for Image based competitions, and restrict tta to only image based ones
    if use_tta and img_embed is not None:
        print(f"[START] Generate test predictions using TTA")
        tta_dir = current_dir / "tta"
        os.makedirs(tta_dir, exist_ok=True)
        predictions = {"test_transform": no_tta_raw_preds}  # store raw predis for the different variants

        # use random transforms in the test_dataset and take majority vote
        test_dataset.img_input_transform = CustomTrainImageInputTransform
        test_dataset.load_img_input_transform()

        for tta_round in tqdm(range(TTA_ROUNDS), desc="Generating TTA predictions"):
            test_submission, _, raw_preds = model.get_submissions(dataloader=test_dataloader, get_raw_preds=True)

            create_submission_files(
                submission=test_submission, submission_tag=f"tta_round_{tta_round}-", dst_dir=tta_dir
            )

        # construct the submission from raw preds

        raw_preds_mean = np.stack(list(predictions.values())).mean(0)
        tta_submission_transform = generate_submission_transform(preds_mean=raw_preds_mean, pred_ref=no_tta_raw_preds)
        n_submissions = create_submission_files(
            submission=tta_submission_transform, submission_tag=f"tta-", dst_dir=current_dir
        )

    if n_submissions > 0:
        print(f"[END] Generate test predictions in {current_dir}")
    else:
        print(f"[END] Failed to create a submission in {current_dir}")


@rank_zero_only
def generate_cv_submissions(training_config: dict[str, ...], device: str) -> None:
    test_dataloader = get_data_loader(
        dataset=test_dataset, batch_size=training_config["test_batch_size"], is_sample_required=False,
        is_shuffle_required=False
    )
    current_dir = pathlib.Path(__file__).parent
    cv_dir = current_dir / "cv"
    os.makedirs(cv_dir, exist_ok=True)
    cv_predictions = []
    raw_pred_ref = None
    cv_trial_dir = CV_TRIALS_DIR

    print(f"[START] Generate cv test predictions")

    use_tta = os.getenv("TTA", False) in ["1", "true", "True"]

    all_tta_prediction = []
    img_embed, _ = get_img_embedder()

    for fold in tqdm(range(NUM_FOLDS), desc="Generating CV predictions", position=0):
        model_checkpoint_path = f"./{cv_trial_dir}/trial_{fold}.ckpt"
        model = SubmissionModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
        model.eval()
        torch.cuda.empty_cache()

        device = torch.device(device)
        model.to(device)

        test_submission, _, raw_preds = model.get_submissions(dataloader=test_dataloader, get_raw_preds=True)
        if raw_pred_ref is None:
            raw_pred_ref = raw_preds
        cv_predictions.append(raw_preds)
        create_submission_files(submission=test_submission, submission_tag=f"cv_round_{fold}-", dst_dir=cv_dir)

        # TTA is only implemented for Image based competitions, and restrict tta to only image based ones
        if use_tta and img_embed is not None:
            print(f"[START] Generate test predictions using TTA")
            # use random transforms in the test_dataset and take majority vote
            test_dataset.img_input_transform = CustomTrainImageInputTransform
            tta_dir = current_dir / "tta"
            os.makedirs(tta_dir, exist_ok=True)
            all_tta_prediction.append(raw_pred_ref)

            for tta_round in tqdm(range(TTA_ROUNDS), desc="Generating TTA predictions"):
                test_submission, _, raw_preds = model.get_submissions(dataloader=test_dataloader, get_raw_preds=True)
                create_submission_files(
                    submission=test_submission, submission_tag=f"tta_round_{tta_round}-{fold}-", dst_dir=tta_dir
                )

            print(f"[END] Generate test predictions using TTA")

    if all_tta_prediction:
        # construct the submission from tta preds
        tta_preds_mean = np.stack(all_tta_prediction).mean(0)
        submission = generate_submission_transform(preds_mean=tta_preds_mean, pred_ref=raw_pred_ref)
        create_submission_files(submission=submission, submission_tag='cv-tta-', dst_dir=current_dir)

    # construct the submission from cv raw preds
    raw_preds_mean = np.stack(cv_predictions).mean(0)
    submission = generate_submission_transform(preds_mean=raw_preds_mean, pred_ref=raw_pred_ref)
    n_submissions = create_submission_files(submission=submission, submission_tag='cv-', dst_dir=current_dir)
    if n_submissions > 0:
        print(f"[END] Generate cv test predictions in {current_dir}, removing {CV_TRIALS_DIR}")
        remove_trials(CV_TRIALS_DIR)
    else:
        print(f"[END] Failed to create a cva submission in {current_dir}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "0"  # avoid deadlocks in NLP tasks

    parser = ArgumentParser()
    parser.add_argument("--max_total_runtime", type=float, required=True, help="max time for solve.py to run")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default='auto')
    parser.add_argument("--keep_img_shm", default=False, action="store_true")
    args = parser.parse_args()

    if not args.keep_img_shm:
        for ds in (dataset, train_dataset, test_dataset, validation_dataset):
            atexit.register(ds.unload_img_data)

    main(
        training_config=TRAINING_CONFIG,
        max_total_runtime=args.max_total_runtime,
        accelerator=args.accelerator,
        devices=args.devices,
    )
