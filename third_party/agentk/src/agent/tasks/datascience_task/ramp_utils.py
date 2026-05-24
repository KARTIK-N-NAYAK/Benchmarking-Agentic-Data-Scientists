from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import rampwf.score_types as scores

from agent.tools.data_map.map_dataset import TabularOnlyDataset

RAMP_METRICS = [score.lower() for score in scores.__all__]
MAX_ROWS = 3_000
col_types_map = {
    "categorical": "cat", "numerical": "num", "boolean": "bool", "text": "text", "datetime": "date",
    "num": "num", "bool": "bool", "cat": "cat"
}


class DataExtension(Enum):
    CSV = ".csv"
    PKL = ".pkl"
    TXT = ".txt"
    NPY = ".npy"


@dataclass
class DataDescription:
    features: list
    target_cols: list
    description: str
    feature_types: dict[str, str]
    feature_types_to_cast: dict[str, ...]  # to read big dataframes
    positive_target_values: dict[str, str]  # to be compatible with ramp-setup


@dataclass
class MetaData:
    title: str
    kaggle_name: str
    raw_description: str
    task_description: str
    data_description: DataDescription
    score_name: str | None
    metric_description: str
    id_col: str
    positive_class_name: str
    prediction_type: str
    score_path: str | None

    def save(self, save_path: str | Path):
        """Save the metadata as a json

        Args:
            save_path (str | Path): save path
        """
        save_path = Path(save_path)
        metadata_dict = self.asdict()

        json.dump(
            obj=metadata_dict,
            indent=2,
            fp=open(
                save_path / "metadata.json",
                "w",
            ),
        )

    def asdict(self) -> dict:
        """Returns the metadata as a dictionary

        Returns:
            dict: _description_
        """
        metadata_dict = asdict(self)
        for key in metadata_dict:
            if isinstance(metadata_dict[key], DataExtension):
                metadata_dict[key] = metadata_dict[key].value

        if self.score_name not in ["ngini", "auc"]:
            del metadata_dict["positive_class_name"]

        if self.score_name is not None:
            del metadata_dict["score_path"]
        return metadata_dict


def prepare_for_ramp_setup(
        info_path: str | Path,
        data_path: str | Path,
        challenge_name: str,
        output_path: str | Path | None = None,
        post_setup: bool = False,
        original_target: bool = False,
) -> None:
    """This function prepares the train, test, sample_submission and metadata.json needed to run the ramp-setup

    Args:
        info_path (str | Path): Path of the information extracted by the LLM
        data_path (str | Path):  Path of the data downloaded by agent
        challenge_name (str): Name of the challenge
        output_path (Optional[str  |  Path], optional): Path where to save everything. Defaults to None.
        post_setup (bool, optional): Whether the ramp-setup is done in the context of the benchmark post-setup.
                                     This means that some tasks are sub-sampled because their dataset is too large.
        original_target (bool, optional): using original target instead of applying the inverse target transform
    """
    data_path = Path(data_path)
    if output_path is None:
        output_path = data_path
    output_path = Path(output_path)

    column_types_to_cast = make_train_test(
        data_path=data_path,
        output_path=output_path,
        challenge_name=challenge_name,
        post_setup=post_setup,
        original_target=original_target,
    )
    make_metadata(
        info_path=info_path,
        output_path=output_path,
        challenge_name=challenge_name,
        column_types_to_cast=column_types_to_cast,
        data_path=data_path,
        post_setup=post_setup,
    )


def make_train_test(
        data_path: str | Path,
        challenge_name: str,
        output_path: str | Path | None = None,
        post_setup: bool = False,
        original_target: bool = False,
) -> dict[str, ...]:
    """
    This function prepares the train.csv and test.csv necessary to run the ramp-setup
    Note that for some competitions, the dataframes can be very large. This can cause
        pandas to mistakenly load the data as mixed-type because types are inferred
        on chunk as the table is loaded. This will cause issues in RAMP later.
        To prevent, that we can simply specify the type we want to load the problematic
        columns in, so that there are no further issues.
        See https://pandas.pydata.org/docs/reference/api/pandas.errors.DtypeWarning.html

    Args:
        data_path (str | Path): Path of the data downloaded by agent
        output_path (str | Path): Path where to save the data
        post_setup (bool, optional): Whether the ramp-setup is done in the context of the benchmark post-setup.
        original_target (bool, optional): Using original target instead of applying the inverse target transform
    Returns:
        dictionary of column names and their associated cast type
    """
    data_path = Path(data_path)
    if output_path is None:
        output_path = data_path
    output_path = Path(output_path)

    tabular_dataset = TabularOnlyDataset(setup_dir=str(data_path), original_target=original_target)
    train_data = tabular_dataset.get_train_dataset()
    test_data = tabular_dataset.get_test_dataset()

    # get mixed-typed columns
    train_data_columns_to_str = tabular_dataset.get_mixed_typed_columns(train_data)
    test_data_columns_to_str = tabular_dataset.get_mixed_typed_columns(test_data)
    if len(train_data_columns_to_str) == 0 and len(test_data_columns_to_str) == 0:
        column_types_to_cast = None
    else:
        if len(set(test_data_columns_to_str).difference(set(train_data_columns_to_str))) > 0:
            print("Train data mixed type columns: " + '\n\t-' + '\n\t-'.join(train_data_columns_to_str))
            print("Test data mixed type columns: " + '\n\t-' + '\n\t-'.join(test_data_columns_to_str))
            raise ValueError(
                f"Train data has {len(train_data_columns_to_str)} mixed-type columns"
                f" while test data has {len(test_data_columns_to_str)} mixed-type columns"
            )
        column_types_to_cast = {c: 'str' for c in train_data_columns_to_str}  # passing type as string for serialization

    sample_submission = tabular_dataset.get_sample_submission()
    # --- FIX: Always align sample_submission to test_data ids ---
    try:
        if hasattr(tabular_dataset, "id_name") and tabular_dataset.id_name in test_data.columns:
            test_id_col = tabular_dataset.id_name
        elif "Id" in test_data.columns:
            test_id_col = "Id"
        elif "id" in test_data.columns:
            test_id_col = "id"
        else:
            raise ValueError(f"Cannot find id column in test_data. Columns: {list(test_data.columns)[:30]}")

        sub_id_col = sample_submission.columns[0]
        sub_target_cols = list(sample_submission.columns[1:]) if sample_submission.shape[1] > 1 else ["target"]

        aligned = pd.DataFrame({sub_id_col: test_data[test_id_col].values})
        for c in sub_target_cols:
            aligned[c] = 0

        sample_submission = aligned
        print(
            f"[INFO] Aligned sample_submission to test_data: "
            f"test_rows={len(test_data)}, sub_rows={len(sample_submission)}",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] Could not align sample_submission to test_data ids: {e}", flush=True)
    # --- END FIX ---
    print(f"Saving train and test csv at {output_path}")
    # if os.getenv("AGENT_DEBUG", False):
    #     train_data = train_data.iloc[:min(2_000, len(train_data), len(test_data), len(sample_submission))]
    #     test_data = test_data.loc[train_data.index]
    #     sample_submission = sample_submission.loc[train_data.index]

    # tasks with too large datasets might never finish, subsampling their data for faster ramp-setup check
    if post_setup and len(train_data) > MAX_ROWS:
        print(f"Dataset of {challenge_name} is very large, we sub-sample 1k rows", flush=True)
        # sub-sample 10k rows of train data
        train_sub_idx = np.random.choice(train_data.index, size=min(train_data.shape[0], MAX_ROWS), replace=False)
        train_data = train_data.iloc[train_sub_idx]
        # sub-sample 10k rows of test data
        test_sub_idx = np.random.choice(
            sample_submission.index, size=min(test_data.shape[0], sample_submission.shape[0], MAX_ROWS), replace=False
        )
        test_data = test_data.iloc[test_sub_idx]
        sample_submission = sample_submission.iloc[test_sub_idx]

    if not Path.exists(output_path):
        os.makedirs(str(output_path))
    train_data.to_csv(output_path / "train.csv", index=False)
    test_data.to_csv(output_path / "test.csv", index=False)
    sample_submission.to_csv(output_path / "sample_submission.csv", index=False)

    # create results summary csv if not there
    if not (output_path.parent / 'results_summary.csv').exists():
        df = pd.DataFrame(
            columns=[
                'ramp_kit', 'version', 'number', 'kaggle_private_prank_best_public',
                'kaggle_private_prank_best_submission',
                'server', 'run_finished', 'kaggle_finished', 'valid_last_blend', 'valid_growing_folds',
                'valid_mean_lgbm',
                'valid_bagged_lgbm', 'valid_mean_xgboost', 'valid_bagged_xgboost', 'valid_mean_catboost',
                'valid_bagged_catboost', 'kaggle_private_last_blend', 'kaggle_private_growing_folds',
                'kaggle_private_lgbm',
                'kaggle_private_xgboost', 'kaggle_private_catboost', 'kaggle_private_prank_last_blend',
                'kaggle_private_prank_growing_folds', 'kaggle_private_prank_lgbm', 'kaggle_private_prank_xgboost',
                'kaggle_private_prank_catboost', 'kaggle_public_last_blend', 'kaggle_public_growing_folds',
                'kaggle_public_lgbm', 'kaggle_public_xgboost', 'kaggle_public_catboost',
                'kaggle_public_prank_last_blend',
                'kaggle_public_prank_growing_folds', 'kaggle_public_prank_lgbm', 'kaggle_public_prank_xgboost',
                'kaggle_public_prank_catboost', 'contributivity_last_blend_lgbm', 'contributivity_last_blend_xgboost',
                'contributivity_last_blend_catboost', 'contributivity_growing_folds_lgbm',
                'contributivity_growing_folds_xgboost', 'contributivity_growing_folds_catboost',
                'contributivity_bagged_then_blended_lgbm', 'contributivity_bagged_then_blended_xgboost',
                'contributivity_bagged_then_blended_catboost', 'runtime_hyperopt', 'runtime_hyperopt_lgbm',
                'runtime_hyperopt_xgboost', 'runtime_hyperopt_catboost', 'runtime_last_blend', 'runtime_growing_folds',
                'rounds_hyperopt_lgbm', 'rounds_hyperopt_xgboost', 'rounds_hyperopt_catboost',
                'kaggle_private_bagged_then_blended', 'kaggle_public_bagged_then_blended',
                'kaggle_private_prank_bagged_then_blended', 'kaggle_public_prank_bagged_then_blended',
                'valid_bagged_then_blended'
            ]
        )
        df.to_csv(output_path.parent / 'results_summary.csv')

    return column_types_to_cast


def make_metadata(
        info_path: str | Path,
        output_path: str | Path,
        challenge_name: str,
        column_types_to_cast: dict[str, ...] = None,
        data_path: str | Path | None = None,
        post_setup: bool = False,
) -> None:
    """The function that builds the MetaData and saves it from the files stored in path

    Args:
        info_path (str): Path of the information extracted by the LLM
        output_path (str): Path where to save the metadata
        challenge_name (str): Name of the challenge
        column_types_to_cast (dict[str, ...]): map of column names and types for loading big dataframes
        post_setup (bool, optional): Whether the ramp-setup is done in the context of the benchmark post-setup.
    """
    info_path = Path(info_path)
    output_path = Path(output_path)

    # Make data description
    # ---------------------------
    with open(info_path / "metadata" / "submission_names.json", "r") as f:
        infos = json.load(f)

    with open(info_path / "metadata" / "column_types.json", "r") as f:
        column_types = json.load(f)
    if 'id' in column_types:
        column_types.pop('id')

    features = list(column_types.keys())
    # if infos["id_name"] in features:
    #     features.remove(infos["id_name"])
    for target in infos["target_names"]:
        if target in features:
            features.remove(target)
    # ---------------------------

    # Get raw challenge description
    # ---------------------------
    if Path.exists(info_path / "metadata" / "raw_description.txt"):
        with open(info_path / "metadata" / "raw_description.txt", "r") as file:
            raw_description = file.read()
    else:
        raw_description = ""
    # ---------------------------

    # Get task description and type
    # ---------------------------
    if Path.exists(info_path / "metadata" / "task_description.txt"):
        with open(info_path / "metadata" / "task_description.txt", "r") as file:
            task_description = file.read()
    else:
        task_description = ""

    with open(info_path / "metadata" / "task_category.json", "r") as f:
        prediction_type = json.load(f)
        # should be task_type but some seeds might have task_category because of a typo when fixing the parsing
        # as it is not really a big issue I let it check also here in order to prevent this from crashing
        if "task_type" in prediction_type:
            prediction_type = prediction_type["task_type"].lower()
        if "task_category" in prediction_type:
            prediction_type = prediction_type["task_category"].lower()
    # ---------------------------

    # ---------------------------
    # Get metric information
    # ---------------------------
    with open(info_path / "metadata" / "metric_description.txt", "r") as file:
        metric_description = file.read()

    metric_path = None
    if Path.exists(info_path / "metadata" / "metric.json"):
        with open(info_path / "metadata" / "metric.json", "r") as f:
            metric_type = json.load(f)
            metric_type = metric_type["metric"]
    elif Path.exists(info_path / "code_metric.py"):
        metric_path = str(info_path / "code_metric.py")
        metric_type = "accuracy" if "classification" in prediction_type else "rmse"
        print(
            f"Warning! Assigning forcefully metric={metric_type} in a {prediction_type} task by default, "
            f"but please check!", flush=True
        )
    else:
        raise ValueError("Metric not present!")

    # Fix naming for RAMP
    if metric_type == "normalizedgini": metric_type = "ngini"
    if metric_type == "negativeloglikelihood": metric_type = "nll"
    if metric_type == "rocauc": metric_type = "auc"
    if metric_type == "f1micro": metric_type = "f1-micro"
    if prediction_type == "regression" and metric_type == "kappa":
        metric_type = "rmsle"
        print(
            f"Warning! Assigning forcefully metric={metric_type} in a {prediction_type} task by default, "
            f"but please check!", flush=True
        )
    # ---------------------------

    # ---------------------------
    # Get positive class name
    # ---------------------------
    if "binary" in prediction_type:  # or metric_type.lower() in ["auc", "ngini"]:
        try:
            tabular_dataset = TabularOnlyDataset(setup_dir=str(info_path))
            positive_class = tabular_dataset.get_positive_class()
            positive_class = {k: str(v) for k, v in positive_class.items()}
            positive_class_name, positive_class_value = list(positive_class.items())[0]
        except FileNotFoundError as e:
            print(f"prediction_type: {prediction_type}")
            print(f"metric_type: {metric_type}")
            raise e
    elif "multi" in prediction_type:
        # actually positive_target_values (here positive_class) needs to be precisely the empty string for the
        # setup to run correctly. see ./ramp-hyperopt/ramphy/ramp_setup/scripts/tabular.py
        positive_class_name = ""
        positive_class = {}
    else:
        positive_class_name = None
        positive_class = None
    # ----------------------------

    # ----------------------------
    # revert back to classes instead of probabilities for classification tasks with probas targets
    # ----------------------------
    if "classification" in prediction_type:
        train_data = pd.read_csv(output_path / "train.csv")
        test_data = pd.read_csv(output_path / "test.csv")
        train_target_names = set([c for c in train_data.columns if c not in test_data.columns])

        if "multi" in prediction_type:
            # e.g. for multiclass classification, the target_names in infos will be the classes so we need to
            # correct that there will now just be one column with the labels
            infos["target_names"] = [t for t in train_target_names]

        try:
            train_data_float = train_data[list(train_target_names)].values.astype(float)
            if (train_data_float == train_data[list(train_target_names)].values).all():
                make_train_test(
                    data_path=data_path,
                    output_path=output_path,
                    challenge_name=challenge_name,
                    post_setup=post_setup,
                    original_target=True
                )
        except ValueError as e:
            print(f"Target values cannot be converted to float: {e}")
    # ----------------------------

    # ----------------------------
    # Add feature types to metadata
    # ---------------------------
    #  The reason why we do this is because
    #  the code from ramp-hyperopt/ramphy/ramp_setup/problems/tabular_classification_problem.py is loaded as a string
    #  and then the metadata is used to replace f-strings, and it's looking for
    #  metadata[data_description[positive_target_values]]
    with open(info_path / "metadata" / "data_description.txt", "r") as file:
        description = file.read()

    feature_types = {feat: col_types_map[column_types[feat]] for feat in features}
    # Check if columns are really boolean and if they are then cast them to categorical as ramp doesn't support
    #  boolean variables with anything different from 0, 1 of True, False but sometimes we have yes/no or similar...
    train_data = pd.read_csv(output_path / "train.csv")
    for fname, ctype in column_types.items():
        if ctype in ['bool', 'boolean']:
            feature = train_data[fname].dropna()
            feature_values_set = set(feature.values)
            assert len(feature_values_set) < 3, (f"feature {fname} is identified as BOOLEAN but "
                                                 f"has {len(feature_values_set)} values: {feature_values_set}")
            feature_types[fname] = 'cat'

    feature_types.update({infos["id_name"]: 'num'})
    # ---------------------------

    data_description = DataDescription(
        features=features,
        target_cols=infos["target_names"],
        description=description,
        feature_types=feature_types,
        feature_types_to_cast=column_types_to_cast,
        positive_target_values=positive_class,
    )

    metadata = MetaData(
        title=challenge_name,
        kaggle_name=challenge_name,
        raw_description=raw_description,
        task_description=task_description,
        data_description=data_description,
        score_name=metric_type,
        score_path=metric_path,
        metric_description=metric_description,
        id_col=infos["id_name"],
        prediction_type=prediction_type,
        positive_class_name=positive_class_name
    )

    print(f"Saving metadata at: {output_path}")
    if not Path.exists(output_path):
        os.makedirs(str(output_path))
    metadata.save(str(output_path))
