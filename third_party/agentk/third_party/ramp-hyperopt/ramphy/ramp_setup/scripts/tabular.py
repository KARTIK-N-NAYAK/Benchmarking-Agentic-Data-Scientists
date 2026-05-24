import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ramphy as rh
from ramphy import actions as ra
from ramphy import ramp_setup as rs
from ramphy.actions import ramp_action


def create_dummy_targets_and_encode_labels(train_data, test_data, target_cols, prediction_type):
    """Mock test labels if they do not exist and encode train and test labels.

    This is for compatibility between rampwf which expects targets for the test and
    Kaggle challenges for which we do not have the targets for the test:
    """
    # use mean and sigma from training set
    np.random.seed(43)
    if prediction_type == "regression":
        for target_col in target_cols:
            if target_col not in test_data.columns:
                test_data[target_col] = np.random.normal(
                    train_data[target_col].mean(),
                    train_data[target_col].std(),
                    size=len(test_data),
                )
    elif "classification" in prediction_type:
        for target_col in target_cols:
            target_values = list(train_data[target_col].unique())
            new_target_values = list(range(len(target_values)))
            binary_transf = dict(zip(target_values, new_target_values))
            train_data = train_data.replace({target_col: binary_transf})
            test_data = test_data.replace({target_col: binary_transf})
            if target_col not in test_data.columns:
                counts = train_data[target_col].value_counts()
                p = [counts[t] / len(train_data) for t in new_target_values]
                test_data[target_col] = np.random.choice(new_target_values, len(test_data), p=p)

    return train_data, test_data


@ramp_action
def tabular_setup(
    download_dir: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Sets up the kit from the metadata.json, train.csv and test.csv

    Args:
        download_dir (str | Path): Dir where metadata and the data are
        ramp_kit_dir (str | Path, optional): Dir where to put the kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Dir where the kit data will be stored. Defaults to None.
    """

    download_dir = Path(download_dir)
    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)
    problem_f_name = ramp_kit_dir / "problem.py"
    train_data = pd.read_csv(download_dir / "train.csv")
    test_data = pd.read_csv(download_dir / "test.csv")

    metadata = json.load(open(download_dir / "metadata.json"))
    feature_types = metadata["data_description"]["feature_types"]
    target_cols = metadata["data_description"]["target_cols"]
    prediction_type = metadata["prediction_type"]

    if "regression" in prediction_type:
        problem_code = rs.utils.load_template(package=rs, template_path="problems/tabular_regression_problem.py")
    elif "classification" in prediction_type:
        target_values = {target_col: [str(tv) for tv in train_data[target_col].unique()] for target_col in target_cols}
        metadata["data_description"]["target_values"] = target_values
        problem_code = rs.utils.load_template(package=rs, template_path="problems/tabular_classification_problem.py")
    else:
        raise NotImplementedError("Unknown tabular prediction type.")

    # Converting col names to a name that can become a python variable
    # and adding _<i> to col names to avoid clash
    new_feature_types = {}
    for col_i, (col, col_type) in enumerate(feature_types.items()):
        var_col = "_" + re.sub(r"[^a-zA-Z0-9_]", "_", col)
        new_col = f"{var_col}_{col_i}"
        if col == metadata["id_col"]:
            new_feature_types[col] = col_type
        else:
            new_feature_types[new_col] = col_type
    train_data = train_data.rename(columns=dict(zip(feature_types.keys(), new_feature_types.keys())))
    test_data = test_data.rename(columns=dict(zip(feature_types.keys(), new_feature_types.keys())))

    if (
        "feature_types_to_cast" in metadata["data_description"]
        and metadata["data_description"]["feature_types_to_cast"] is not None
    ):
        feature_types_to_cast = metadata["data_description"]["feature_types_to_cast"]
        for old_key, new_key in zip(feature_types.keys(), new_feature_types.keys()):
            if old_key in feature_types_to_cast:
                feature_types_to_cast[new_key] = feature_types_to_cast.pop(old_key)
        metadata["data_description"]["feature_types_to_cast"] = feature_types_to_cast
    else:
        metadata["data_description"]["feature_types_to_cast"] = None

    feature_types = metadata["data_description"]["feature_types"] = new_feature_types

    problem_code = problem_code.format_map(metadata)
    with open(problem_f_name, "w") as f_out:
        f_out.write(problem_code)
    (ramp_data_dir / "data").mkdir(parents=True, exist_ok=True)
    (ramp_kit_dir / "submissions").mkdir(parents=True, exist_ok=True)

    sample_submission_path = download_dir / "sample_submission.csv"
    if sample_submission_path.exists():
        shutil.copy(sample_submission_path, ramp_data_dir / "data" / "sample_submission.csv")

    feature_values = {}
    missing_data_count = {}
    unique_value_count = {}
    for col, col_type in feature_types.items():
        # in some challenges some columns can be in the train and not in the test and
        # vice versa (see for instance https://www.kaggle.com/c/bike-sharing-demand)
        missing_data_count[col] = 0
        if col in train_data.columns:
            missing_data_count[col] += int(train_data[col].isna().sum())
        if col in test_data.columns:
            missing_data_count[col] += int(test_data[col].isna().sum())
        if col_type == "cat" or col_type == "bin":
            # Deleting categoric values from train if not present in test
            # See https://www.kaggle.com/competitions/playground-series-s4e8/discussion/523656
            allowed_vals = test_data[col].unique()
            #            train_data.loc[~train_data[col].isin(allowed_vals), col] = np.nan
            #            test_data.loc[~test_data[col].isin(allowed_vals), col] = np.nan
            print(
                col,
                len(train_data.loc[~train_data[col].isin(allowed_vals), col]),
                len(test_data.loc[~test_data[col].isin(allowed_vals), col]),
            )

            # Ensure the column is treated as string to safely use .str accessor
            try:
                train_data[col] = train_data[col].astype(str)
                train_data[col] = train_data[col].str.strip()
                train_data[col] = train_data[col].replace("nan", "")
                test_data[col] = test_data[col].astype(str)
                test_data[col] = test_data[col].str.strip()
                test_data[col] = test_data[col].replace("nan", "")
                feature_values[col] = sorted(pd.concat([train_data[col], test_data[col]]).dropna().unique().tolist())
            except KeyError as e:
                print(e)
                print("Train data head:")
                print(train_data.head())
                raise
        if col_type in ["num", "cat"]:
            unique_value_count[col] = len(train_data[col].unique())

    metadata["data_description"]["feature_values"] = feature_values
    metadata["data_description"]["missing_data_count"] = missing_data_count
    metadata["data_description"]["unique_value_count"] = unique_value_count
    metadata["n_train"] = len(train_data)
    metadata["n_test"] = len(test_data)

    train_data, test_data = create_dummy_targets_and_encode_labels(train_data, test_data, target_cols, prediction_type)

    test_data.to_csv(ramp_data_dir / "data" / "test.csv", index=False)
    train_data.to_csv(ramp_data_dir / "data" / "train.csv", index=False)

    #    metadata.save(ramp_data_dir)
    json.dump(metadata, open(ramp_data_dir / "data" / "metadata.json", "w"), indent=4)


@ramp_action
def tabular_regression_submit(
    submission: str | Path,
    regressor: str = "xgboost",
    feature_extractor: str = "empty",
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    submission_dir = ramp_kit_dir / "submissions" / submission
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Regressor
    # ---------------------------
    regressor_template_path = Path("workflow_elements") / "tabular_regressors" / f"{regressor}.py"
    regressor_code = rs.utils.load_template(package=rs, template_path=regressor_template_path)
    regressor_code = regressor_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "regressor.py", "w") as f_out:
        f_out.write(regressor_code)
    # ---------------------------

    # Feature Extractor
    # ---------------------------
    fe_template_path = Path("workflow_elements") / "tabular_feature_extractors" / f"{feature_extractor}.py"
    fe_code = rs.utils.load_template(package=rs, template_path=fe_template_path)
    fe_code = fe_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "feature_extractor.py", "w") as f_out:
        f_out.write(fe_code)


@ramp_action
def tabular_classification_submit(
    submission: str | Path,
    classifier: str = "xgboost",
    feature_extractor: str = "empty",
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    submission_dir = ramp_kit_dir / "submissions" / submission
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Regressor
    # ---------------------------
    classifier_template_path = Path("workflow_elements") / "tabular_classifiers" / f"{classifier}.py"
    classifier_code = rs.utils.load_template(package=rs, template_path=classifier_template_path)
    classifier_code = classifier_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "classifier.py", "w") as f_out:
        f_out.write(classifier_code)
    # ---------------------------

    # Feature Extractor
    # ---------------------------
    fe_template_path = Path("workflow_elements") / "tabular_feature_extractors" / f"{feature_extractor}.py"
    fe_code = rs.utils.load_template(package=rs, template_path=fe_template_path)
    fe_code = fe_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "feature_extractor.py", "w") as f_out:
        f_out.write(fe_code)


@ramp_action
def tabular_data_preprocessor_submit(
    submission: str | Path,
    data_preprocessor: str,
    hyper_type: Optional[str] = None,
    hyper_suffix: Optional[str] = None,
    column_types: Optional[list[str]] = None,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> str:
    """Submit data preprocessor.

    Args:
        submission (str | Path): New submission name
        data_preprocessor (str): data preprocessor to submit.
        hyper_type (str, optional): how to add hypers, defaults to None.
        hyper_suffix (str, optional): the name suffix of the hyper, defaults to None.
        column_types (list[str], optional): which column types to include, defaults to None (all).
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Return
        The name of the submitted preprocessor
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / f"{data_preprocessor}.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_code = dp_code.format_map(metadata | {"hyper_suffix": hyper_suffix})
    submission_path = ramp_kit_dir / "submissions" / submission
    wen = f"data_preprocessor_{dp_idx}_{data_preprocessor}"
    cols = metadata["data_description"]["feature_types"].keys()
    if column_types is not None:
        cols = [
            col
            for col in cols
            if metadata["data_description"]["feature_types"][col] in column_types
            and (hyper_suffix != "to_target_encode" or metadata["data_description"]["unique_value_count"][col] < 300)
        ]
    if len(cols) > 0:
        with open(submission_path / f"{wen}.py", "w") as f_out:
            f_out.write(dp_code)
        # Add a selection hyper per column
        if hyper_type == "select_column":
            id_name = metadata["id_col"]
            hs = [
                rh.Hyperparameter(dtype="bool", default=False, values=[False, True], name=f"{col}_{hyper_suffix}")
                for col in cols
                if not col == id_name
            ]
            rh.write_hyperparameters_per_element(submission_path, submission_path, hs, wen)
    return wen


@ramp_action
def tabular_data_preprocessors_submit(
    submission: str | Path,
    data_preprocessors: list[str] = ["drop_id", "col_in_train_only"],
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit data preprocessor.

    Args:
        submission (str | Path): New submission name
        data_preprocessors (list[str], optional): List of data preprocessors to submit. Defaults to ['drop_id', 'col_in_train_only'].
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_names = []
    for dp in data_preprocessors:
        dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / f"{dp}.py"
        dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
        dp_code = dp_code.format_map(metadata)
        dp_name = f"data_preprocessor_{dp_idx}_{dp}"
        dp_names.append(dp_name)
        with open(ramp_kit_dir / "submissions" / submission / f"{dp_name}.py", "w") as f_out:
            f_out.write(dp_code)
        dp_idx += 1
    return dp_names


@ramp_action
def tabular_cat_col_imputers_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit column imputer, a preprocessor that makes sure that no NaN are present in the categorical columns.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "cat_col_imputing.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_names = []
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "cat" and metadata["data_description"]["missing_data_count"][col] > 0:
            dp_code_formatted = dp_code.format_map(metadata | {"col": f"{col}", "str_col": f'"{col}"'})
            dp_name = f"data_preprocessor_{dp_idx}{col}_cat_col_imputing"
            dp_names.append(dp_name)
            with open(ramp_kit_dir / "submissions" / submission / f"{dp_name}.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1
    return dp_names


@ramp_action
def tabular_num_col_imputers_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit column imputer, a preprocessor that makes sure that no NaN are present in the numerical columns.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "num_col_imputing.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_names = []
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "num" and metadata["data_description"]["missing_data_count"][col] > 0:
            dp_code_formatted = dp_code.format_map(metadata | {"col": f"{col}", "str_col": f'"{col}"'})
            dp_name = f"data_preprocessor_{dp_idx}{col}_num_col_imputing"
            dp_names.append(dp_name)
            with open(ramp_kit_dir / "submissions" / submission / f"{dp_name}.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1
    return dp_names


@ramp_action
def tabular_cat_col_encoders_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit categorical column encoders.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "cat_col_encoding.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_names = []
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type in ["cat", "location"]:
            dp_code_formatted = dp_code.format_map(metadata | {"col": f"{col}", "str_col": f'"{col}"'})
            dp_name = f"data_preprocessor_{dp_idx}{col}_cat_col_encoding"
            dp_names.append(dp_name)
            with open(ramp_kit_dir / "submissions" / submission / f"{dp_name}.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1
    return dp_names


@ramp_action
def tabular_text_col_encoders_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit text column encoders.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    # dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "text_col_encoding.py"
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "tokenizer_text2vec.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)

    if "text" in list(metadata["data_description"]["feature_types"].values()):
        dp_name = f"data_preprocessor_{dp_idx}_tokenizer_text_encoding"
        dp_code_formatted = dp_code.format_map(metadata)
        with open(ramp_kit_dir / "submissions" / submission / f"{dp_name}.py", "w") as f_out:
            f_out.write(dp_code_formatted)
        dp_idx += 1
        return [dp_name]
    return []


@ramp_action
def tabular_date_col_encoder_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit datetime column encoders.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    Returns:
        List of the names of the submitted dp
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "date_col_encoding.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_names = []
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "date":
            dp_code_formatted = dp_code.format_map(metadata | {"col": f"{col}", "str_col": f'"{col}"'})
            dp_name = f"data_preprocessor_{dp_idx}{col}_date_col_encoding"
            dp_names.append(dp_name)
            with open(
                ramp_kit_dir / "submissions" / submission / f"{dp_name}.py",
                "w",
            ) as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1
    return dp_names


@ramp_action
def tabular_add_holidays_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit add holidays column if we have a date and country column.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "add_holidays.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    contains_date_type = 0
    contains_location_type = 0
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "date":
            date_col = col
            contains_date_type += 1
        if col_type == "location":
            contains_location_type += 1
            location_col = col

    dp_names = []
    if contains_date_type == 1 and contains_location_type == 1:
        dp_name = f"data_preprocessor_{dp_idx}_{date_col}{location_col}_add_holidays"
        dp_code_formatted = dp_code.format_map(
            metadata | {"date_col": f'"{date_col}"', "location_col": f'"{location_col}"'}
        )
        with open(
            ramp_kit_dir / "submissions" / submission / f"{dp_name}.py",
            "w",
        ) as f_out:
            f_out.write(dp_code_formatted)
        dp_names.append(dp_name)
    return dp_names


def tabular_encoder_imputer_submit(
    submission: str | Path,
    cat_col_impute: bool = True,
    num_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_encode: bool = True,
    date_col_encode: bool = True,
    text_col_encode: bool = True,
    add_holidays: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> List[str]:
    """Submit column encoders and imputers.

    Args:
        submission (str | Path): Submission name
        cat_col_impute (bool, optional): If True appends a cat_col_imputer to the list of preprocessors. Defaults to True.
        num_col_impute (bool, optional): If True appends a num_col_impute to the list of preprocessors. Defaults to True.
        cat_col_encode (bool, optional): If True appends a cat_col_encode to the list of preprocessors. Defaults to True.
        num_col_encode (bool, optional): If True appends a num_col_encode to the list of preprocessors. Defaults to True.
        date_col_encode (bool, optional): If True appends a date_col_encode to the list of preprocessors. Defaults to True.
        text_col_encode (bool, optional): If True appends a text_col_encode to the list of preprocessors. Defaults to True.
        add_holidays (bool, optional): If True appends a add_holidays to the list of
        preprocessors. Defaults to True.
        ramp_kit_dir (str | Path, optional): Path of kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of data dir. Defaults to None.
    Returns:
        List of the names of the submitted dps
    """
    dp_names = []
    if text_col_encode:
        dp_names += tabular_text_col_encoders_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if add_holidays:
        # call before date_col_encode and cat_col_encode which removes
        # the potential date and location columns which are needed to compute holidays
        dp_names += tabular_add_holidays_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if date_col_encode:
        dp_names += tabular_date_col_encoder_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_impute:
        dp_names += tabular_cat_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if num_col_impute:
        dp_names += tabular_num_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_encode:
        dp_names += tabular_cat_col_encoders_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    return dp_names


@ramp_action
def tabular_regression_ordered_submit(
    submission: str | Path,
    regressor: str = "xgboost",
    feature_extractor: str = "empty",
    data_preprocessors: list[str] = [
        "drop_id",
        "drop_columns",
        "col_in_train_only",
        "base_columnwise",
        "rm_constant_col",
    ],
    cat_col_impute: bool = True,
    num_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_encode: bool = True,
    date_col_encode: bool = True,
    text_col_encode: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    strict_order: bool = False,
) -> Dict[str, list]:
    """Make new submission with ordered preprocessors.
    The cat_col encoders and imputers are put in the place where base_columnwise is in the list

    Args:
        submission (str | Path): Submission name
        regressor (str, optional): Regressor. Defaults to 'xgboost'.
        feature_extractor (str, optional): FE. Defaults to 'empty'.
        data_preprocessors (list[str], optional): List of data preprocessor. Defaults to ['drop_id'].
        cat_col_impute (bool, optional): If True appends a cat_col_imputer to the list of preprocessors. Defaults to True.
        num_col_impute (bool, optional): If True appends a num_col_impute to the list of preprocessors. Defaults to True.
        cat_col_encode (bool, optional): If True appends a cat_col_encode to the list of preprocessors. Defaults to True.
        num_col_encode (bool, optional): If True appends a num_col_encode to the list of preprocessors. Defaults to True.
        date_col_encode (bool, optional): If True appends a date_col_encode to the list of preprocessors. Defaults to True.
        ramp_kit_dir (str | Path, optional): Path of kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of data dir. Defaults to None.
    """
    print(submission)
    tabular_regression_submit(
        submission=submission,
        regressor=regressor,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    dp_names = []
    for dp in data_preprocessors:
        if dp == "drop_columns":
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor="drop_columns",
                hyper_type="select_column",
                hyper_suffix="to_drop",
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)
        elif dp == "cat_target_encoding":
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor="cat_target_encoding",
                hyper_type="select_column",
                hyper_suffix="to_target_encode",
                column_types=["cat", "num"],
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)
        elif dp == "base_columnwise":
            dp_names += tabular_encoder_imputer_submit(
                submission=submission,
                cat_col_impute=cat_col_impute,
                num_col_impute=num_col_impute,
                cat_col_encode=cat_col_encode,
                num_col_encode=num_col_encode,
                date_col_encode=date_col_encode,
                text_col_encode=text_col_encode,
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
        elif dp == "rm_constant_col":
            if strict_order:
                raise ValueError(f"{dp} is not in last position!")
            print(f"ATTENTION! - {dp} is not in last position! It will be added last!")
        else:
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor=dp,
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)
    # this needs to be at the end for the constant columns to be used elsewhere
    # location column for instance
    if "rm_constant_col" in data_preprocessors:
        dp_name = tabular_data_preprocessor_submit(
            submission=submission,
            data_preprocessor=dp,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
        dp_names.append(dp_name)
    print(f"Submitted preprocessors: {dp_names}")
    return {"created_submissions": [submission], "submitted_data_preprocessors": dp_names}


@ramp_action
def tabular_classification_ordered_submit(
    submission: str | Path,
    classifier: str = "xgboost",
    feature_extractor: str = "empty",
    data_preprocessors: list[str] = [
        "drop_id",
        "drop_columns",
        "col_in_train_only",
        "base_columnwise",
        "rm_constant_col",
    ],
    cat_col_impute: bool = True,
    num_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_encode: bool = True,
    date_col_encode: bool = True,
    text_col_encode: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    strict_order: bool = False,
) -> Dict[str, list]:
    """Make new submission with ordered preprocessors.
    The cat_col encoders and imputers are put in the place where base_columnwise is in the list

    Args:
        submission (str | Path): Submission name
        classifier (str, optional): Classifier. Defaults to 'xgboost'.
        feature_extractor (str, optional): FE. Defaults to 'empty'.
        data_preprocessors (list[str], optional): List of data preprocessor. Defaults to ['drop_id'].
        cat_col_impute (bool, optional): If True appends a cat_col_imputer to the list of preprocessors. Defaults to True.
        num_col_impute (bool, optional): If True appends a num_col_impute to the list of preprocessors. Defaults to True.
        cat_col_encode (bool, optional): If True appends a cat_col_encode to the list of preprocessors. Defaults to True.
        num_col_encode (bool, optional): If True appends a num_col_encode to the list of preprocessors. Defaults to True.
        date_col_encode (bool, optional): If True appends a date_col_encode to the list of preprocessors. Defaults to True.
        ramp_kit_dir (str | Path, optional): Path of kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of data dir. Defaults to None.
    """
    print(submission)
    tabular_classification_submit(
        submission=submission,
        classifier=classifier,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    dp_names = []
    for dp in data_preprocessors:
        if dp == "drop_columns":
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor="drop_columns",
                hyper_type="select_column",
                hyper_suffix="to_drop",
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)
        elif dp == "cat_target_encoding":
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor="cat_target_encoding",
                hyper_type="select_column",
                hyper_suffix="to_target_encode",
                column_types=["cat", "num"],
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)
        elif dp == "base_columnwise":
            dp_names += tabular_encoder_imputer_submit(
                submission=submission,
                cat_col_impute=cat_col_impute,
                num_col_impute=num_col_impute,
                cat_col_encode=cat_col_encode,
                num_col_encode=num_col_encode,
                date_col_encode=date_col_encode,
                text_col_encode=text_col_encode,
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
        elif dp == "rm_constant_col":
            if strict_order:
                raise ValueError(f"{dp} is not in last position!")
            print(f"ATTENTION! - {dp} is not in last position! It will be added last!")
        else:
            dp_name = tabular_data_preprocessor_submit(
                submission=submission,
                data_preprocessor=dp,
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
            )
            dp_names.append(dp_name)

    # this needs to be at the end for the constant columns to be used elsewhere
    # location column for instance
    if "rm_constant_col" in data_preprocessors:
        dp_name = tabular_data_preprocessor_submit(
            submission=submission,
            data_preprocessor=dp,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
        dp_names.append(dp_name)

    print(f"Submitted preprocessors: {dp_names}")
    return {"created_submissions": [submission], "submitted_data_preprocessors": dp_names}
