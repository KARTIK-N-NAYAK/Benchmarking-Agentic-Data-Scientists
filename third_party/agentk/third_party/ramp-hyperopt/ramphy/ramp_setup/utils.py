import glob
import re
import rampwf as rw
from pathlib import Path
from typing import Tuple, Optional, List
import configparser
import ast

try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources

score_name_type_map = {
    # regression
    "rmse": rw.score_types.RMSE,
    "mae": rw.score_types.MAE,
    "r2": rw.score_types.R2,
    "rmsle": rw.score_types.RMSLE,
    "medae": rw.score_types.MedAE,
    "smape": rw.score_types.SMAPE,
    # classification
    "auc": rw.score_types.ROCAUC,
    "gini": rw.score_types.Gini,
    "ngini": rw.score_types.NormalizedGini,
    "accuracy": rw.score_types.Accuracy,
    "nll": rw.score_types.NegativeLogLikelihood,
    "f1-micro": rw.score_types.F1Micro,
    # "kappa": rw.score_types.LogLikelihood,  # bleding with kappa doesn't seem to work
    "kappa": rw.score_types.Kappa,  # TODO REVERT ME
    "mcc": rw.score_types.MatthewsCorrcoef,
}


def load_template(package, template_path) -> str:
    """Loads a template from the package

    Args:
        package (_type_): Package providing the template
        template_path (_type_): Template path

    Returns:
        _type_: The code of the template
    """
    templates_dir = impresources.files(package)
    try:
        template_f_name = templates_dir / template_path
        with template_f_name.open("r") as f:
            template_code = f.read()
    except AttributeError:
        # Python < PY3.9, fall back to method deprecated in PY3.11.
        template_code = impresources.read_text(package / template_path)
    return template_code


def num_data_preprocessors(submission, ramp_kit_dir):
    submission_path = ramp_kit_dir / "submissions" / submission
    dp_idx = 0
    while True:
        submissions_f_names = glob.glob(f"{submission_path}/data_preprocessor_{dp_idx}_*.py")
        if len(submissions_f_names) == 0:
            break
        dp_idx += 1
    return dp_idx


def save_config(save_path: Path, config_name: str = "config.ini", **kwargs):
    with open(save_path / config_name, "w") as configfile:
        for key, value in kwargs.items():
            if isinstance(value, str):
                configfile.write(f"{key} = '{value}'\n")
            else:
                configfile.write(f"{key} = {value}\n")

    print(f"Config file saved at {save_path / config_name}")


def load_config(load_path: Path, config_name: str = "config.ini") -> dict:
    config = configparser.ConfigParser()
    with open(load_path / config_name, "r") as configfile:
        # Add a dummy section header
        file_content = "[dummy_section]\n" + configfile.read()
    config.read_string(file_content)
    config_dict = {}
    for key, value in config["dummy_section"].items():
        try:
            # Try to evaluate the value to handle lists, dicts, etc.
            config_dict[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If evaluation fails, keep the value as a string
            config_dict[key] = value
    return config_dict


def get_full_preprocessor_name(data_preprocessor: str, submitted_preprocessors: List[str]):
    if data_preprocessor == "base_columnwise":
        full_names = []
        for dp in submitted_preprocessors:
            print(dp)
            # Imputers
            if "cat_col_imputing" in dp:
                full_names.append(dp)
            if "num_col_imputing" in dp:
                full_names.append(dp)
            # Encoders
            if "cat_col_encoding" in dp:
                full_names.append(dp)
            if "num_col_encoding" in dp:
                full_names.append(dp)
            if "text_col_encoding" in dp:
                full_names.append(dp)
            if "date_col_encoding" in dp:
                full_names.append(dp)
    else:
        full_names = [dp for dp in submitted_preprocessors if data_preprocessor in dp]
    return full_names
