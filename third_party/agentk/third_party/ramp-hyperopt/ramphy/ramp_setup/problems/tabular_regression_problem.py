import json
import pandas as pd
import rampwf as rw
import ramphy.ramp_setup as rs
from pathlib import Path

# TEMPLATE INPUTS START
score_name = "{score_name}"
target_cols = {data_description[target_cols]}
title = "{title}"
id_col = "{id_col}"
prediction_type = "{prediction_type}"
feature_types_to_cast_dict = {data_description[feature_types_to_cast]}
if feature_types_to_cast_dict is not None:
    feature_types_to_cast = dict()
    for n, t in feature_types_to_cast_dict.items():
        feature_types_to_cast[n] = eval(t)
else:
    feature_types_to_cast = None
# TEMPLATE INPUTS END

problem_title = f"{{title}} tabular {{prediction_type}}"
Predictions = rw.prediction_types.make_regression(
    label_names=target_cols)
workflow = rw.workflows.TabularRegressor()
score_types = [
    rs.score_name_type_map[score_name](name=score_name, precision=4),
]
#get_cv = rw.cvs.GrowingFolds().get_cv
get_cv = rw.cvs.RTimesK().get_cv

def _read_data(path, f_name, data_label, target_cols):
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
    data = pd.read_csv(data_path / f_name, dtype=feature_types_to_cast)
    y_array = data[target_cols].to_numpy()
    if len(y_array.shape) == 1:
        y_array = y_array.reshape((len(y_array), 1))
    X_df = data.drop(target_cols, axis=1)
    return X_df, y_array

def get_train_data(path=".", data_label=None):
    return _read_data(path, "train.csv", data_label, target_cols=target_cols)

def get_test_data(path=".", data_label=None):
    return _read_data(path, "test.csv", data_label, target_cols=target_cols)

def get_metadata(path=".", data_label=None) -> dict:
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
    metadata = json.load(open(data_path / "metadata.json"))
    return metadata

def save_submission(y_pred, data_path=".", output_path=".", suffix="test"):
    if "test" not in suffix:
        df = pd.DataFrame()
#        return  # we don't care about saving the training predictions
    else:
        sample_submission_path = Path(data_path) / "data" / "sample_submission.csv"
        if sample_submission_path.exists():
            df = pd.read_csv(sample_submission_path)
        else:
            test_path = Path(data_path) / "data" / "test.csv"
            df = pd.read_csv(test_path)
            df = df[[id_col]]
    df[target_cols] = y_pred
    output_f_name = Path(output_path) / f"submission_{{suffix}}.csv"
    print(f"Writing submissions into {{output_f_name}}")
    df.to_csv(output_f_name, index=False)
