"""
This script creates the transform functions for the tabular training targets.
"""
import json
from typing import Any, Union, Tuple

import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

# Read target tab file
df_train_target = pd.read_csv('./train_tab_target_map.csv')

# Create onehot encoder for transforming the target
class_names_columns_regression = [col for col in df_train_target.columns if col.endswith('_regression')]
class_names_columns_classification = [col for col in df_train_target.columns if col.endswith('_classification')]
class_names_data_classification = df_train_target[class_names_columns_classification].values
if len(class_names_columns_classification) > 0:
    enc = OneHotEncoder(handle_unknown='ignore', dtype=np.bool_)
    enc.fit(class_names_data_classification)
    for col, cat in zip(class_names_columns_classification, enc.categories_):
        if len(cat) > 0.5 * class_names_data_classification.shape[0]:
            raise ValueError(f"Trying to one-hot encode {len(class_names_columns_classification)} classification "
                             f"columns, but at least one column has {len(cat)} distinct values. Are you sure "
                             f"the column {col} is a categorical column?")
else:
    enc = None

# Read the target_columns_transform
with open('target_columns_transform.json') as json_file:
    target_columns_transform = json.load(json_file)


def tab_target_transform(original_target: pd.DataFrame) -> pd.DataFrame:
    """
    Transform for tabular targets, maps original submission format to usable numerical format.
    It cannot return `None`, it has to at least return `original_target` if no transform is needed.
    """
    # Filter regression columns
    transformed_target = original_target.filter(like='_regression', axis=1)
    if enc is not None:
        # Convert class labels to one-hot encodings
        class_name = list(enc.get_feature_names_out(class_names_columns_classification))
        x = original_target[class_names_columns_classification].values
        onehot_class = enc.transform(x)

        # Make sure the index matches the index the of original_target for concat to work correctly
        df_classification = pd.DataFrame(onehot_class.toarray(), columns=class_name, index=original_target.index)

        # Append to regression targets
        transformed_target = pd.concat([df_classification, transformed_target], axis=1)

    transformed_target.insert(loc=0, column='id', value=original_target['id'].values)
    return transformed_target


def onehot_to_classname(
        onehot: np.array,
        probabilities: np.array,
        classification_threshold: float,
        unknown_classname: str | None
):
    """
    This function converts a one-hot encoded array back to class names, and replaces classes with probabilities below a
    threshold with a specified 'unknown' class name.

    Parameters:
    onehot (np.array): A one-hot encoded numpy array representing class memberships.
    probabilities (np.array): A numpy array of class probabilities corresponding to the classes in 'onehot'.
    classification_threshold (float): A threshold for class probabilities. Classes with probabilities below this threshold are considered 'unknown'.
    unknown_classname (str): A string to replace the class names of 'unknown' classes (i.e., classes with probabilities below the threshold).

    Returns:
    class_names_array (np.array): A numpy array of class names. 'Unknown' classes have been replaced with 'unknown_classname'.
    """

    class_names_array = enc.inverse_transform(onehot)
    class_names_array = class_names_array[class_names_array != np.array(None)]

    if unknown_classname:
        unknown_classes_idx = np.argwhere(probabilities < classification_threshold)
        class_names_array = class_names_array.astype(object)
        class_names_array[unknown_classes_idx] = unknown_classname

    return class_names_array


def extract_values_top_k(input_string: str) -> Tuple[int, str]:
    """
    Extracts values between parentheses and the comma.

    Args:
        input_string (str): A string containing values in the format '(x, y)'.

    Returns:
        list: A list containing the extracted values.

    Example:
        extract_values('(5, new_whale)')
        ['5', 'new_whale']
    """
    result = re.findall(r'[^,()]+', input_string)
    top_k = int(result[0])
    unknown_class_name = result[1].replace(' ', '')
    unknown_class_name = None if 'none' in unknown_class_name.lower() else unknown_class_name
    return top_k, unknown_class_name


def process_classification_target(
        classification_probits: np.array,
        columns_target_map: dict,
        columns_class_name: list,
        classification_threshold: float = 1e-5

) -> pd.DataFrame:
    """
    This function processes a classification target by grouping labels, extracting features, and creating a DataFrame.

    Parameters:
    classification_target (np.array): A numpy array representing the classification probabilities.
    columns_target_map (dict): A dictionary mapping column names to targets.
    columns_class_name (list): A list of class names for the columns.
    classification_threshold (float, optional): A threshold for classification. Default is 1e-5.

    Returns:
    df_classification (pd.DataFrame): A pandas DataFrame that contains the processed classification target.
    """
    groups_label = enc.categories_
    idx_start = 0
    df_classification = pd.DataFrame()

    for group in groups_label:

        # Extract the label columns names and probits of the current classification group
        label_cols_names = columns_class_name[idx_start:idx_start + len(group)]
        group_probits = classification_probits[:, idx_start:idx_start + len(group)]

        # check that probits are indeed probits
        assert np.isclose(group_probits.sum(axis=-1), 1.0).all(), \
            f"Probits of group {group} do not sum to 1! Did you forget to apply softmax?"

        # Extract the feature column name
        feature_column_name = label_cols_names[0][:label_cols_names[0].rfind("_classification")] + "_classification"

        if columns_target_map[feature_column_name] == "proba":
            df = pd.DataFrame(group_probits, columns=label_cols_names)

        else:
            raw_top_k = columns_target_map[feature_column_name].split('top_k')[1]
            top_k, unknown_classname = extract_values_top_k(input_string=raw_top_k)
            onehot_group, proba_group = get_topk_onehot(probits=group_probits, k=top_k)

            # Format the onot hot to match shape of the onehot encoder
            onehot = np.zeros((classification_probits.shape[0] * top_k, classification_probits.shape[1]))
            onehot[:, idx_start:idx_start + len(group)] = onehot_group

            class_names_array = onehot_to_classname(
                onehot=onehot,
                probabilities=proba_group,
                classification_threshold=classification_threshold,
                unknown_classname=unknown_classname,
            )

            class_names_array = class_names_array.reshape(classification_probits.shape[0], top_k)

            if top_k > 1:
                class_names_array = pd.Series(list(class_names_array))

            df = pd.DataFrame(class_names_array, columns=[feature_column_name])

        df_classification = pd.concat([df_classification, df], axis=1)

        idx_start += len(group)

    return df_classification


def get_topk_onehot(
        probits: np.array,
        k: int = 1,
) -> np.array:
    """
    This function returns the top 'k' values and their indices from the input array 'probits'.

    Parameters:
    probits (np.array): A numpy array from which to select the top 'k' values.
    k (int, optional): The number of top values to select. Default is 1.

    Returns:
    onehot (np.array): A one-hot encoded numpy array of shape (probits.shape[0] * k, probits.shape[1]).
                       The 'i'th row of 'onehot' corresponds to the 'i'th top value in 'probits'.
    values (np.array): A flattened numpy array of the top 'k' values in 'probits'.
    """
    values, indices = torch.topk(torch.Tensor(probits), k)
    onehot = np.zeros((probits.shape[0] * k, probits.shape[1]))
    onehot[np.arange(onehot.shape[0]), indices.flatten().numpy()] = 1

    return onehot, values.flatten().numpy()


def tab_target_inverse_transform(target_values: np.ndarray, ids: Union[list[Any], np.array]) -> pd.DataFrame:
    """
    Inverse transform for tabular targets, maps back to original submission format.
    It cannot return `None`, it has to at least return the `transformed_target` if no inverse transform is needed.
    Args:
        target_values: batch of target values
        ids: list of the entry ids
    """
    if enc is None:
        regression_target = target_values
    else:
        columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))
        classification_target = target_values[:, :len(columns_class_name)]
        regression_target = target_values[:, len(columns_class_name):]

    df_regression_target = pd.DataFrame(regression_target, columns=class_names_columns_regression)
    if enc is None:
        df_regression_target.insert(0, "id", ids)
        return df_regression_target

    # We assume that if we are here, it's because at least one target column is a classification,
    # otherwise, if the task is purely a classification task, we probably shouldn't end up here ...
    df_transformed_classification_target = process_classification_target(
        classification_probits=classification_target,
        columns_target_map=target_columns_transform,
        columns_class_name=columns_class_name,
    )

    transformed_target = pd.concat([df_transformed_classification_target, df_regression_target], axis=1)
    transformed_target.insert(0, "id", ids)

    for pattern in ["_regression", "_classification"]:
        transformed_target.rename({c: c.replace(pattern, "") for c in transformed_target.columns}, axis=1, inplace=True)
    return transformed_target
