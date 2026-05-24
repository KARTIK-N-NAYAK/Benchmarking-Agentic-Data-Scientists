import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.utils import _get_ordinal_encoder, _fix_dtypes, _process_text_na_dataframe

CONTEXT_LENGTH = 10000


def get_dataset_path(dataset_path: str) -> str | None:
    data_paths = list(Path(dataset_path).rglob("problem.py"))

    if not data_paths:
        return None

    return data_paths[-1].parent


def check_necessary_files_exists(dataset_path: str) -> bool:
    if dataset_path is None:
        return False

    data_path = Path(dataset_path)
    train_data_path = data_path / 'data' / 'train.csv'
    test_data_path = data_path / 'data' / 'test.csv'
    metadata_path = data_path / 'data' / 'metadata.json'
    problem_py_path = data_path / 'problem.py'

    required_paths = [train_data_path, test_data_path, metadata_path, problem_py_path]
    if any(not req_path.exists() for req_path in required_paths):
        return False

    return True


def get_data_paths(dataset_path: str) -> tuple[Path, Path, Path]:
    data_path = Path(dataset_path)
    train_data_path = data_path / 'data' / 'train.csv'
    test_data_path = data_path / 'data' / 'test.csv'
    problem_py_path = data_path / 'problem.py'

    return train_data_path, test_data_path, problem_py_path


def write_training_points(output_path: str, num_training_points: int) -> None:
    os.makedirs(output_path, exist_ok=True)
    training_points = os.path.join(output_path, 'training_points.txt')
    if not os.path.exists(training_points):
        with open(training_points, 'w') as f:
            f.write(str(num_training_points))


def prepare_test_set_chunks(dataset: pd.DataFrame | np.ndarray, chunk_size: int) -> list[pd.DataFrame]:
    return [dataset[i:i + chunk_size] for i in range(0, dataset.shape[0], chunk_size)]


def prepare_dataset(
        train_data: pd.DataFrame, id_field: str, target_cols: list, test_sample: pd.DataFrame, context_length: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_data.shape[0] > context_length:
        subsample_ids = get_samples(
            train_data=train_data.copy(),
            id_field=id_field, test_data=test_sample.copy(),
            target_columns=target_cols,
            total_samples=context_length
        )
        train_subsample = train_data[train_data[id_field].isin(subsample_ids)]
    else:
        train_subsample = train_data

    y_train = train_subsample[target_cols]
    x_train = train_subsample.drop(columns=target_cols + [id_field], errors='ignore')

    return x_train, y_train


def get_samples(
        train_data: pd.DataFrame, id_field: str, test_data: pd.DataFrame, target_columns: list, total_samples: int
) -> pd.DataFrame:
    test_data.dropna(inplace=True)
    train_data.dropna(inplace=True)

    train_data_no_id = train_data.drop(columns=[id_field] + target_columns, errors='ignore')
    test_data_no_id = test_data.drop(columns=[id_field] + target_columns, errors='ignore')

    # K-means clustering
    kmeans = KMeans(n_clusters=10, random_state=42).fit(test_data_no_id)
    test_data['cluster'] = kmeans.labels_

    train_data['cluster'] = kmeans.predict(train_data_no_id)
    # Calculate the number of samples to draw from each cluster
    cluster_sizes = train_data['cluster'].value_counts()

    if cluster_sizes.sum() < total_samples:
        samples_per_cluster = cluster_sizes
    else:
        samples_per_cluster = (cluster_sizes / cluster_sizes.sum() * total_samples).astype(int)

    while samples_per_cluster.sum() != total_samples:
        if samples_per_cluster.sum() < total_samples:
            # Add one sample to the smallest cluster that has room for more samples
            valid_clusters = samples_per_cluster[samples_per_cluster < cluster_sizes]
            if not valid_clusters.empty:
                smallest_cluster = valid_clusters.idxmin()
                samples_per_cluster[smallest_cluster] += 1
            else:
                break
        else:
            # Remove one sample from the largest cluster that has more than one sample
            valid_clusters = samples_per_cluster[samples_per_cluster > 1]
            if not valid_clusters.empty:
                largest_cluster = valid_clusters.idxmax()
                samples_per_cluster[largest_cluster] -= 1
            else:
                break
    subsample_ids = train_data.groupby('cluster').apply(
        lambda x: x.sample(n=samples_per_cluster[x.name], random_state=42)[id_field], include_groups=False
    ).reset_index(drop=True)

    return subsample_ids


def get_submission_module(dataset_path: str) -> importlib:
    submission_python_file = os.path.join(dataset_path, "problem.py")
    spec = importlib.util.spec_from_file_location('submission', submission_python_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['submission'] = module
    spec.loader.exec_module(module)
    return module


def create_submission(predictions: list, dataset_path: str, output_path: str) -> None:
    y_pred = np.concatenate(predictions, axis=0)
    module = get_submission_module(dataset_path=dataset_path)
    module.save_submission(y_pred, data_path=dataset_path, output_path=output_path, suffix="test")


def normalize_predictions(batch_probs: np.ndarray, batch_classes: np.ndarray[...], all_classes: list[str],
                          fill_value=1e-9):
    n_samples = batch_probs.shape[0]
    n_total_classes = len(all_classes)

    full_probs = np.full((n_samples, n_total_classes), fill_value)
    class_to_col = {cls: idx for idx, cls in enumerate(all_classes)}
    for batch_idx, cls in enumerate(batch_classes):
        if cls in class_to_col:
            full_probs[:, class_to_col[cls]] = batch_probs[:, batch_idx]
    return full_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_name", type=str, required=True)
    parser.add_argument("--setup_path", type=str, required=True)
    parser.add_argument("--workspace_path", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)

    return parser.parse_args()


def fit_and_predict(
        model: TabPFNClassifier | TabPFNRegressor,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        test_sample: pd.DataFrame,
        is_classification: bool,
        all_classes: list[str]
) -> np.ndarray | list[np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        y_train = np.ravel(y_train)
        if is_classification:
            model.fit(x_train, y_train)
            predictions = model.predict_proba(test_sample)
            batch_classes = model.classes_
            predictions = normalize_predictions(predictions, batch_classes, all_classes)
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(test_sample).reshape(-1, 1)

    return predictions


def get_tabpfn_class(is_classification: bool, **kwargs) -> TabPFNClassifier | TabPFNRegressor:
    if is_classification:
        model_base = TabPFNClassifier(**kwargs)
    else:
        model_base = TabPFNRegressor(**kwargs)

    return model_base


def preprocess_dataset(
        test_data: pd.DataFrame, train_data: pd.DataFrame, is_classification: bool, id_field: str, target_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = train_data[target_cols]

    test_ids = test_data[id_field]
    train_ids = train_data[id_field]
    test_data.drop(columns=[id_field], inplace=True, errors='ignore')
    test_data.drop(columns=target_cols, inplace=True, errors='ignore')
    train_data.drop(columns=target_cols, inplace=True, errors='ignore')
    train_data.drop(columns=[id_field], inplace=True, errors='ignore')

    ord_encoder = _get_ordinal_encoder()
    train_dataset = _fix_dtypes(train_data, cat_indices=None)
    test_dataset = _fix_dtypes(test_data, cat_indices=None)

    columns_train = train_dataset.columns
    columns_test = test_dataset.columns
    train_dataset = _process_text_na_dataframe(train_dataset, ord_encoder=ord_encoder, fit_encoder=True)
    test_dataset = _process_text_na_dataframe(test_dataset, ord_encoder=ord_encoder, fit_encoder=False)
    train_dataset = pd.DataFrame(train_dataset, columns=columns_train)
    test_dataset = pd.DataFrame(test_dataset, columns=columns_test)
    train_dataset[id_field] = train_ids
    train_dataset[id_field] = test_ids

    if is_classification:
        label_encoder_ = LabelEncoder()
        train_dataset[target_cols[0]] = label_encoder_.fit_transform(y_train)
    else:
        train_dataset[target_cols[0]] = y_train
    return train_dataset, test_dataset
