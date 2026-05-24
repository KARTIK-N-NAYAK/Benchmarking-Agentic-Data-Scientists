import json
import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import warnings
from copy import deepcopy
from typing import Optional, Tuple, Dict

from requests.exceptions import RequestsDependencyWarning
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Suppress only the RequestsDependencyWarning
warnings.filterwarnings("ignore", category=RequestsDependencyWarning, message=".*urllib3.*")

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
import torch
from ramphy import Hyperparameter
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

pd.set_option("display.max_columns", None)

# RAMP START HYPERPARAMETERS
encoding_mode_tokenizer = Hyperparameter(
    dtype="str",
    default="pca_5",
    values=["extremes", "pca_1", "pca_3", "pca_5", "pca_max"],
)
# RAMP END HYPERPARAMETERS

IMPUTE_STRATEGY_NUM = "mean"
FILL_VALUE_NUM = -1.0
ENCODING_MODE = str(encoding_mode_tokenizer)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def __init__(self, col={str_col}):
        self.to_cache = True
        self.col = col

        if "pca" in ENCODING_MODE:
            self.scaler: Dict[str, StandardScaler] = {{}}
            self.pca: Dict[str, PCA] = {{}}

    def load_tokenizer(self):
        model_id = "meta-llama/Meta-Llama-3-8B"

        print("Loading tokenizer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False, padding_side="left")
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, padding_side="left")
        self.tokenizer.add_special_tokens({{"pad_token": "<pad>"}})
        print("Loading done")

    def preprocess(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        self.load_tokenizer()  # Load it here so we don't load it when caching

        # We store the idx so we can concat by ignoring them
        train_len = X_train.shape[0]
        train_idx = X_train.index
        test_idx = X_test.index
        all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

        # Encode columns
        encoded_columns = self.encode_column(column_name=self.col, dataset=all_data)
        all_data = pd.concat([all_data, encoded_columns], axis=1)
        new_feature_names = list(encoded_columns.columns)

        # Drop text columns from datasets
        all_data.drop(columns=self.col, inplace=True)
        # We split and restore the original inputs
        X_train = all_data[:train_len]
        X_train.index = train_idx
        X_test = all_data[train_len:]
        X_test.index = test_idx

        for new_col in new_feature_names:
            X_train, X_test = self.impute_column(col_name=new_col, X_train=X_train, X_test=X_test)

        # Update metadata
        metadata = deepcopy(metadata)
        try:
            metadata["data_description"]["feature_values"].pop(self.col)
        except KeyError:
            pass
        try:
            metadata["data_description"]["feature_types"].pop(self.col)
        except KeyError:
            pass
        try:
            metadata["data_description"]["features"].pop(self.col)
        except KeyError:
            pass
        missing_count = None
        if self.col in metadata["data_description"]["missing_data_count"]:
            missing_count = metadata["data_description"]["missing_data_count"][self.col]
            metadata["data_description"]["missing_data_count"].pop(self.col)

        for new_column in new_feature_names:
            metadata["data_description"]["feature_types"][new_column] = "num"
            if missing_count is not None:
                metadata["data_description"]["missing_data_count"][new_column] = missing_count

        return X_train, y_train, X_test, metadata

    def impute_column(
        self, col_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Imputes the missing values in the column

        Args:
            col_name (str): _description_
            X_train (pd.DataFrame): _description_
            X_test (pd.DataFrame): _description_

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        imputer = SimpleImputer(strategy=IMPUTE_STRATEGY_NUM, fill_value=FILL_VALUE_NUM)
        imputer.fit(pd.concat([X_train, X_test])[[col_name]])
        X_train[[col_name]] = imputer.transform(X_train[[col_name]])
        X_test[[col_name]] = imputer.transform(X_test[[col_name]])
        return X_train, X_test

    def encode_column(self, column_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
        """This function encodes the given column from the given dataset.

        Args:
            column_name (str): _description_
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: The dataframe of the encoded columns
        """
        column = dataset[column_name].dropna()  # We do this and with the idx as well so we only encode the values
        values = list(column.values)
        indexes = column.index
        raw_encoding = self.tokenizer.batch_encode_plus(values, return_attention_mask=False, padding="longest")
        raw_encoding = np.array(raw_encoding["input_ids"])

        encoding = self.postprocess_encoding(raw_encoding, column_name)
        new_col_names = [f"{{column_name}}_{{idx}}" for idx in range(encoding.shape[1])]
        encoded_columns = pd.DataFrame(data=encoding, columns=new_col_names, index=indexes)
        return encoded_columns

    def postprocess_encoding(self, raw_encoding: np.ndarray, column_name: str) -> np.ndarray:
        """This function postprocesses the raw encoding from the LLM

        Args:
            raw_encoding (_type_): Array of the raw encoding
        """
        if ENCODING_MODE == "full":
            # We just return the full raw encoding
            return raw_encoding
        elif ENCODING_MODE == "pos_max":
            # We select the max and return [normalized_max_position, max]
            normalized_idx = np.argmax(raw_encoding, axis=1) / raw_encoding.shape[1]
            max_values = np.max(raw_encoding, axis=1)
            encoding = np.array([max_values, normalized_idx]).swapaxes(1, 0)
            return encoding
        elif ENCODING_MODE == "extremes":
            norm_max_idx = np.argmax(raw_encoding, axis=1) / raw_encoding.shape[1]
            norm_min_idx = np.argmin(raw_encoding, axis=1) / raw_encoding.shape[1]
            max_values = np.max(raw_encoding, axis=1)
            min_values = np.min(raw_encoding, axis=1)
            encoding = np.array([max_values, norm_max_idx, min_values, norm_min_idx]).swapaxes(1, 0)
            return encoding
        elif ENCODING_MODE == "pca_max":
            if column_name not in self.scaler:
                self.scaler[column_name] = StandardScaler()
                components = min(raw_encoding.shape)
                self.pca[column_name] = PCA(n_components=components)
                scaled_data = self.scaler[column_name].fit_transform(raw_encoding)
                encoding = self.pca[column_name].fit_transform(scaled_data)
            else:
                scaled_data = self.scaler[column_name].transform(raw_encoding)
                encoding = self.pca[column_name].transform(scaled_data)
            return encoding
        elif "pca" in ENCODING_MODE:
            # This happens when we pass the train dataset
            # We use the dict, cause each column might be encoded with tensors of diff lenghts, depending on the len of the longest word
            if column_name not in self.scaler:
                self.scaler[column_name] = StandardScaler()
                components = int(ENCODING_MODE.split("_")[-1])
                self.pca[column_name] = PCA(n_components=components)
                scaled_data = self.scaler[column_name].fit_transform(raw_encoding)
                encoding = self.pca[column_name].fit_transform(scaled_data)
            else:
                scaled_data = self.scaler[column_name].transform(raw_encoding)
                encoding = self.pca[column_name].transform(scaled_data)
            return encoding
        else:
            raise ValueError(f"Encoding {{ENCODING_MODE}} not implemented")
