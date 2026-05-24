import json
import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import warnings
from copy import deepcopy
from typing import Optional, Tuple

from requests.exceptions import RequestsDependencyWarning
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Suppress only the RequestsDependencyWarning
warnings.filterwarnings(
    "ignore", category=RequestsDependencyWarning, message=".*urllib3.*"
)

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
import torch
from ramphy import Hyperparameter
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# RAMP START HYPERPARAMETERS
encoding_mode_llama2vec = Hyperparameter(
    dtype="str",
    default="pos_max",
    values=["pos_max", "extremes", "pca_2", "pca_3", "pca_5"],
)
model_type_llama2vec = Hyperparameter(dtype="str", default="chat", values=["chat"])
# RAMP END HYPERPARAMETERS

IMPUTE_STRATEGY_NUM = "mean"
FILL_VALUE_NUM = -1.0
ENCODING_MODE = str(encoding_mode_llama2vec)
MODEL_TYPE = str(model_type_llama2vec)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def __init__(self):
        self.to_cache = True

        if "pca" in ENCODING_MODE:
            self.scaler: Optional[StandardScaler] = None
            self.pca: Optional[PCA] = None

    def load_llm(self):
        if MODEL_TYPE == "chat":
            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        elif MODEL_TYPE == "classic":
            model_id = "meta-llama/Meta-Llama-3-70B"
        else:
            raise ValueError(f"Model type {{MODEL_TYPE}} not implemented")

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=False, padding_side="left"
        )
        self.tokenizer.add_special_tokens({{"pad_token": "<pad>"}})
        print("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Loading done")

    def preprocess(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        self.load_llm()  # Load it here so we don't load it when caching
        # Find text columns
        feature_types = metadata["data_description"]["feature_types"]
        text_columns = []
        for feature in feature_types:
            if feature_types[feature] == "text":
                text_columns.append(feature)

        print(f"Found the following text columns: {{text_columns}}")

        # Encode columns
        new_feature_names = {{}}  # Needed for metadata update
        all_new_features = []  # Needed for imputing
        for feature in text_columns:
            print(f"Encoding {{feature}}")
            encoded_columns = self.encode_column(column_name=feature, dataset=X_train)
            X_train = pd.concat([X_train, encoded_columns], axis=1)
            new_feature_names[feature] = list(encoded_columns.columns)
            all_new_features += new_feature_names[feature]

            encoded_columns = self.encode_column(column_name=feature, dataset=X_test)
            X_test = pd.concat([X_test, encoded_columns], axis=1)

        # Drop text columns from datasets
        X_train.drop(columns=text_columns, inplace=True)
        X_test.drop(columns=text_columns, inplace=True)

        for new_col in all_new_features:
            X_train, X_test = self.impute_column(
                col_name=new_col, X_train=X_train, X_test=X_test
            )

        # Update metadata
        metadata = deepcopy(metadata)
        for feat in text_columns:
            try:
                metadata["data_description"]["feature_values"].pop(feat)
            except KeyError:
                pass
            try:
                metadata["data_description"]["feature_types"].pop(feat)
            except KeyError:
                pass
            missing_count = None
            if feat in metadata["data_description"]["missing_data_count"]:
                missing_count = metadata["data_description"]["missing_data_count"][feat]
                metadata["data_description"]["missing_data_count"].pop(feat)

            for new_column in new_feature_names[feat]:
                metadata["data_description"]["feature_types"][new_column] = "num"
                if missing_count is not None:
                    metadata["data_description"]["missing_data_count"][new_column] = (
                        missing_count
                    )

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

    def chat_encoding(self, values, column_name, batch_size=512) -> np.ndarray:
        instruction = f"Given the text feature named {{column_name}} of a tabular dataset, extract the corresponding value in the given example."
        queries = [
            [
                {{"role": "system", "content": instruction}},
                {{"role": "user", "content": f"The value of {{column_name}} is {{val}}"}},
            ]
            for val in values
        ]
        input_ids = self.tokenizer.apply_chat_template(
            queries,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        raw_encoding = []
        for batch_idx in tqdm(range(0, len(input_ids), batch_size), desc="Batch idx"):
            with torch.no_grad():
                out = self.model(input_ids[batch_idx : batch_idx + batch_size].cuda())

            raw_encoding.append(out["logits"][:, -1].cpu().numpy())
        raw_encoding = np.concatenate(raw_encoding)
        return raw_encoding

    def classic_encoding(self, values, column_name, batch_size=512) -> np.ndarray:
        queries = [
            f"The value of the column {{column_name}} is {{val}}" for val in values
        ]
        input_ids = self.tokenizer.batch_encode_plus(
            queries, return_tensors="pt", padding=True, truncation=False
        )

        raw_encoding = []
        for batch_idx in tqdm(range(0, len(input_ids), batch_size), desc="Batch idx"):
            batch = input_ids[batch_idx : batch_idx + batch_size]

            with torch.no_grad():
                out = self.model(
                    batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                )

            raw_encoding.append(out["logits"][:, -1].cpu().numpy())
        raw_encoding = np.concatenate(raw_encoding)
        return raw_encoding

    def encode_column(self, column_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
        """This function encodes the given column from the given dataset.

        Args:
            column_name (str): _description_
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: The dataframe of the encoded columns
        """
        column = dataset[
            column_name
        ].dropna()  # We do this and with the idx as well so we only encode the values
        values = column.values
        indexes = column.index

        batch_size = 256
        if MODEL_TYPE == "chat":
            raw_encoding = self.chat_encoding(
                values=values, column_name=column_name, batch_size=batch_size
            )
        elif MODEL_TYPE == "classic":
            raw_encoding = self.classic_encoding(
                values=values, column_name=column_name, batch_size=batch_size
            )
        else:
            raise ValueError(f"Model type {{MODEL_TYPE}} not implemented")

        encoding = self.postprocess_encoding(raw_encoding)
        new_col_names = [f"{{column_name}}_{{idx}}" for idx in range(encoding.shape[1])]
        encoded_columns = pd.DataFrame(
            data=encoding, columns=new_col_names, index=indexes
        )
        return encoded_columns

    def postprocess_encoding(self, raw_encoding: np.ndarray) -> np.ndarray:
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
            encoding = np.array(
                [max_values, norm_max_idx, min_values, norm_min_idx]
            ).swapaxes(1, 0)
            return encoding
        elif "pca" in ENCODING_MODE:
            # This happens when we pass the train dataset
            if self.scaler is None:
                self.scaler = StandardScaler()
                components = int(ENCODING_MODE.split("_")[-1])
                self.pca = PCA(n_components=components)
                scaled_data = self.scaler.fit_transform(raw_encoding)
                encoding = self.pca.fit_transform(scaled_data)
            else:
                scaled_data = self.scaler.transform(raw_encoding)
                encoding = self.pca.transform(scaled_data)
            return encoding
        else:
            raise ValueError(f"Encoding {{ENCODING_MODE}} not implemented")
