import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter

llm = Hyperparameter(dtype="str", default="fschat/llama-3-8B-Instruct", values=["fschat/llama-3-8B-Instruct"])
max_llm_trials = Hyperparameter(dtype="int", default=5, values=[1, 2, 5, 10])

LLM = str(llm)
MAX_LLM_TRIALS = int(max_llm_trials)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        if "PANGU_PATH" not in os.environ:
            raise ValueError("You should set the PANGU_PATH environment variable.")

        ramp_kit_dir = Path(__file__).parent.parent.parent
        submission_dir = Path(__file__).parent
        llm_workspace_path = submission_dir / "llm_workspace"

        dropped_features_path = llm_workspace_path / "dropped_features.json"

        if not dropped_features_path.exists():
            # We do this because sometimes the LLM fails to generate the file, and relaunching the script makes it work.
            # We still limit the number of trials we have
            trials = 0
            while not dropped_features_path.exists() and trials < MAX_LLM_TRIALS:
                trials += 1
                print(f"Trial {{trials}} for LLM feature drop.")
                rs.pangu_actions.llm_drop_feature(
                    pangu_root=os.environ["PANGU_PATH"], output_path=llm_workspace_path, kit_path=ramp_kit_dir, llm=LLM
                )

            assert (
                llm_workspace_path / "dropped_features.json"
            ).exists(), f"Could not create the droppped_features.json after {{MAX_LLM_TRIALS}} trials."
        else:
            print(
                f"We already have a suggestion of dropped features at {{dropped_features_path}}. \
                Not asking the LLM again. If you want new ones, remove the file."
            )

        with open(dropped_features_path, "r") as f:
            self.drop_feats = json.load(f)["features_to_drop"]

        # Find columns to drop
        # ---------------------------
        # We do this because of the categorical encoders before
        data_columns = X_train.columns
        columns_to_drop = []
        for feat in self.drop_feats:
            columns_to_drop += [s for s in data_columns if s.startswith(feat)]
        # ---------------------------

        X_train = X_train.drop(columns_to_drop, axis=1)
        X_test = X_test.drop(columns_to_drop, axis=1)

        # Purge metadata
        metadata = deepcopy(metadata)
        metadata_elements = ["feature_types", "feature_values", "missing_data_count"]
        for feat in self.drop_feats + columns_to_drop:  # So we are sure to remove everything
            for m_el in metadata_elements:
                try:
                    metadata["data_description"][m_el].pop(feat)
                except KeyError as e:
                    print(f"Feature {{feat}} not in {{m_el}}")
                    continue

        return X_train, y_train, X_test, metadata
