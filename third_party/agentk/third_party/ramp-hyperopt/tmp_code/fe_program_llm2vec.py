import os
from pathlib import Path

import ramphy as rh
import rampwf as rw
from ramphy import ramp_setup as rs

os.environ["PANGU_PATH"] = "/home/gpaolo/pangu-agent"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""

# os.system("proxy")

KITS_ROOT = Path("/home/gpaolo/ramp-kits")
KIT_NAME = "kaggle_synthanic_v1_1_n1"

KIT_PATH = KITS_ROOT / KIT_NAME
model = "lgbm"

NEW_SUBMISSION_NAME = "lgbm_llm2vec"

if __name__ == "__main__":
    rs.tabular_classification_columnwise_last_submit(
        submission="llm2vec",
        classifier=model,
        data_preprocessors=["llm_text2vec"],
        text_col_encode=False,
        ramp_kit_dir=KIT_PATH,
    )

    rh.submit_hybrid(
        ramp_kit_dir=str(KIT_PATH),
        new_submission=NEW_SUBMISSION_NAME,
        parent_submissions={
            "data_preprocessors": "llm2vec",
            ("feature_extractor", "classifier"): "lgbm_hyperopt_best",
        },
    )

    rh.actions.train(ramp_kit_dir=str(KIT_PATH), submission=NEW_SUBMISSION_NAME, fold_idxs=range(900, 931))
