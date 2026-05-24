import glob
import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import ramphy as rh
import ramphy.ramp_setup as rs
import rampwf as rw


def submit_foundation_submissions(
    ramp_kit_dir: str,
    base_predictors: list[str],
    workflow: rw.workflows.BaseWorkflow,
    foundation_predictors_dir: str = "best_predictor_arms/hand_selection",
) -> list[str]:
    foundation_submissions = []
    for base_predictor in base_predictors:
        best_arms_path = Path(foundation_predictors_dir) / f"{base_predictor}.csv"
        if best_arms_path.exists():
            predictor_hypers_df = pd.read_csv(best_arms_path)
            submission_path = Path(ramp_kit_dir) / "submissions" / base_predictor
            workflow.set_element_names(submission_path)
            hypers_per_workflow_element = {
                wen: rh.parse_hyperparameters(submission_path, wen) for wen in workflow.element_names
            }
            if "classifier" in workflow.element_names:
                predictor_type = "classifier"
            elif "regressor" in workflow.element_names:
                predictor_type = "regressor"
            else:
                raise ValueError("unknown predictor type, can't find classifier or regressor")
            predictor_hyperparameters = hypers_per_workflow_element[predictor_type]

            for arm_i, arm in predictor_hypers_df.iterrows():
                if not pd.isna(arm["contributivity"]):
                    for h in predictor_hyperparameters:
                        h.default_index = arm[f"{h.name}_i"]
                    all_hyperparameters = []
                    for wen in workflow.element_names:
                        all_hyperparameters += hypers_per_workflow_element[wen]
                    hyper_indices = [h.default_index for h in all_hyperparameters]
                    hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
                    foundation_submission_path = Path(f"{submission_path}_hyperopt_{hyper_hash}")
                    foundation_submission = foundation_submission_path.name
                    foundation_submissions.append(foundation_submission)
                    try:
                        rh.write_hyperparameters(
                            submission_path,
                            foundation_submission_path,
                            hypers_per_workflow_element,
                        )
                    except IndexError as e:
                        print(submission_path)
                        print(arm)
                        raise e
    return foundation_submissions


def foundation_models(
    ramp_kit: str,
    kit_root: str,
    version: str,
    number: str | int,
    n_folds_hyperopt: int = 3,
    n_folds_final_blend: int = 30,
    first_fold_idx: int = 0,
    base_predictors: list[str] = ["lgbm", "xgboost", "catboost", "skmlp"],
    data_preprocessors: list[str] = [
        "drop_id",
        "drop_columns",
        "col_in_train_only",
        "base_columnwise",
        "rm_constant_col",
    ],
    foundation_predictors_dir: str = "best_predictor_arms/hand_selection",
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = Path(kit_root) / f"{ramp_kit}_{kit_suffix}"
    problem = rw.utils.assert_read_problem(str(ramp_kit_dir))
    with open(ramp_kit_dir / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)
    rs.orchestration.submit_base_submissions(
        ramp_kit_dir=ramp_kit_dir,
        metadata=metadata,
        base_predictors=base_predictors,
        data_preprocessors=data_preprocessors,
    )
    foundation_submissions = submit_foundation_submissions(
        ramp_kit_dir=ramp_kit_dir,
        base_predictors=base_predictors,
        workflow=problem.workflow,
        foundation_predictors_dir=foundation_predictors_dir,
    )
    print(f"Got the following foundation submissions: {foundation_submissions}")
    rs.orchestration.train_on_all_folds(
        submissions=foundation_submissions,
        ramp_kit_dir=ramp_kit_dir,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
    )
    rs.orchestration.update_hyperopt_summary(
        base_predictors=base_predictors,
        ramp_kit_dir=str(ramp_kit_dir),
    )
    rs.orchestration.final_blend_then_bag(
        submissions=foundation_submissions,
        ramp_kit_dir=ramp_kit_dir,
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        n_rounds=0,
    )
    rs.orchestration.final_bag_then_blend(
        submissions=foundation_submissions,
        ramp_kit_dir=ramp_kit_dir,
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        n_rounds=0,
    )
    # to prepare for resuming by the hyperopt race
    rs.orchestration.final_blend_then_bag(
        submissions=foundation_submissions,
        ramp_kit_dir=ramp_kit_dir,
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_hyperopt,
        first_fold_idx=first_fold_idx,
        n_rounds=0,
    )
