import glob
import json
import shutil
import numpy as np
import pandas as pd
import rampwf as rw
import ramphy as rh
import ramphy.ramp_setup as rs
from pathlib import Path

def blend_at_round(
    ramp_kit,
    version,
    number,
    n_folds_final_blend,
    n_folds_hyperopt,
    first_fold_idx,
    round_idx,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"
    print(ramp_kit_dir)

    stop_fold_idx = first_fold_idx + n_folds_final_blend
    bagged_then_blended_f_name = (
        Path(ramp_kit_dir)
        / "final_test_predictions"
        / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{round_idx}.csv"
    )
    last_blend_f_name = (
        Path(ramp_kit_dir)
        / "final_test_predictions"
        / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{round_idx}.csv"
    )
    if bagged_then_blended_f_name.exists() and last_blend_f_name.exists():
        print(f"These files exist, delete them if you want to re-blend:")
        print(bagged_then_blended_f_name)
        print(last_blend_f_name)
        return
        
    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
    problem = rw.utils.assert_read_problem(ramp_kit_dir=ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    score_type = problem.score_types[-1]
    valid_score_name = f"valid_{score_names[-1]}"
    
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rh.actions.load_ramp_action(action_f_name))
    
    hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
    blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
    train_actions = [ra for ra in ramp_program if ra.name == "train"]

    n_rounds = 0
    for blend_action in blend_actions:
        if blend_action.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt):
            n_rounds += 1
            last_race_blend_action = blend_action
            if round_idx > 0 and n_rounds >= round_idx:
                break
    if round_idx > 0 and n_rounds < round_idx:
        print(f"round_idx = {round_idx} > n_rounds = {n_rounds}")
        return
    last_race_blended_submissions = [key for key, value in last_race_blend_action.contributivities.items() if value > 0]
    print(last_race_blend_action.__dict__)

    rs.orchestration.train_on_all_folds(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        n_folds_final_blend = n_folds_final_blend,
        first_fold_idx = first_fold_idx,
    )
    # Blend then bag the final blend of the hyperopt race on all the folds
    rs.orchestration.final_blend_then_bag(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds_final_blend = n_folds_final_blend,
        first_fold_idx = first_fold_idx,
        n_rounds = n_rounds,
        round_datetime = last_race_blend_action.start_time
    )
    # Bag then blend the final blend of the hyperopt race on all the folds
    rs.orchestration.final_bag_then_blend(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds_final_blend = n_folds_final_blend,
        first_fold_idx = first_fold_idx,
        n_rounds = n_rounds,
        round_datetime = last_race_blend_action.start_time
    )
