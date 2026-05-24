import glob
import json
import random
import shutil
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import ramphy as rh
import ramphy.ramp_setup as rs
import rampwf as rw


def last_action(
    ramp_kit_dir: str,
    name: str,
    fold_idxs: range = None,
) -> rh.actions.RampAction | None:
    """Last action of a given action name."""
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort(reverse=True)
    for i in range(len(action_f_names)):
        ramp_action_object = rh.actions.load_ramp_action(Path(action_f_names[i]))
        if ramp_action_object.name == name:
            if fold_idxs is None or ramp_action_object.kwargs["fold_idxs"] == fold_idxs:
                return ramp_action_object
    return None


@rh.actions.ramp_action
def submit_final_test_predictions(
    submission_source_f_name: str,
    submission_target_f_name: str,
    ramp_kit_dir: str,
):
    """Copy a submission file into a submission folder.

    Typically into <ramp_kit_dir>/final_test_predictions.
    The main function of this is to save the action so we can recover what was
    submitted.
    """
    shutil.copy(submission_source_f_name, submission_target_f_name)


def run_race(
    base_predictors: list[str],
    action_stats: dict,
    ramp_kit_dir: str,
    kit_suffix: str,
    metadata: dict,
    n_rounds: int,
    n_trials_per_round: int,
    patience: int,
    n_folds_hyperopt: int,
    n_folds_final_blend: int,
    first_fold_idx: int,
    start_round: int,
    scores: list[float],
    is_lower_the_better: bool,
    contributivity_floor: int,
    blended_submissions: set[str],
    max_time: float,
    elapsed_time: float,  # in hours
    preprocessors_to_hyper: Optional[list[str]] = None,
    n_cpu_per_run: int = None,
) -> set[str]:
    # can be deleted, once the algorithm settles
    improvement_speed_df = pd.DataFrame(columns=["round"] + base_predictors)

    if "regression" in metadata["prediction_type"]:
        predictor_we_name = "regressor"
    elif "classification" in metadata["prediction_type"]:
        predictor_we_name = "classifier"

    # NOTE if you want these now you can add them with the additional_preprocessors cli flag and
    # then select for optim with the preprocessors_to_hyperopt
    # base_we_names = [predictor_we_name, "data_preprocessor_1_drop_columns", "data_preprocessor_2_cat_target_encoding"]

    # The WEs the orchestrator needs to choose from in every round
    base_we_names = [predictor_we_name]
    if preprocessors_to_hyper is not None:
        base_we_names += preprocessors_to_hyper
    round_datetime = None
    for round_idx in range(start_round, n_rounds):
        if patience >= 0 and len(scores) > patience:
            if is_lower_the_better:
                print(f"Best occured {len(scores) - scores.index(min(scores))} rounds ago.")
                if min(scores[:-patience]) <= min(scores):
                    break
            else:
                print(f"Best occured {len(scores) - scores.index(max(scores))} rounds ago.")
                if max(scores[:-patience]) >= max(scores):
                    break
        # We'll choose the submission that improves the results the fastest
        improvement_speeds = {}
        for predictor in base_predictors:
            ast = action_stats[predictor]
            # sum of contributivities, applying this action
            if len(ast) >= 2:
                last_ast = ast[-1]
                total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                contributivity = last_ast["contributivities"][predictor]
                speed = contributivity / total_time
            else:
                speed = np.finfo(float).max
            improvement_speeds[predictor] = speed
        print(f"improvement_speeds:\n{improvement_speeds}")
        row = pd.DataFrame(dict({"round_idx": round_idx}, **improvement_speeds), index=[round_idx])
        improvement_speed_df = pd.concat([improvement_speed_df, row], ignore_index=True)

        max_speed = max(improvement_speeds.values())
        best_predictors = [predictor for predictor, speed in improvement_speeds.items() if speed == max_speed]
        predictor = random.choice(best_predictors)  # in case of tie (at zero typically), or eps greedy, random choice
        print(f"best_predictors: {best_predictors}")
        print(f"selected predictor : {predictor}")
        #    input("Press Enter to continue...")
        n_trials = n_trials_per_round * n_folds_hyperopt
        wes_to_hyperopt = random.sample(base_we_names, 1)
        print(f"Workflow elements to choose from: {base_we_names}")
        print(f"Choosen workflow element: {wes_to_hyperopt[0]}")
        submission_to_hyperopt = predictor  # default: hyperopt one of the base submissions
        blended_submissions_of_predictor = [
            submission for submission in blended_submissions if submission[:-20] == predictor
        ]
        print(f"Submissions to choose from: {blended_submissions_of_predictor}")
        if len(blended_submissions_of_predictor) > 0:
            submission_to_hyperopt = random.choice(blended_submissions_of_predictor)
        print(f"Choosen submission: {submission_to_hyperopt}")
        rh.actions.hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission_to_hyperopt,
            workflow_element_names=wes_to_hyperopt,
            n_trials=n_trials,
            fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_hyperopt),
            resume=True,
            subtract_existing=False,
            n_cpu_per_run=n_cpu_per_run
        )
        hyperopt_action = last_action(ramp_kit_dir, "hyperopt")
        elapsed_time += hyperopt_action.runtime.total_seconds() / 3600
        if hyperopt_action is None:
            continue
        elif len(hyperopt_action.mean_scores) > 0:
            # Add the best submission from this round of hyperopt to the set to be blended
            for hyperopt_submission, mean_score in hyperopt_action.mean_scores.items():
                # Add best
                # if mean_score == hyperopt_action.mean_score:
                blended_submissions.add(hyperopt_submission)

            # The blended score improvement is wrt the previous blended score. If it doesn't exist
            # (in the first iteration, or if no submission was blended for a reason) use the mean
            # score.
            previous_blend_action = last_action(ramp_kit_dir, "blend", fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_hyperopt))
            rh.actions.blend(
                ramp_kit_dir=ramp_kit_dir,
                submissions=list(blended_submissions),
                fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_hyperopt),
            )
            blend_action = last_action(ramp_kit_dir, "blend", fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_hyperopt))
            round_datetime = blend_action.start_time
            elapsed_time += blend_action.runtime.total_seconds() / 3600
            if hasattr(blend_action, "blended_score"):
                blended_score = blend_action.blended_score
                contributivities = {
                    s: contributivity_floor
                    + np.array([c for sh, c in blend_action.contributivities.items() if sh[: len(s)] == s]).sum()
                    for s in base_predictors
                }
            else:
                print("something wrong: no blended score")
                raise RuntimeError("something wrong: no blended score")
                blended_score = hyperopt_action.mean_score
                contributivities = {s: 1000 / len(predictors) for s in base_predictors}
            scores.append(blended_score)
            action_stats[predictor].append(
                {
                    "runtime": hyperopt_action.runtime,
                    "contributivities": contributivities,
                }
            )

            # Delete submissions with zero contributivity from the submissions to be blended
            contributivities_df = rh.actions.load_contributivities(ramp_kit_dir)
            non_contributive_submissions = set(contributivities_df[contributivities_df["contributivity"] == 0].index)
            print(f"blended submissions: {blended_submissions}")
            print(f"non-contributive submissions: {non_contributive_submissions}")
            blended_submissions = blended_submissions - non_contributive_submissions
            print(f"new blended submissions: {blended_submissions}")
            if max_time > 0:
                estimated_runtime_for_final_blend = 0
                for submission in blended_submissions:
                    scores_df = pd.read_csv(f"{ramp_kit_dir}/submissions/{submission}/training_output/fold_{first_fold_idx}/scores.csv")
                    estimated_runtime_for_final_blend += scores_df["time"].sum()
                estimated_runtime_for_final_blend *= (n_folds_final_blend - n_folds_hyperopt) / 3600
                estimated_final_blending_time = 2 * blend_action.runtime.total_seconds() * n_folds_final_blend / n_folds_hyperopt / 3600
                with open(f"{ramp_kit_dir}/timing.txt", "w") as file:
                    file.write(f"Elapsed time: {elapsed_time:.2f} hours")
                    file.write(f"\nEstimated runtime (train + valid + test) for final blend: {estimated_runtime_for_final_blend:.2f} hours")
                    file.write(f"\nEstimated final blending time: {estimated_final_blending_time:.2f} hours")
                if elapsed_time + estimated_runtime_for_final_blend + estimated_final_blending_time > max_time:
                    print("Stopping for time limit")
                    break
        #    input("Press Enter to continue...")
    return blended_submissions, round_datetime


def resume_race(
    action_stats: dict,
    ramp_kit_dir: str,
    base_predictors: list[str],
    contributivity_floor: int,
    n_folds_hyperopt: int,
    first_fold_idx: int,
) -> tuple[int, set[str], dict, list[float]]:
    print("Loading actions...")
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rh.actions.load_ramp_action(Path(action_f_name)))
    blend_actions = [
        ra for ra in ramp_program if ra.name == "blend" and ra.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt)
    ]
    
    if len(blend_actions) == 0:
        # Crash occured early, before the first round
        start_round = 0
        blended_submissions = set()
        scores = []
        return start_round, blended_submissions, action_stats, scores

    stop_time = blend_actions[-1].stop_time
    print(f"Last blending action at {stop_time}, deleting all actions after...")
    actions_f_names_to_delete = [a for a in action_f_names if pd.to_datetime(Path(a).stem) > stop_time]
    for f_name in actions_f_names_to_delete:
        Path(f_name).unlink()
    print("Re-loading actions...")
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rh.actions.load_ramp_action(Path(action_f_name)))
    # we only need race blend actions
    blend_actions = [ra for ra in ramp_program if ra.name == "blend" and ra.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt)]
    hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
    start_round = len(hyperopt_actions)
    scores = []
    print(f"Recovering {len(hyperopt_actions)} hyperopt and blend actions...")
    blend_action_idx = 0
    for hyperopt_action in hyperopt_actions:
        submission = hyperopt_action.kwargs["submission"]
        predictor = submission
        if "hyperopt" in submission:
            predictor = submission[:-20]
        if len(hyperopt_action.mean_scores) > 0:
            try:
                blend_action = blend_actions[blend_action_idx]
            except IndexError:
                print(f"blend_actions[{blend_action_idx}] does not exist, possibly corrupted actions")
                blend_action_idx += 1
                continue
            blend_action_idx += 1  # if hyperopt did not return with any submissions, there was no blend
            blended_score = blend_action.blended_score
            scores.append(blended_score)
            contributivities = {
                s: contributivity_floor
                + np.array([c for sh, c in blend_action.contributivities.items() if sh[: len(s)] == s]).sum()
                for s in base_predictors
            }
            action_stats[predictor].append(
                {
                    "runtime": hyperopt_action.runtime,
                    "contributivities": contributivities,
                }
            )
    blend_action = blend_actions[-1]
    blended_submissions = set([sh for sh, c in blend_action.contributivities.items() if c > 0])
    print(f"Blended submissions: {blended_submissions}")
    return start_round, blended_submissions, action_stats, scores


def final_blend_growing_folds(
    base_predictors: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    n_folds_hyperopt: int,
    first_fold_idx: int,
    top_n_for_mean: int,
    n_sigma: float,
):
    """Select best of each base submission (hyperopt) within n sigma and blend.

    We'll probably drop this since blend_and_bag and bag_and_blend work better,
    but keep it for early versions to experiment.
    """
    for stop_fold_idx in range(first_fold_idx + n_folds_hyperopt + 1, first_fold_idx + n_folds_final_blend + 1):
        selected_submissions = []
        for submission in base_predictors:
            rh.actions.select_top_hyperopt_and_train(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                fold_idxs=range(first_fold_idx, stop_fold_idx),
                trained_fold_idxs=range(first_fold_idx, stop_fold_idx - 1),
                top_n=top_n_for_mean,
                n_sigma=n_sigma,
            )
            top_hyperopt_dict = rh.actions.select_top_hyperopt(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                fold_idxs=range(first_fold_idx, stop_fold_idx),
                n_sigma=n_sigma,
            )
            if "selected_submissions" in top_hyperopt_dict:
                selected_submissions += top_hyperopt_dict["selected_submissions"]
        rh.actions.blend(
            ramp_kit_dir=ramp_kit_dir,
            submissions=selected_submissions,
            fold_idxs=range(first_fold_idx, stop_fold_idx),
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        )
        submission_target_f_name = (
            Path(ramp_kit_dir)
            / "final_test_predictions"
            / f"auto_{kit_suffix}_growing_folds_{str(stop_fold_idx).zfill(3)}.csv"
        )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


def update_hyperopt_summary(
    base_predictors: list[str],
    ramp_kit_dir: str,
):
    """Update the hyperopt summary.

    This is needed when submissions are trained outside hyperopt.
    """
    for submission in base_predictors:
        rh.actions.update_hyperopt_score_summary(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
        )


def train_on_all_folds(
    submissions: list[str],
    ramp_kit_dir: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
):
    """We retrain the final blend on all folds."""
    for submission in submissions:
        rh.actions.train(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
        )


@rh.actions.ramp_action
def final_blend_then_bag(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
    n_rounds: int = -1,
    round_datetime = None,  # for recording in the action, not used in the function
):
    """Blend then bag and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all in <ramp_kit_dir>/final_test_predictions.
    """
#    for stop_fold_idx in range(first_fold_idx + 1, first_fold_idx + n_folds_final_blend + 1):
    for stop_fold_idx in range(first_fold_idx + n_folds_final_blend, first_fold_idx + n_folds_final_blend + 1):
        rh.actions.blend(
            ramp_kit_dir=ramp_kit_dir,
            submissions=submissions,
            fold_idxs=range(first_fold_idx, stop_fold_idx),
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_valid.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{n_rounds}_valid.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_valid.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


@rh.actions.ramp_action
def final_bag_then_blend(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
    n_rounds: int = -1,
    round_datetime = None,  # for recording in the action, not used in the function
):
    """Bag then blend and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all.
    """
#    for stop_fold_idx in range(first_fold_idx + 1, first_fold_idx + n_folds_final_blend + 1):
    for stop_fold_idx in range(first_fold_idx + n_folds_final_blend, first_fold_idx + n_folds_final_blend + 1):
        rh.actions.bag_then_blend(
            ramp_kit_dir=ramp_kit_dir,
            submissions=submissions,
            fold_idxs=range(first_fold_idx, stop_fold_idx),
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_bagged_then_blended_test.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_bagged_then_blended_valid.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{n_rounds}_valid.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_valid.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


def submit_best_submissions(
    base_predictors: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
):
    """Submit the best of each submission (classical hyperopt) in
    <ramp_kit_dir>/final_test_predictions.
    """
    for submission in base_predictors:
        best_submissions = rh.actions.select_top_hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
            top_n=1,
        )["selected_submissions"]
        if len(best_submissions) == 0:
            print(f"No best {submission}")
            continue
        best_submission = best_submissions[0]
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_test.csv"
        )
        if not submission_source_f_name.exists():
            print(f"Bagging {submission}")
            # It wasn't in the final blends so we need to bag it
            rh.actions.train(
                ramp_kit_dir=ramp_kit_dir,
                submission=best_submission,
                fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
                bag=True,
            )
        submission_target_f_name = (
            Path(ramp_kit_dir) / "final_test_predictions" / f"auto_{kit_suffix}_best_{submission}.csv"
        )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_valid.csv"
        )
        submission_target_f_name = (
            Path(ramp_kit_dir) / "final_test_predictions" / f"auto_{kit_suffix}_best_{submission}_valid.csv"
        )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


def submit_base_submissions(
    ramp_kit_dir: str,
    metadata: dict,
    base_predictors: list[str],
    data_preprocessors: list[str],
) -> list[str]:
    for submission in base_predictors:
        if "regression" in metadata["prediction_type"]:
            submitted_elements = rs.scripts.tabular.tabular_regression_ordered_submit(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                regressor=submission,
                data_preprocessors=data_preprocessors,
            )

        elif "classification" in metadata["prediction_type"]:
            submitted_elements = rs.scripts.tabular.tabular_classification_ordered_submit(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                classifier=submission,
                data_preprocessors=data_preprocessors,
            )
    final_test_predictions_path = ramp_kit_dir / "final_test_predictions"
    final_test_predictions_path.mkdir(parents=False, exist_ok=True)
    print(submitted_elements)
    return submitted_elements


def hyperopt_race(
    ramp_kit: str,
    kit_root: str,
    version: str,
    number: str | int,
    resume: bool,
    n_rounds: int = 100,
    n_trials_per_round: int = 5,
    patience: int = -1,
    n_folds_hyperopt: int = 3,
    n_folds_final_blend: int = 30,
    first_fold_idx: int = 0,
    base_predictors: list[str] = ["lgbm", "xgboost", "catboost"],
    data_preprocessors: list[str] = [
        "drop_id",
        "drop_columns",
        "col_in_train_only",
        "base_columnwise",
        "rm_constant_col",
    ],
    preprocessors_to_hyperopt: Optional[list[str]] = None,
    max_time: float = 1000000,
    top_n_for_mean: int = 10,
    n_sigma: float = 1.0,
    contributivity_floor: int = 100,  # on 1000, added to contributivity to give a chance to every submission
    no_growing_folds: bool = True,
    n_cpu_per_run: int = None,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = Path(kit_root) / f"{ramp_kit}_{kit_suffix}"
    problem = rw.utils.assert_read_problem(str(ramp_kit_dir))
    score_names = [st.name for st in problem.score_types]
    valid_score_name = f"valid_{score_names[0]}"
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    with open(ramp_kit_dir / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)

    if n_cpu_per_run is not None:
        n_cpu_per_run = int(n_cpu_per_run)

    # Dictionary of submissions: list of dictionary of run times and scores
    action_stats = {submission: [] for submission in base_predictors}
    if resume:
        start_round, blended_submissions, action_stats, scores = resume_race(
            action_stats=action_stats,
            ramp_kit_dir=str(ramp_kit_dir),
            base_predictors=base_predictors,
            contributivity_floor=contributivity_floor,
            n_folds_hyperopt=n_folds_hyperopt,
            first_fold_idx=first_fold_idx,
        )
        try:
            config = rs.utils.load_config(load_path=ramp_kit_dir)
            dp_hyperopt_full_name = config["preprocessors_to_hyperopt"]
            print(dp_hyperopt_full_name)
        except FileNotFoundError:
            print("No config file, resuming with command-line parameters.")
            print("Happens only on early versions that crashed without saving the config.")
            dp_hyperopt_full_name = preprocessors_to_hyperopt
    else:
        start_round = 0
        blended_submissions = set()
        scores = []
        # submit base submissions
        submitted_elements = submit_base_submissions(
            ramp_kit_dir=ramp_kit_dir,
            metadata=metadata,
            base_predictors=base_predictors,
            data_preprocessors=data_preprocessors,
        )
        dp_hyperopt_full_name = []
        if preprocessors_to_hyperopt is not None:
            for dp in preprocessors_to_hyperopt:
                dp_hyperopt_full_name += rs.utils.get_full_preprocessor_name(
                    data_preprocessor=dp, submitted_preprocessors=submitted_elements["submitted_data_preprocessors"]
                )

        rs.utils.save_config(
            ramp_kit=ramp_kit,
            kit_root=kit_root,
            version=version,
            number=number,
            resume=resume,
            n_rounds=n_rounds,
            n_trials_per_round=n_trials_per_round,
            patience=patience,
            n_folds_hyperopt=n_folds_hyperopt,
            n_folds_final_blend=n_folds_final_blend,
            first_fold_idx=first_fold_idx,
            max_time=max_time,
            base_predictors=base_predictors,
            data_preprocessors=data_preprocessors,
            preprocessors_to_hyperopt=dp_hyperopt_full_name,
            top_n_for_mean=top_n_for_mean,
            n_sigma=n_sigma,
            contributivity_floor=contributivity_floor,
            no_growing_folds=no_growing_folds,
            save_path=ramp_kit_dir,
        )

    blended_submissions, round_datetime = run_race(
        base_predictors=base_predictors,
        action_stats=action_stats,
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        metadata=metadata,
        n_rounds=n_rounds,
        n_trials_per_round=n_trials_per_round,
        patience=patience,
        n_folds_hyperopt=n_folds_hyperopt,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        start_round=start_round,
        scores=scores,
        is_lower_the_better=is_lower_the_better,
        contributivity_floor=contributivity_floor,
        blended_submissions=blended_submissions,
        max_time=max_time,
        elapsed_time=0.0,
        preprocessors_to_hyper=dp_hyperopt_full_name,
        n_cpu_per_run=n_cpu_per_run,
    )

    # Run the growing folds algorithm: select best of each base submission within
    # n sigma and train on one more fold, then repeat. Then blend the best ones.
    # Conservative blend (discard diverse but bad submissions), we'll probably drop
    # this since blend_and_bag and bag_and_blend work better, but keep it for early
    # versions to experiment.
    if not no_growing_folds:
        final_blend_growing_folds(
            base_predictors=base_predictors,
            ramp_kit_dir=str(ramp_kit_dir),
            kit_suffix=kit_suffix,
            n_folds_final_blend=n_folds_final_blend,
            n_folds_hyperopt=n_folds_hyperopt,
            first_fold_idx=first_fold_idx,
            top_n_for_mean=top_n_for_mean,
            n_sigma=n_sigma,
        )
    # Train the final blend of the hyperopt race on all the folds
    train_on_all_folds(
        submissions=list(blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
    )
    # Whenever hyperopt submissions are trained outside of hyperopt, we need to update the summary.
    update_hyperopt_summary(
        base_predictors=base_predictors,
        ramp_kit_dir=str(ramp_kit_dir),
    )
    # Blend then bag the final blend of the hyperopt race on all the folds
    final_blend_then_bag(
        submissions=list(blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        round_datetime=round_datetime,
    )
    # Bag then blend the final blend of the hyperopt race on all the folds
    final_bag_then_blend(
        submissions=list(blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        round_datetime=round_datetime,
    )
    # Submit the best of each base submission (classical hyperopt)
    submit_best_submissions(
        base_predictors=base_predictors,
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
    )
