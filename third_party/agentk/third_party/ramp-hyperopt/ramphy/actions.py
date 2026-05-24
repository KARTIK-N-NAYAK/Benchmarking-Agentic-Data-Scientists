"""Actions for RAMP agent."""

import os
import time
import glob
import json
import shutil
import pickle
import hashlib
import pathlib
import datetime
import itertools
import functools
import importlib

from typing import Sequence, Optional, List, Tuple, Dict
import torch
import numpy as np
import pandas as pd
import rampwf as rw
from .hyperopt import (
    run_hyperopt,
    parse_hyperparameters,
    parse_all_hyperparameters,
    write_hyperparameters,
)
from pathlib import Path
from ray.tune.error import TuneError

RAMP_ACTIONS = dict()
# Set it to False in script to only dump action functions to turn into a plan
EXECUTE_PLAN = True


class RampAction:
    def __init__(self, module, name, args=(), kwargs={}):
        self.module = module
        self.name = name
        self.args = args
        self.kwargs = kwargs

    @property
    def runtime(self):
        return self.stop_time - self.start_time

    def execute(self):
        module = importlib.import_module(self.module)
        action_function = getattr(module, self.name)
        return action_function(*self.args, **self.kwargs)

    def save(self, f_name):
        f_name = f"{f_name}.pkl"
        with open(f_name, "wb") as f:
            pickle.dump(self, f)


def load_ramp_action(f_name: Path) -> RampAction:
    with open(f_name, "rb") as f:
        return pickle.load(f)


def ramp_action(action_function):
    """ """
    global RAMP_ACTIONS
    RAMP_ACTIONS[f"{action_function.__module__}.{action_function.__name__}"] = action_function

    @functools.wraps(action_function)
    def ramp_decorator(*args, **kwargs):
        global EXECUTE_PLAN
        # If EXECUTE_PLAN is False, we don't execute the action, only dump it
        # into <ramp_kit_dir>/actions.
        ramp_action_object = RampAction(
            module=action_function.__module__,
            name=action_function.__name__,
            args=args,
            kwargs=kwargs,
        )
        ramp_action_object.start_time = datetime.datetime.utcnow()
        action_return = {}
        if EXECUTE_PLAN:
            action_return = action_function(*args, **kwargs)
            ramp_action_object.stop_time = datetime.datetime.utcnow()
            try:
                for key, value in action_return.items():
                    setattr(ramp_action_object, key, value)
            except:
                pass
        ramp_kit_dir = kwargs["ramp_kit_dir"]
        actions_dir = Path(ramp_kit_dir) / "actions"
        actions_dir.mkdir(parents=False, exist_ok=True)
        f_name = actions_dir / f"{ramp_action_object.start_time}"
        ramp_action_object.save(f_name)
        return action_return

    return ramp_decorator


def get_all_actions(ramp_kit_dir):
    action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
    action_f_names.sort()
    all_actions = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        all_actions.append(load_ramp_action(action_f_name))
    return all_actions

def _bagged_score(score_type, bagged_f_name):
    bagged_scores_df = pd.read_csv(bagged_f_name)
    valid_scores_df = bagged_scores_df[bagged_scores_df["step"] == "valid"]
    return valid_scores_df.iloc[-1][score_type.name]


def load_contributivities(ramp_kit_dir):
    contributivites_df = pd.read_csv(f"{ramp_kit_dir}/submissions/training_output/contributivities.csv")
    contributivites_df = contributivites_df.set_index("submission")
    contributivites_df["contributivity"] = (
        contributivites_df[[f for f in contributivites_df.columns if f[:5] == "fold_"]].sum(axis=1).round(3) * 1000
    ).astype(int)
    return contributivites_df


def load_contributivities_bagged_then_blended(ramp_kit_dir):
    contributivites_df = pd.read_csv(
        f"{ramp_kit_dir}/submissions/training_output/contributivities_bagged_then_blended.csv"
    )
    contributivites_df = contributivites_df.set_index("submission")
    return contributivites_df


def _mean_score(submission, fold_idxs, score_type, ramp_kit_dir):
    foldwise_scores = []
    for fold_idx in fold_idxs:
        score_df = pd.read_csv(
            Path(ramp_kit_dir) / "submissions" / submission / "training_output" / f"fold_{fold_idx}" / "scores.csv"
        )
        score_df = score_df.set_index("step")
        foldwise_scores.append(score_df.loc["valid", score_type.name])
    return np.mean(foldwise_scores)


def _make_fold_idxs(fold_idxs, ramp_kit_dir, ramp_data_dir):
    if fold_idxs is None:
        cv = rw.utils.assert_cv(ramp_kit_dir, ramp_data_dir, fold_idxs=fold_idxs)
        fold_idxs = range(len(cv))
    else:
        fold_idxs = list(fold_idxs)
    return fold_idxs


def convert_ramp_dirs(ramp_kit_dir: Path | str, ramp_data_dir: Optional[Path | str]) -> Tuple[Path, Path]:
    """Convert ramp dirs to Path.

    Remember that ramp_data_dir does not include the
    /data subfolder, it is usually the same as ramp_kit_dir,
    but can point to an alternative data source for the same
    kit.
    Args:
        ramp_kit_dir (str): ramp_kit_dir
        ramp_data_dir (str): ramp_data_dir

    Returns:
        (Path, Path): converted dirs
    """
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)
    return ramp_kit_dir, ramp_data_dir


@ramp_action
def hyperopt(
    ramp_kit_dir: str,
    submission: str,
    n_trials: int,
    fold_idxs: Optional[Sequence[int]] = None,
    workflow_element_names: Optional[Sequence[str]] = None,
    resume: Optional[bool] = True,
    subtract_existing: Optional[bool] = False,
    ramp_data_dir: Optional[str] = None,
    n_cpu_per_run: Optional[int] = None,
) -> Dict:
    """Hyperopting action.

    Hyperopts submission using HEBO, creating
    <submission>_hyperopt_<hash> submissions, where hash hashes
    the list of hyperparameter values (even those that are not hyperopted,
    not selected in workflow_element_names). The number of trials
    n_trials is counted per fold, so should be a multiple of len(folds_idxs).
    Resume means we restart from existing models, fully trained on all folds.
    If subtract_existing is True, we subtract the number of hyperopted models
    times the number of folds from n_trials.

    We catch all exceptions in case at least one model is trained, and resume.
    If no model is trained, we re-raise the exception.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the submission to be hyperopted.
    n_trials : int
        Number of hyperopt trials, counting per fold.
        Should be a multiple of len(fold_idxs).
    fold_idxs : list of int, default=None
        Fold indices to hyperopt on.
        If None, we will train on all folds in problem.cv.
    workflow_element_names : list of strings, default=None
        The workflow elements that we hyperopt. None means we
        hyperopt all elements. For elements not hyperopted, we
        set the hyperopt space to the list of the single instance
        of default hyperparameter in the submission, so it seems
        like all elements are hyperopted, except that hypers of the
        unselected elements cannot "move".
    resume : bool, default=True
        If True, we resume from existing submissions hyperopted on
        all the folds in fold_idxs.
    subtract_existing : bool, default=False
        If True, we discount len(fold_idxs) x len(hyperopted submssions)
        from n_trials.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    scores : dict
        Dictionary containing "created_submissions": list(str) - all the
        submissions created in this round of hyperopt;
        "mean_scores": list(float) - all the scores of created submissions;
        "mean_score": float - the score of the best submission, given that
        at least one submission was successfully trained.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    fold_idxs = _make_fold_idxs(fold_idxs, ramp_kit_dir, ramp_data_dir)

    top_hyperopt_dict = select_top_hyperopt(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )
    if "selected_submissions" in top_hyperopt_dict:
        existing_submissions = top_hyperopt_dict["selected_submissions"]
    else:
        existing_submissions = []
    previous_submissions = existing_submissions.copy()
    n_existing_submissions = len(existing_submissions)
    n_existing_trials = n_existing_submissions * len(fold_idxs)
    if subtract_existing:
        n_total_trials = n_trials
    else:
        n_total_trials = n_trials + n_existing_trials
    n_exceptions = 0
    while True:
        n_trials_remaining = n_total_trials - n_existing_trials
        print(f"remaining trials = {n_trials_remaining}")
        if n_trials_remaining <= 0:
            print("n_trials_remaining <= 0, finished")
            break
        exception = ValueError("Unknown hyperopt exception, no submission trained.")
        try:
            created_submissions = run_hyperopt(
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
                ramp_submission_dir=ramp_kit_dir / "submissions",
                data_label=None,
                submission=submission,
                engine_name="ray_hebo",
                n_trials=n_trials_remaining,
                workflow_element_names=workflow_element_names,
                fold_idxs=fold_idxs,
                save_output=True,
                test=False,
                label=False,
                resume=resume,
                max_concurrent_runs=1,
                n_cpu_per_run=n_cpu_per_run,
                n_gpu_per_run=0,
                verbose=3,
            )
            n_trained_submissions = len(created_submissions)
            existing_submissions = existing_submissions + created_submissions
        except Exception as e:
            exception = e
            print(e)
            #            raise e
            top_hyperopt_dict = select_top_hyperopt(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                fold_idxs=fold_idxs,
                ramp_data_dir=ramp_data_dir,
            )
            if "selected_submissions" in top_hyperopt_dict:
                existing_submissions = top_hyperopt_dict["selected_submissions"]
            else:
                existing_submissions = []
            n_trained_submissions = len(existing_submissions) - n_existing_submissions
            # We raise only if there was no new submissions trained,
            # and it happened 10x in a row
        if n_trained_submissions <= 0:
            if n_exceptions > 10:
                r = dict()
                r["created_submissions"] = []
                r["mean_scores"] = {}
                return r
            else:
                n_exceptions += 1
        else:
            n_exceptions = 0
        n_existing_submissions = len(existing_submissions)
        n_existing_trials = n_existing_submissions * len(fold_idxs)
    r = dict()
    created_submissions = set(existing_submissions) - set(previous_submissions)
    r["created_submissions"] = list(created_submissions)
    r["mean_scores"] = {}
    for submission in created_submissions:
        r["mean_scores"][submission] = _mean_score(submission, fold_idxs, problem.score_types[0], ramp_kit_dir)
    if len(r["mean_scores"]) > 0:
        if problem.score_types[0].is_lower_the_better:
            r["mean_score"] = min(r["mean_scores"].values())
        else:
            r["mean_score"] = max(r["mean_scores"].values())
    #    ray_trash_folders = glob.glob("/tmp/ray/*")
    #    for folder in ray_trash_folders:
    #        shutil.rmtree(folder)
    #    ray_trash_folders = glob.glob(Path.home() / "ray_results")
    #    for folder in ray_trash_folders:
    #        shutil.rmtree(folder)
    return r


@ramp_action
def train(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Optional[Sequence[int]] = None,
    bag: Optional[bool] = True,
    force_retrain: Optional[bool] = False,
    ignore_errors: Optional[bool] = False,
    ramp_data_dir: Optional[str] = None,
) -> Dict:
    """Training action.

    Trains and bags a submission on a set of folds.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the submission to be tested.
    fold_idxs : list of int, default=None
        Fold indices to train on.
        If None, we will train on all folds in problem.cv.
    bag : bool, default=True
        If True, we bag the folds after training.
    force_retrain : bool, default=False
        If True, we retrain on folds with existing scores.
    ignore_errors : bool, default=False
        If True, we silently return
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    scores : dict
        Dictionary of mean score (if training is successful) and bagged score
        (if bag is True).
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    fold_ixs = _make_fold_idxs(fold_idxs, ramp_kit_dir, ramp_data_dir)
    scores = dict()
    try:
        rw.utils.testing.assert_submission(
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            ramp_submission_dir=ramp_kit_dir / "submissions",
            submission=submission,
            save_output=True,
            retrain=False,
            bag=bag,
            force_retrain=force_retrain,
            fold_idxs=fold_idxs,
        )
        foldwise_scores = []
        for fold_idx in fold_idxs:
            score_df = pd.read_csv(
                ramp_kit_dir / "submissions" / submission / "training_output" / f"fold_{fold_idx}" / "scores.csv"
            )
            score_df = score_df.set_index("step")
            foldwise_scores.append(score_df.loc["valid", problem.score_types[0].name])
        scores["mean_score"] = _mean_score(submission, fold_idxs, problem.score_types[0], ramp_kit_dir)
    except Exception as e:
        if not ignore_errors:
            raise e
        return scores
    if bag:
        submission_dir = Path(ramp_kit_dir) / "submissions" / submission
        bagged_f_name = submission_dir / "training_output" / "bagged_scores.csv"
        scores["bagged_score"] = _bagged_score(problem.score_types[0], bagged_f_name)
    return scores


@ramp_action
def retrain(
    ramp_kit_dir: str,
    submission: str,
    ramp_data_dir: Optional[str] = None,
) -> Dict:
    """Retraining action.

    Trains the submissin on full training data. No reward since no
    validation set.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the submission to be retrained.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    scores : dict
        Empty dict since no validaton score.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    rw.utils.testing.assert_submission(
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        submission=submission,
        save_output=True,
        retrain=True,
        fold_idxs=[],
    )
    return dict()


@ramp_action
def blend(
    ramp_kit_dir: str,
    submissions: List[str],
    fold_idxs: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
    ramp_data_dir: Optional[str] = None,
) -> Dict:
    """Blending action.

    Blends a list of submissions.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submissions : list of str
        The name of the submissions to be blended.
    fold_idxs : list of int, default=None
        Fold indices to blend.
        If None, we will blend all folds.
    output_path : str, default=None.
        The folder where bagged_scores_combined.csv and
        submission_combined_bagged_test.csv are saved. If None, defaults
        to <ramp_kit_dir>/submissions/training_output.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    scores : dict
        A dictionary with a single element "blended_score": float. If submissions = []
        it returns an empty dictionary.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    if len(submissions) > 0:
        problem = rw.utils.assert_read_problem(ramp_kit_dir)
        if output_path is None:
            output_path = ramp_kit_dir / "submissions" / "training_output"
        else:
            output_path = Path(output_path)
        rw.utils.testing.blend_submissions(
            submissions,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            save_output=True,
            output_path=str(output_path),
            fold_idxs=fold_idxs,
            bag_ranks=problem.score_types[0].is_rank_based,
        )
        r = {}
        bagged_f_name = output_path / "bagged_scores_combined.csv"
        r["blended_score"] = _bagged_score(problem.score_types[0], bagged_f_name)
        contributivities_df = load_contributivities(ramp_kit_dir)
        r["contributivities"] = contributivities_df["contributivity"].to_dict()
        return r
    else:
        return {}


@ramp_action
def bag_then_blend(
    ramp_kit_dir: str,
    submissions: List[str],
    fold_idxs: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
    ramp_data_dir: Optional[str] = None,
) -> Dict:
    """Bag-then-blending action.

    Bags the blends a list of submissions.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submissions : list of str
        The name of the submissions to be blended.
    fold_idxs : list of int, default=None
        Fold indices to blend.
        If None, we will blend all folds.
    output_path : str, default=None.
        The folder where bagged_then_blended_scores.csv and
        submission_bagged_then_blended_scores_test.csv are saved. If None, defaults
        to <ramp_kit_dir>/submissions/training_output.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    scores : dict
        A dictionary with a single element "blended_score": float. If submissions = []
        it returns an empty dictionary.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    if len(submissions) > 0:
        problem = rw.utils.assert_read_problem(ramp_kit_dir)
        if output_path is None:
            output_path = ramp_kit_dir / "submissions" / "training_output"
        else:
            output_path = Path(output_path)
        rw.utils.testing.bag_then_blend_submissions(
            submissions,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            save_output=True,
            output_path=str(output_path),
            fold_idxs=fold_idxs,
        )
        r = {}
        scores_df = pd.read_csv(output_path / "bagged_then_blended_scores.csv")
        r["blended_score"] = scores_df["valid"][0]
        contributivities_df = load_contributivities_bagged_then_blended(ramp_kit_dir)
        r["contributivities"] = contributivities_df["contributivity"].to_dict()
        return r
    else:
        return {}


# has to be redesigned because of variable number of data preprocessors.
@ramp_action
def submit_hybrid(
    ramp_kit_dir: str,
    new_submission: str,
    parent_submissions: Dict[str | Tuple[str], str],
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Combines workflow elements coming from different submissions.

    Inheriting workflow elements from parent submissions.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    new_submission : str
        The name of the new submission to be submitted.
    parent_submissions : dict
        A dictionary of elements {workflow_element or tuple of elements: parent_submission}.
        For the data preprocessors, if you do data_preprocessors: parent_submission, it will copy all
        the preprocessors of that submission
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    new_submission_dir = ramp_kit_dir / "submissions" / new_submission
    if new_submission_dir.exists():
        shutil.rmtree(new_submission_dir)
    new_submission_dir.mkdir(parents=False, exist_ok=True)

    # Unroll any list
    # We do it so we respect the order
    # Note that we take advantage of the fact that since Python3.7 dicts are ordered
    elements_dict = {}
    for wf_element in parent_submissions:
        submission = parent_submissions[wf_element]
        if isinstance(wf_element, tuple):
            for element in wf_element:
                elements_dict[element] = submission
        # in this case we copy all preprocessors
        elif wf_element == "data_preprocessors":
            source_dir = Path(ramp_kit_dir) / "submissions" / submission
            d_preprocessors = [str(file.name)[:-3] for file in source_dir.glob("data_preprocessor*") if file.is_file()]
            # Use this to sort them otherwise we'll have 1, 10, 2
            for i in range(len(d_preprocessors)):
                dp = [s for s in d_preprocessors if s.startswith(f"data_preprocessor_{i}_")][0]
                elements_dict[dp] = submission
        else:
            elements_dict[wf_element] = submission

    # Copy elements
    prepr_index = 0
    for wf_element in elements_dict:
        parent = elements_dict[wf_element]
        if "data_preprocessor" in wf_element:
            name = wf_element.split("_")[3:]
            name = "_".join(name)
            new_name = f"data_preprocessor_{prepr_index}_{name}"
            prepr_index += 1
            from_file = Path(ramp_kit_dir) / "submissions" / parent / f"{wf_element}.py"
            to_file = Path(ramp_kit_dir) / "submissions" / new_submission / f"{new_name}.py"
            shutil.copy(from_file, to_file)
            print(f"Copying {from_file} to {to_file}")
        else:
            from_file = Path(ramp_kit_dir) / "submissions" / parent / f"{wf_element}.py"
            to_file = Path(ramp_kit_dir) / "submissions" / new_submission / f"{wf_element}.py"
            shutil.copy(from_file, to_file)
            print(f"Copying {from_file} to {to_file}")


def update_hyperopt_score_summary(
    ramp_kit_dir: str,
    submission: str,
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Updates hyperopt_output/summary.csv by reading existing scores.

    Should be called any time when hyperopt submissions are manipualated
    outside hyperopt, e.g., retrained on folds or deleted.
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    hyperopt_output_dir = ramp_kit_dir / "submissions" / submission / "hyperopt_output"
    hyperopt_output_dir.mkdir(parents=False, exist_ok=True)
    summary_fname = hyperopt_output_dir / "summary.csv"
    print(f"Updating {summary_fname} from score files...")
    summary_df = get_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        ramp_data_dir=ramp_data_dir,
        force_reload=True,
    )
    if len(summary_df) > 0:
        summary_df.to_csv(summary_fname)


def get_hyperopt_score_summary(
    ramp_kit_dir: str,
    submission: str,
    selected_submissions: Optional[Sequence[str]] = None,
    fold_idxs: Optional[Sequence[int]] = None,
    ramp_data_dir: Optional[str] = None,
    force_reload: Optional[bool] = False,
    test: Optional[bool] = False,
    data_label: Optional[str] = None,
) -> pd.DataFrame:
    """Returns a summary DataFrame.

    Selects all {submission}_hyperopt* submissions which have been
    trained on fold_idxs.
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    selected_submissions : list of str, default=None
        The list of hyperopted submissions. If None, all
        hyperopted submissions will be loaded.
    fold_idxs : list or generator of int, default=None
        Fold indices that the {submission}_hyperopt*'s
        have been trained on. If None, return summary of all
        folds.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame of submissions, hypers, scores
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)

    summary_fname = ramp_kit_dir / "submissions" / submission / "hyperopt_output" / "summary.csv"
    if summary_fname.is_file() and not force_reload:
        return pd.read_csv(summary_fname, index_col=0)

    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    X_train, y_train, X_test, y_test = rw.utils.assert_data(
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, data_label=data_label)
    cv = rw.utils.assert_cv(ramp_kit_dir, ramp_data_dir, fold_idxs=fold_idxs)
    score_names = [st.name for st in problem.score_types]
    valid_score_name = f"valid_{score_names[0]}"
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    submission_path = ramp_kit_dir / "submissions" / submission
    hypers = parse_all_hyperparameters(submission_path, problem.workflow)
    hyper_names = [f"hyper_{h.name}" for h in hypers]
    hyper_index_names = [f"{hn}_i" for hn in hyper_names]
    hyper_grid_sizes = [len(h.values) for h in hypers]

    if selected_submissions is None:
        score_f_names = glob.glob(
            f"{str(ramp_kit_dir)}/submissions/{submission}_hyperopt*/training_output/fold*/scores.csv"
        )
    else:
        score_f_names = []
        for ss in selected_submissions:
            score_f_names.append(glob.glob(f"{str(ramp_kit_dir)}/submissions/{ss}/training_output/fold*/scores.csv"))
    row_dicts = []
    print("Updating hyperparameter summaries...")
    for score_f_name in score_f_names:
        row_dict = {}
        fold_idx = int(Path(score_f_name).parent.name.split("_")[1])
        if fold_idxs is None or fold_idx in fold_idxs:
            row_dict["hyperopt_submission"] = Path(score_f_name).parent.parent.parent.name
            row_dict["fold_idx"] = fold_idx
            hyper_submission_path = ramp_kit_dir / "submissions" / row_dict["hyperopt_submission"]
            hyper_hypers = parse_all_hyperparameters(hyper_submission_path, problem.workflow)
            for h in hyper_hypers:
                row_dict[f"hyper_{h.name}"] = h.default
            for h in hyper_hypers:
                row_dict[f"hyper_{h.name}_i"] = h.default_index
            score_df = pd.read_csv(hyper_submission_path / "training_output" / f"fold_{fold_idx}" / "scores.csv")
            score_df = score_df.set_index("step")
            steps = ["train", "valid"]
            if test:
                steps += ["test"]
            for step in steps:
                for sn in score_names:
                    row_dict[f"{step}_{sn}"] = score_df.loc[step, sn]
            for step in steps:
                row_dict[f"{step}_time"] = score_df.loc[step, "time"]
            if fold_idxs is None:
                fold_i = fold_idx
            else:
                fold_i = fold_idxs.index(fold_idx)
            row_dict["n_train"] = len(cv[fold_i][0])
            row_dict["n_valid"] = len(cv[fold_i][1])
            if test:
                row["n_test"] = len(X_test)
            row_dicts.append(row_dict)
    summary_df = pd.DataFrame.from_records(row_dicts)
    return summary_df


def filter_full_folds(
    summary_df: pd.DataFrame,
    fold_idxs: Sequence[int],
) -> pd.DataFrame:
    """Returns a summary DataFrame.

    Filters submissions that were trained on all the given folds.
    Parameters
    ----------
    summary_df : pd.DataFrame
        All submissions, folds, and scores.
    fold_idxs : list or generator of int
        Fold indices that we want to filter.
    Returns
    -------
    new_summary_df : pd.DataFrame
        The filtered summary.
    """
    summary_filtered_df = summary_df[summary_df["fold_idx"].isin(fold_idxs)]
    groupby_columns = ["hyperopt_submission"]
    counts_df = summary_filtered_df.set_index(groupby_columns).groupby(groupby_columns).count()
    full_hyperopt_submissions = counts_df[counts_df["fold_idx"] == len(fold_idxs)].index.to_numpy()
    return summary_filtered_df.loc[summary_filtered_df["hyperopt_submission"].isin(full_hyperopt_submissions)]


def get_hyperopt_score_means(
    summary_df: pd.DataFrame,
    fold_idxs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Returns a mean DataFrame.

    Computes the mean of each submission over folds. If
    fold_idxs is not None, it first filters submissions
    that were trained on all the given folds.
    Parameters
    ----------
    summary_df : pd.DataFrame
        All submissions, folds, and scores.
    fold_idxs : list or generator of int, default=None
        Fold indices that we want to take the mean over.
        If None, mean over all existing folds will be returned.
    Returns
    -------
    means_df : pd.DataFrame
        The mean of submissions over folds.
    """
    groupby_columns = ["hyperopt_submission"]
    non_hyper_columns = [col for col in summary_df.set_index(groupby_columns).columns if col[:6] != "hyper_"]

    if fold_idxs is not None:
        summary_df = filter_full_folds(summary_df, fold_idxs)

    agg = {
        col: "mean" if col in non_hyper_columns else "first" for col in summary_df.set_index(groupby_columns).columns
    }
    means_df = summary_df.set_index(groupby_columns).groupby(groupby_columns).agg(agg)
    return means_df


def get_hyperopt_score_stds(
    summary_df: pd.DataFrame,
    fold_idxs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Returns an std DataFrame.

    Computes the std of each submission over folds. If
    fold_idxs is not None, it first filters submissions
    that were trained on all the given folds.
    Parameters
    ----------
    summary_df : pd.DataFrame
        All submissions, folds, and scores.
    fold_idxs : list or generator of int, default=None
        Fold indices that we want to take the std over.
        If None, std over all existing folds will be returned.
    Returns
    -------
    stds_df : pd.DataFrame
        The std of submissions over folds.
    """
    groupby_columns = ["hyperopt_submission"]
    non_hyper_columns = [col for col in summary_df.set_index(groupby_columns).columns if col[:6] != "hyper_"]

    if fold_idxs is not None:
        summary_df = filter_full_folds(summary_df, fold_idxs)

    agg = {col: "std" if col in non_hyper_columns else "first" for col in summary_df.set_index(groupby_columns).columns}
    stds_df = summary_df.set_index(groupby_columns).groupby(groupby_columns).agg(agg)
    return stds_df


def get_hyperopt_score_counts(
    summary_df: pd.DataFrame,
    fold_idxs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Returns a count DataFrame.

    Computes the std of each submission over folds. If
    fold_idxs is not None, it first filters submissions
    that were trained on all the given folds.
    Parameters
    ----------
    summary_df : pd.DataFrame
        All submissions, folds, and scores.
    fold_idxs : list or generator of int, default=None
        Fold indices that we want to take the std over.
        If None, std over all existing folds will be returned.
    Returns
    -------
    counts_df : pd.DataFrame
        The count of submissions over folds.
    """
    groupby_columns = ["hyperopt_submission"]
    non_hyper_columns = [col for col in summary_df.set_index(groupby_columns).columns if col[:6] != "hyper_"]

    if fold_idxs is not None:
        summary_df = filter_full_folds(summary_df, fold_idxs)

    agg = {
        col: "count" if col in non_hyper_columns else "first" for col in summary_df.set_index(groupby_columns).columns
    }
    counts_df = summary_df.set_index(groupby_columns).groupby(groupby_columns).agg(agg)
    return counts_df


def save_hyperopt_score_summary(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Optional[Sequence[int]] = None,
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Saves a summary csv.

    Selects all {submission}_hyperopt* submissions which have been
    trained on fold_idxs and saves their scores and hypers into
    submissions/{submission}/training_output/hyperopt_summary.csv.

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int, default=None
        Fold indices that the {submission}_hyperopt*'s
        have been trained on. If None, return summary of all
        folds.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    """
    print("Loading hyperopt scores.")
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    summary_df = get_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )
    training_output_path = ramp_kit_dir / "submissions" / submission / "training_output"
    training_output_path.mkdir(parents=False, exist_ok=True)
    f_name = training_output_path / "hyperopt_summary.csv"
    print(f"Saving hyperopt scores into {f_name}.")
    summary_df.to_csv(f_name)


@ramp_action
def select_top_hyperopt(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    score_cutoff: Optional[float] = None,
    top_n: Optional[int] = None,
    n_sigma: Optional[float] = None,
    ramp_data_dir: Optional[str] = None,
) -> Dict:
    """Returns submissions {submission}_hyperopt* with top score.

    Selects all {submission}_hyperopt* submissions which have been
    trained on fold_idxs. Then selects either the top_n
    according to mean score, or those with mean score better than
    score_cutoff, or those that are within n_sigma x SE of the top.
    If all three are None, return all hyperopts that were trained
    on fold_idxs.
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s
        have been trained on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s
        to be returned.
        Either score_cutoff or top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s
        to be returned.
    n_sigma : float, default=None
        Top 20 submissions: score_cutoff = mean - sigma * n
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    new_submissions : list of str
        The list of selected submissions.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    valid_score_name = f"valid_{score_names[0]}"
    is_lower_the_better = problem.score_types[0].is_lower_the_better

    print("Loading existing hyperopt submissions...")
    summary_df = get_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )
    if len(summary_df) == 0:
        return {"selected_submissions": [], "score_cutoff": None}

    means_df = get_hyperopt_score_means(summary_df, fold_idxs)
    if len(means_df) == 0:
        return {"selected_submissions": [], "score_cutoff": None}

    if n_sigma is not None:
        # foldwise means for unbiasing
        summary_df = filter_full_folds(summary_df, fold_idxs)
        groupby_columns = ["fold_idx"]
        non_hyper_columns = [col for col in summary_df.set_index(groupby_columns).columns if col[:6] != "hyper_"]
        agg = {
            col: "mean" if col in non_hyper_columns else "first"
            for col in summary_df.set_index(groupby_columns + ["hyperopt_submission"]).columns
        }
        foldwise_means_df = summary_df.groupby(groupby_columns).agg(agg)
        foldwise_means_df[f"{valid_score_name}_bias"] = (
            foldwise_means_df[f"{valid_score_name}"] - foldwise_means_df[f"{valid_score_name}"].mean()
        )
        # unbiasing summary
        summary_df = (
            summary_df.set_index("fold_idx").join(foldwise_means_df[[f"{valid_score_name}_bias"]]).reset_index()
        )
        summary_df[f"{valid_score_name}_unbiased"] = (
            summary_df[f"{valid_score_name}"] - summary_df[f"{valid_score_name}_bias"]
        )
        # computing standard errors
        stds_df = get_hyperopt_score_stds(summary_df, fold_idxs)
        counts_df = get_hyperopt_score_counts(summary_df, fold_idxs)
        counts_df["fold_count"] = counts_df["fold_idx"]
        means_df = means_df.join(stds_df[[f"{valid_score_name}_unbiased"]])
        means_df = means_df.join(counts_df[["fold_count"]])
        means_df["mean_std"] = means_df[f"{valid_score_name}_unbiased"] / np.sqrt(means_df["fold_count"])

        means_df = means_df.sort_values(valid_score_name, ascending=is_lower_the_better)
        top_mean = means_df.iloc[:top_n][valid_score_name].mean()
        top_std = means_df.iloc[:top_n]["mean_std"].mean()
        if is_lower_the_better:
            score_cutoff = top_mean + n_sigma * top_std
        else:
            score_cutoff = top_mean - n_sigma * top_std
    if score_cutoff is not None:
        if is_lower_the_better:
            new_submissions = means_df[means_df[valid_score_name] <= score_cutoff].index
        else:
            new_submissions = means_df[means_df[valid_score_name] >= score_cutoff].index
    else:
        sorted_df = means_df.sort_values(valid_score_name, ascending=is_lower_the_better)
        if top_n is None:
            top_n = len(sorted_df)
        new_submissions = sorted_df.iloc[:top_n].index
        score_cutoff = sorted_df.iloc[top_n - 1][valid_score_name]
    print(f"Selected {len(new_submissions)} submissions")
    print(f"score_cutoff = {score_cutoff}")

    r = dict()
    r["selected_submissions"] = new_submissions.to_list()
    r["score_cutoff"] = score_cutoff
    return r


@ramp_action
def rename_best_hyperopt_submissions(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    top_n: Optional[int] = None,
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Renames best submissions for archiving.

    Selects all {submission}_hyperopt* submissions which have been
    trained on fold_idxs blends them, then orders them according
    to the mean validation score, selects the top 10 and

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s
        have been trained on.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s
        to be renamed.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    valid_score_name = f"valid_{score_names[0]}"
    is_lower_the_better = problem.score_types[0].is_lower_the_better

    submission_path = Path(ramp_kit_dir) / "submissions" / submission
    hypers = parse_all_hyperparameters(submission_path, problem.workflow)
    hyper_names = [f"hyper_{h.name}" for h in hypers]
    hyper_index_names = [f"{hn}_i" for hn in hyper_names]

    print("Loading hyperopt scores.")
    summary_df = get_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )

    means_df = get_hyperopt_score_means(summary_df, fold_idxs)

    blended_return = select_top_hyperopt_and_blend(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )
    if "blended_score" in blended_return.keys():  # otherwise no submissions were blended
        contributivites_df = pd.read_csv(f"{ramp_kit_dir}/submissions/training_output/contributivities.csv")
        contributivites_df = load_contributivities(ramp_kit_dir)
        contributivites_df = pd.merge(
            contributivites_df,
            means_df.reset_index().set_index("hyperopt_submission")[[valid_score_name]],
            left_index=True,
            right_index=True,
        )

        for hyperopt_submission_i, hyperopt_submission in enumerate(
            contributivites_df.sort_values(valid_score_name, ascending=is_lower_the_better).index
        ):
            contributivity = int(contributivites_df.loc[hyperopt_submission]["contributivity"])
            if top_n is None or hyperopt_submission_i < top_n or contributivity > 0:
                from_f_name = f"{ramp_kit_dir}/submissions/{hyperopt_submission}"
                to_f_name = f"{ramp_kit_dir}/submissions/{submission}_best_{hyperopt_submission_i}_{contributivity}"
                print(f"{from_f_name} -> {to_f_name}")
                shutil.move(from_f_name, to_f_name)


@ramp_action
def delete_duplicates_hyperopt(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Deletes duplicates of {submission}_hyperopt*.

    Two submission are considered duplicates if their valid score
    is exactly the same.
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s
        have been trained on.
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    valid_score_name = f"valid_{score_names[0]}"

    summary_df = get_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )

    means_df = get_hyperopt_score_means(summary_df, fold_idxs)

    unique_submissions = means_df.reset_index().groupby(valid_score_name).first()["hyperopt_submission"].to_numpy()
    for s in means_df.reset_index()["hyperopt_submission"]:
        if s not in unique_submissions:
            print(f"Removing {ramp_kit_dir}/submissions/{s}")
            shutil.rmtree(Path(ramp_kit_dir) / "submissions" / s)

    update_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
    )


def select_top_hyperopt_and_train(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    trained_fold_idxs: Optional[Sequence[int]] = None,
    score_cutoff: Optional[float] = None,
    top_n: Optional[int] = None,
    n_sigma: Optional[float] = None,
    ignore_errors: Optional[bool] = False,
    ramp_data_dir: Optional[str] = None,
):
    """Selects and trains submissions {submission}_hyperopt*.

    Selects all {submission}_hyperopt* submissions, then those
    which have been trained on trained_fold_idxs, if trained_fold_idxs
    is not None. Then selects either the top_n according to mean
    score, or those with mean score better than score_cutoff. Then
    trains these submissions on fold_idxs.

    If trained_fold_idxs is None, all submissions
    {submission}_hyperopt* will be trained on fold_idxs.

    For example, we have a set of {submission}_hyperopt* hyperopted
    on a smaller fold with a small number of folds, say
    trained_fold_idxs = [400, 401, 402].
    We select the top_n = 50 of them and train them on a larger number
    of larger folds fold_idxs = [700, 701, 702, 703, 704].
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices to train selected submissions on.
    train_fold_idxs : list or generator of int, default=None
        Fold indices that the {submission}_hyperopt*'s
        have been trained on. If None, all submission_hyperopt*'s
        will be trained on fold_idxs.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        If train_fold_idxs is not None, either score_cutoff or
        top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
        If train_fold_idxs is not None, either score_cutoff or
        top_n must be non None.
    n_sigma : float, default=None
        Top 20 submissions: score_cutoff = mean - sigma * n
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    Returns
    -------
    new_submissions : list of str
        The list of selected submissions.
    """
    ramp_kit_dir, ramp_data_dir = convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    if trained_fold_idxs is None:
        submissions_paths = glob.glob(f"{ramp_kit_dir}/submissions/{submission}_hyperopt*")
        new_submissions = [pathlib.PurePath(path) for path in submissions_paths]
    else:
        top_hyperopt_dict = select_top_hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=trained_fold_idxs,
            score_cutoff=score_cutoff,
            top_n=top_n,
            n_sigma=n_sigma,
            ramp_data_dir=ramp_data_dir,
        )
        if "selected_submissions" in top_hyperopt_dict:
            new_submissions = top_hyperopt_dict["selected_submissions"]
        else:
            new_submissions = []
    for i, new_submission in enumerate(new_submissions):
        print(f"Training submission {i}/{len(new_submissions)}")
        train(
            ramp_kit_dir=ramp_kit_dir,
            submission=new_submission,
            fold_idxs=fold_idxs,
            bag=True,
            ignore_errors=ignore_errors,
            ramp_data_dir=ramp_data_dir,
        )
    update_hyperopt_score_summary(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
    )


def select_top_hyperopt_and_blend(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    score_cutoff: Optional[float] = None,
    top_n: Optional[int] = None,
    n_sigma: Optional[float] = None,
    ramp_data_dir: Optional[str] = None,
) -> None:
    """Selects and blends submissions {submission}_hyperopt*.

    Selects {submission}_hyperopt* submissions, then those
    which have been trained on fold_idxs, and blends them.
    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s
        have been trained on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
    n_sigma : float, default=None
        Top 20 submissions: score_cutoff = mean - sigma * n
    ramp_data_dir : str, default=None.
        Alternative ramp_kit_dir for using another data set. If None,
        set to ramp_kit_dir.
    """
    top_hyperopt_dict = select_top_hyperopt(
        ramp_kit_dir=ramp_kit_dir,
        submission=submission,
        fold_idxs=fold_idxs,
        score_cutoff=score_cutoff,
        top_n=top_n,
        n_sigma=n_sigma,
        ramp_data_dir=ramp_data_dir,
    )
    if "selected_submissions" in top_hyperopt_dict:
        submissions = top_hyperopt_dict["selected_submissions"]
    else:
        submissions = []
    return blend(
        ramp_kit_dir=ramp_kit_dir,
        submissions=submissions,
        fold_idxs=fold_idxs,
        ramp_data_dir=ramp_data_dir,
    )


def select_top_hyperopt_and_submit_hybrid(
    new_submission: str,
    parent_submissions: dict,
    select_element: str,
    fold_idxs: Sequence[int],
    score_cutoff: Optional[float] = None,
    top_n: Optional[int] = None,
    keep_hypers: bool = False,
    ramp_kit_dir: str = ".",
    ramp_data_dir: str = ".",
) -> None:
    """Selects top {submission}_hyperopt*'s and combines them with a new we.

    Selects all {parent_submissions[select_element]}_hyperopt* submissions
    which have been trained on fold_idxs, then selects either the
    top_n according to mean score, or those with mean score better than
    score_cutoff. Then hybridizes these hyperopted submissions with the
    workflow elements of the other parent_submissions.

    For example, say LLM submits a new feature extractor that we pair with
    xgboost, calling it xgboost_fe. We also have 100 submissions
    xgboost_hyperopt_* with the old feature extractor, hyperopted on
    fold_idxs. We want to select the top 50 of them, and submit these 50 new
    submissions where the classifiers come from xgboost_hyperopt_*, and the
    feature extractor comes from xgboost_fe. We call this function then
    with new_submission = 'xgboost_fe' (will become the prefix of the new
    50 submissions; the hash in xgboost_fe_<hash> will be computed on the
    new submission files since the new feature extractor might have changed
    hyper values, on which the hash is computed); parent_submissions =
    {'feature_extractor':'xgboost_fe', 'classifier':'xgboost'}, indicating that the feature extractor (the
    first workflow element) will come from xgboost_fe, and the second
    (the classifier) will come from 'xgboost_hyperopt*, and select_element='classifier',
    telling that it is the hyperopted children of the parent providing the classifier that will
    be hybridized. In case the LLMs submit the classifier, we'd have
    parent_submissions = {'feature_extractor':'xgboost', 'classifier':'new_classifier'}, and select_element="feature_extractor.

    Parameters
    ----------
    new_submission : str
        The name of the new submission to be submitted.
    parent_submissions : list of str
        The names of the submissions from which the workflow elements
        will come.
    select_element: str,
        The element of the parent submission whose (typically)
        hyperopted children will be used in the hybrid.
    fold_idxs : list or generator of int
        Fold indices to train selected submissions on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        Either score_cutoff or top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
    keep_hypers : bool, default=False
        If True, hypers of the top
        {parent_submissions[select_idx]}_hyperopt* will be kept
        even for the workflow elements that do not come
        from the {parent_submissions[select_idx]}_hyperopt*.
        Only works if all the parent submissions share the same
        hypers.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    top_hyperopt_dict = select_top_hyperopt(
        ramp_kit_dir=ramp_kit_dir,
        submission=parent_submissions[select_element],
        fold_idxs=fold_idxs,
        score_cutoff=score_cutoff,
        top_n=top_n,
        ramp_data_dir=ramp_data_dir,
    )
    if "selected_submissions" in top_hyperopt_dict:
        new_submissions = top_hyperopt_dict["selected_submissions"]
    else:
        new_submissions = []
    for submission in new_submissions:
        parent_submissions[select_element] = submission
        submit_hybrid("__new_submission__", parent_submissions, ramp_kit_dir, ramp_data_dir)
        submission_path = Path(ramp_kit_dir) / "submissions" / "__new_submission__"
        if keep_hypers:
            orig_submission_path = Path(ramp_kit_dir) / "submissions" / submission
            hypers_per_workflow_element = {
                wen: parse_hyperparameters(orig_submission_path, wen)
                for wen in problem.workflow.element_names
            }
            write_hyperparameters(submission_path, submission_path, hypers_per_workflow_element)
        hypers = parse_all_hyperparameters(submission_path, problem.workflow)
        hyper_indices = [h.default_index for h in hypers]
        hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
        output_submission_dir = Path(ramp_kit_dir) / "submissions" / f"{new_submission}_hyperopt_{hyper_hash}"
        if not output_submission_dir.exists():  # force resubmit perhaps?
            shutil.move(submission_path, output_submission_dir)


def clean_up_predictions(
    ramp_kit_dir: str,
    submission: str,
    fold_idxs: Sequence[int],
    score_cutoff: Optional[float] = None,
) -> None:
    """

    Parameters
    ----------
    ramp_kit_dir : str
        The directory of the ramp-kit.
    submission : str
        The name of the new submission to be cleaned.
    fold_idxs : list or generator of int
        Fold indices to select submissions on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        Either score_cutoff or top_n must be non None.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    if score_cutoff is None:
        submissions_f_names = glob.glob(f"{ramp_kit_dir}/submissions/{submission}*")
    else:
        top_hyperopt_dict = select_top_hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=fold_idxs,
            score_cutoff=score_cutoff,
        )
        if "selected_submissions" in top_hyperopt_dict:
            submissions = top_hyperopt_dict["selected_submissions"]
        else:
            submissions = []
        submissions_f_names = [f"submissions/{submission}" for submission in submissions]
    for submission_path in submissions_f_names:
        for fold_idx in fold_idxs:
            fold_path = Path(submission_path) / "training_output" / f"fold_{fold_idx}"
            for f_name in [
                fold_path / "y_pred_test.npz",
                fold_path / "y_pred_train.npz",
            ]:
                if f_name.is_file():
                    print(f"Deleting {f_name}")
                    os.remove(f_name)
