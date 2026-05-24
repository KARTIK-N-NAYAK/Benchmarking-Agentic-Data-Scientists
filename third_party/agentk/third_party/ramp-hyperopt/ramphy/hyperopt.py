"""Hyperparameter optimization for ramp-kits."""

import re
import os
import time
import glob
import json
import shutil
import hashlib
import itertools

import numpy as np
import pandas as pd
import rampwf as rw
import ray
from ray import tune, train
from tempfile import mkdtemp
from pathlib import Path
import warnings

# flake8: noqa: E501


from .engines import RandomEngine

HYPERPARAMS_SECTION_START = "# RAMP START HYPERPARAMETERS"
HYPERPARAMS_SECTION_END = "# RAMP END HYPERPARAMETERS"
HYPERPARAMS_REPL_REGEX = re.compile("{}.*{}".format(HYPERPARAMS_SECTION_START, HYPERPARAMS_SECTION_END), re.S)


def check_disk_space():
    total, used, free = shutil.disk_usage("/")
    usage_percent = (used / total) * 100
    if usage_percent > 94:
        raise Exception("Disk space is full, cannot continue.")


class Hyperparameter(object):
    """Discrete grid hyperparameter.

    Represented by a list of values, a default value, the name of the
    hyperparameter (specified by the user in the workflow element), the
    name of the workflow element in which the hyperparemeter appears, and an
    optional prior probability vector.

    Attributes:
        name : string
            The name of the hyperparameter variable, used in user interface,
            both for specifying the grid of values and getting the report on
            an experiment. Initialized to '' then set in set_names, to the
            name the user chose for the variable in the workflow element.
        workflow_element_name : string
            The name of the workflow element in which the hyperparameter is
            used. Initialized to '' then set in set_names.
        dtype : string
            The dtype of the hyperparameter.
        default_index: int
            The index in values of the current value of the hyperparameter.
        values: numpy array of any dtype
            The list of hyperparameter values.
        prior: numpy array of float
            A list of reals that the hyperopt can use as a prior probability
            over values. Positivity and summing to one are not checked,
            hyperparameter optimizers should do that when using the list
    """

    def __init__(self, dtype, default=None, values=None, prior=None, name=""):
        self.name = name
        self.workflow_element_name = ""
        self.dtype = dtype
        if default is None and values is None:
            raise ValueError("Either default or values must be defined.")
        if values is None:
            self.values = np.array([default], dtype=self.dtype)
        else:
            if len(values) < 1:
                raise ValueError("Values needs to contain at least one element.")
            self.values = np.array(values, dtype=self.dtype)
        if default is None:
            self.default_index = 0
        else:
            if default not in self.values:
                message = "Default must be among values.\n"
                message += f"default: {default}\n"
                message += f"values: {self.values}"
                raise ValueError(message)
            else:
                self.set_default(default)

        if prior is None:
            self.prior = np.array([1.0 / self.n_values] * self.n_values)
        else:
            if len(prior) != len(values):
                raise ValueError(f"len(values) == {len(values)} != {len(prior)} == len(prior)")
            self.prior = prior

    @property
    def n_values(self):
        """The number of hyperparameter values.

        Return:
            n_values : int
                The number of hyperparameter values len(values)
        """
        return len(self.values)

    @property
    def default(self):
        """The current value of the hyperparameter.

        Return:
            default : any dtype
                The current value of the hyperparameter values[default_index].
        """
        return self.values[self.default_index]

    @property
    def default_repr(self):
        """The string representation of the default value.

        It can be used to output the default value into a python file. For
        object types it adds '', otherwise it's the string representation of
        the default value.

        Return:
            default_repr : str
                The string representation of the default value.
        """
        if self.dtype in ["object", "str"]:
            return "'{}'".format(self.default)
        else:
            return str(self.default)

    @property
    def values_repr(self):
        """The string representation of the list of values.

        It can be used to output the list of values into a python file. For
        object types it adds '' around the values, otherwise it's the list of
        string representations of the values in brackets.

        Return:
            values_repr : list of str
                The string representation of the list of values.
        """
        s = "["
        for v in self.values:
            if self.dtype in ["object", "str"]:
                s += "'{}', ".format(v)
            else:
                s += "{}, ".format(v)
        s += "]"
        return s

    @property
    def python_repr(self):
        """The string representation of the hyperparameter.

        It can be used to output the hyperparameter definition into a python
        file:
        <name> = Hyperparameter(
            dtype=<dtype>, default=<default>, values=[<values>])

        Return:
            python_repr : str
                The string representation of the hyperparameter.
        """
        repr = "{} = Hyperparameter(\n".format(self.name)
        repr += "    dtype='{}'".format(str(self.dtype))
        repr += ", default={}".format(self.default_repr)
        repr += ", values={})\n".format(self.values_repr)
        return repr

    def set_names(self, name, workflow_element_name):
        """Set the name and workflow element name.

        Used when a hyperparameter object is loaded from a workflow element.

        Parameters:
            name : str
                The name of the hyperparameter, declared by the user in the
                workflow element.
            workflow_element_name : str
                The name of the workflow element in which the hyperparameter
                is defined.

        """
        self.name = name
        self.workflow_element_name = workflow_element_name

    def get_index(self, value):
        """Get the index of a value.

        Parameters:
            value : any dtype
                The value to look for.
        """
        if self.dtype == "float":
            float_list = list([abs(v - value) < 1e-15 for v in self.values])
            return float_list.index(True)
        else:
            return list(self.values).index(value)

    def set_default(self, default):
        """Set the default value.

        Parameters:
            default : any dtype
                The new default value.
        """
        self.default_index = self.get_index(default)

    def __int__(self):
        """Cast the default value into an integer.

        It can be used in the workflow element for an integer hyperparameter.

        Return:
            int(default) : int
                The integer representation of the default value.
        """
        return int(self.default)

    def __float__(self):
        """Cast the default value into an float.

        It can be used in the workflow element for an float hyperparameter.

        Return:
            float(default) : float
                The float representation of the default value.
        """
        return float(self.default)

    def __str__(self):
        """Cast the default value into a string.

        It can be used in the workflow element for a string hyperparameter.

        Return:
            str(default) : str
                The string representation of the default value.
        """
        return str(self.default)

    def __bool__(self):
        """Cast the default value into a bool.
        It can be used in the workflow element for a bool hyperparameter.
        Return:
            bool(default) : bool
                The string representation of the default value.
        """
        return bool(self.default)


def parse_hyperparameters(submission_path, workflow_element_name, in_play=True):
    """Parse hyperparameters in a workflow element.

    Load the module, take all Hyperparameter objects, and set the name of each
    to the name of the hyperparameter the user chose and the workflow element
    name of each to workflow_element_name.

    Parameters:
        submission_path : str
            The path to the submission directory.
        workflow_element_name : string
            The name of the workflow element.
        in_play : bool
            If False, only default value is in play.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    workflow_element = rw.utils.import_module_from_source(
        os.path.join(submission_path, workflow_element_name + ".py"), workflow_element_name
    )
    for object_name in dir(workflow_element):
        h = getattr(workflow_element, object_name)
        if type(h) == Hyperparameter:
            h.set_names(object_name, workflow_element_name)
            if in_play:
                h.start_value_index = 0
                h.stop_value_index = len(h.values)
            else:
                h.start_value_index = h.default_index
                h.stop_value_index = h.default_index + 1
            hyperparameters.append(h)
    return hyperparameters


def parse_all_hyperparameters(submission_path, workflow, workflow_element_names=None):
    """Parse hyperparameters in a submission.

    Load all the the modules, take all Hyperparameter objects, and set the name
    of each to the name of the hyperparameter the user chose and the workflow
    element name of each to the corresponding workflow_element_name.

    Parameters:
        submission_path : str
            The path to the submission directory.
        workflow : workflow from rw.workflows
            The ramp workflow.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    workflow.set_element_names(submission_path)
    if workflow_element_names is None:
        workflow_element_names = workflow.element_names
    for wen in workflow.element_names:
        in_play = wen in workflow_element_names
        hyperparameters += parse_hyperparameters(submission_path, wen, in_play)
    return hyperparameters


def write_hyperparameters_per_element(submission_path, output_submission_path, hs, wen):
    """Write hyperparameters in a submission.

    Read workflow elements from submission_path, replace the hyperparameter
    section with the hyperparameters in the hypers_per_workflow_element
    dictionary (with new hyperparamter values set by, e.g, a hyperopt engine),
    then write the new workflow elements into output_submission_path (which
    can be a temporary directory or submission_path itself when the function
    is called to replace the hyperparameters in the input submission with the
    best hyperparameters.)

    Parameters:
        submission_path : str
            The path to the submission directory from which the submission is
            read.
        output_submission_path : str
            The path to the output submission directory into which the
            submission with the new hyperparameter values is written.
        hs : list
            List of Hyperparameter instances
        wen : str
            Workflow element name
    """
    hyper_section = "{}\n".format(HYPERPARAMS_SECTION_START)
    for h in hs:
        hyper_section += h.python_repr
    hyper_section += HYPERPARAMS_SECTION_END
    f_name = os.path.join(submission_path, wen + ".py")
    with open(f_name) as f:
        content = f.read()
        content = HYPERPARAMS_REPL_REGEX.sub(hyper_section, content)
    Path(output_submission_path).mkdir(parents=True, exist_ok=True)
    output_f_name = Path(output_submission_path) / f"{wen}.py"
    with open(output_f_name, "w") as f:
        f.write(content)


def write_hyperparameters(submission_path, output_submission_path, hypers_per_workflow_element):
    """Write hyperparameters in a submission.

    Read workflow elements from submission_path, replace the hyperparameter
    section with the hyperparameters in the hypers_per_workflow_element
    dictionary (with new hyperparamter values set by, e.g, a hyperopt engine),
    then write the new workflow elements into output_submission_path (which
    can be a temporary directory or submission_path itself when the function
    is called to replace the hyperparameters in the input submission with the
    best hyperparameters.)

    Parameters:
        submission_path : str
            The path to the submission directory from which the submission is
            read.
        output_submission_path : str
            The path to the output submission directory into which the
            submission with the new hyperparameter values is written.
        hypers_per_workflow_element : dictionary
            Each key is a workflow element name and each value is a list of
            Hyperparameter instances, representing the hyperparemters in
            the workflow element.
    """
    for wen, hs in hypers_per_workflow_element.items():
        write_hyperparameters_per_element(submission_path, output_submission_path, hs, wen)


class HyperparameterOptimization(object):
    """A hyperparameter optimization.

    Attributes:
        hyperparameters: a list of Hyperparameters
        engine: a hyperopt engine
        ramp_kit_dir: the directory where the ramp kit is found
        submission_path: the directory where the submission to be optimized
            is found
    """

    def __init__(
        self,
        hyperparameters,
        engine,
        ramp_kit_dir,
        ramp_data_dir,
        submission_path,
        workflow_element_names,
        fold_idxs,
        data_label,
        test,
    ):
        self.hyperparameters = hyperparameters
        self.engine = engine
        self.ramp_kit_dir = ramp_kit_dir
        self.ramp_data_dir = ramp_data_dir
        self.problem = rw.utils.assert_read_problem(ramp_kit_dir)
        self.X_train, self.y_train, self.X_test, self.y_test = rw.utils.assert_data(
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            data_label=data_label,
        )
        self.cv = rw.utils.assert_cv(ramp_kit_dir, ramp_data_dir, data_label, fold_idxs)
        if fold_idxs is None:
            self.fold_idxs = list(range(len(self.cv)))
        else:
            self.fold_idxs = fold_idxs

        self.workflow = self.problem.workflow
        try:
            self.workflow.metadata = self.problem.get_metadata(ramp_data_dir, data_label)
        except AttributeError:
            print("No metadata")
        self.workflow.set_element_names(submission_path)

        if workflow_element_names is None:
            self.workflow_element_names = self.workflow.element_names
        else:
            self.workflow_element_names = workflow_element_names
            for en in self.workflow_element_names:
                if not en in self.workflow.element_names:
                    raise ValueError(f"{en} is not in workflow elements {self.workflow.element_names}")

        self.data_label = data_label
        # getting the root submission in case we start from a hyperopted submission
        if submission_path[-20:-10] == "_hyperopt_":
            submission_path = submission_path[:-20]
        self.submission_path = submission_path
        self.test = test

        self.hyperopt_output_path = os.path.join(self.submission_path, "hyperopt_output")
        if self.data_label is not None:
            self.hyperopt_output_path = os.path.join(self.hyperopt_output_path, self.data_label)
        if not os.path.exists(self.hyperopt_output_path):
            os.makedirs(self.hyperopt_output_path)

        self.hyperparameter_names = [h.name for h in hyperparameters]
        self.hyperparameters_indices = [h.name + "_i" for h in hyperparameters]
        self.score_names = [s.name for s in self.problem.score_types]
        self.df_summary_ = None
        self.fold_idx = 0

        # Set up hypers_per_workflow_element dictionary: keys are
        # workflow element names, values are lists are hypers belonging
        # to the workflow element
        self.hypers_per_workflow_element = {wen: [] for wen in self.workflow.element_names}
        for h in self.hyperparameters:
            self.hypers_per_workflow_element[h.workflow_element_name].append(h)

        scores_columns = ["hyperopt_submission", "fold_idx"]
        scores_columns += [f"hyper_{hn}" for hn in self.hyperparameter_names]
        scores_columns += [f"hyper_{hn}_i" for hn in self.hyperparameter_names]
        scores_columns += ["train_" + name for name in self.score_names]
        scores_columns += ["valid_" + name for name in self.score_names]
        scores_columns += ["train_time", "valid_time", "n_train", "n_valid"]
        dtypes = (
            ["str", "int"]
            + [h.dtype for h in self.hyperparameters]
            + ["int"] * len(self.hyperparameters)
            + ["float"] * 2 * len(self.score_names)
            + ["float"] * 2
            + ["int"] * 2
        )
        if self.test:
            scores_columns += ["test_" + name for name in self.score_names]
            scores_columns += ["test_time", "n_test"]
            dtypes = dtypes + (["float"] * len(self.score_names) + ["float"] + ["int"])

        self.df_scores_ = pd.DataFrame(columns=scores_columns)
        for column, dtype in zip(scores_columns, dtypes):
            self.df_scores_[column] = self.df_scores_[column].astype(dtype)

    def update_df_scores(self, output_submission, df_scores, fold_i):
        row = {"hyperopt_submission": output_submission, "fold_idx": self.fold_idxs[fold_i]}
        for h in self.hyperparameters:
            row[f"hyper_{h.name}"] = h.default
            row[f"hyper_{h.name}_i"] = h.default_index
        for name in self.score_names:
            row["train_" + name] = df_scores.loc["train"][name]
            row["valid_" + name] = df_scores.loc["valid"][name]
            if self.test:
                row["test_" + name] = df_scores.loc["test"][name]
        row["train_time"] = float(df_scores.loc["train"]["time"])
        row["valid_time"] = float(df_scores.loc["valid"]["time"])
        row["n_train"] = len(self.cv[fold_i][0])
        row["n_valid"] = len(self.cv[fold_i][1])
        if self.test:
            row["test_time"] = float(df_scores.loc["test"]["time"])
            row["n_test"] = len(self.X_test)

        self.df_scores_.loc[len(self.df_scores_)] = pd.Series(row)
        self.df_scores_["fold_idx"] = self.df_scores_["fold_idx"].astype(int)
        for h in self.hyperparameters:
            col = f"hyper_{h.name}_i"
            self.df_scores_[col] = self.df_scores_[col].astype(int)

    def preprocess_data(self, submission_path):
        self.X_train_preprocessed, self.y_train_preprocessed, self.X_test_preprocessed = self.workflow.preprocess_data(
            submission_path, self.X_train, self.y_train, self.X_test
        )

    def run_next_experiment(self, submission_path, fold_i, save_output=False):
        if save_output:
            training_output_path = Path(submission_path) / "training_output"
            training_output_path.mkdir(parents=True, exist_ok=True)
            print("Training output path: {}".format(training_output_path))
            fold_output_path = training_output_path / f"fold_{self.fold_idxs[fold_i]}"
            fold_output_path.mkdir(parents=True, exist_ok=True)
        else:
            fold_output_path = "."
        _, _, df_scores = rw.utils.run_submission_on_cv_fold(
            self.problem,
            submission_path=submission_path,
            fold=self.cv[fold_i],
            X_train=self.X_train_preprocessed,
            y_train=self.y_train_preprocessed,
            X_test=self.X_test_preprocessed,
            y_test=self.y_test,
            save_output=save_output,
            fold_output_path=fold_output_path,
        )
        return df_scores

    def make_and_save_summary(self, path=None, fname="summary.csv", append=False):
        if path is None:
            path = self.hyperopt_output_path
        summary_fname = Path(path) / fname
        if append:
            self.df_scores_.to_csv(summary_fname, mode="a", header=not os.path.exists(summary_fname))
        else:
            self.df_scores_.to_csv(summary_fname)

    def load_summary(self, path=None, fname="summary.csv"):
        if path is None:
            path = self.hyperopt_output_path
        summary_fname = Path(path) / fname
        if summary_fname.is_file():
            self.df_scores_ = pd.read_csv(summary_fname, index_col=0)

    def save_best_model(self):
        official_scores = self.df_summary_["valid_" + self.problem.score_types[0].name + "_m"]
        print("official scores", official_scores)
        if self.problem.score_types[0].is_lower_the_better:
            best_defaults = official_scores.idxmin()
        else:
            best_defaults = official_scores.idxmax()
        print("Best hyperparameters: ", best_defaults)
        try:
            for bd, h in zip(best_defaults, self.hyperparameters):
                h.set_default(bd)
        except TypeError:
            # single hyperparameter
            self.hyperparameters[0].set_default(best_defaults)
        # Overwrite the submission with the best hyperparameter values
        write_hyperparameters(self.submission_path, self.submission_path, self.hypers_per_workflow_element)


def run(hyperparameter_experiment, n_trials, resume=False):
    start_iter = 0
    if resume:
        hyperparameter_experiment.load_summary(hyperparameter_experiment.hyperopt_output_path)
        start_iter = len(hyperparameter_experiment.df_scores_)
    start = pd.Timestamp.now()
    for i_iter in range(start_iter, n_trials):
        # Getting new hyperparameter values from engine
        (
            fold_idx,
            next_value_indices,
        ) = hyperparameter_experiment.engine.next_hyperparameter_indices(
            hyperparameter_experiment.df_scores_,
            len(hyperparameter_experiment.cv),
            hyperparameter_experiment.problem,
        )
        # Updating hyperparameters
        for h, i in zip(hyperparameter_experiment.hyperparameters, next_value_indices):
            h.default_index = i
        # Writing submission files with new hyperparameter values
        output_submission_path = mkdtemp()
        write_hyperparameters(
            hyperparameter_experiment.submission_path,
            output_submission_path,
            hyperparameter_experiment.hypers_per_workflow_element,
        )
        # Calling the training script.

        hyperparameter_experiment.preprocess_data(output_submission_path)
        df_scores = hyperparameter_experiment.run_next_experiment(output_submission_path, fold_idx)
        sn = hyperparameter_experiment.score_names[0]
        hyperparameter_experiment.engine.pass_feedback(fold_idx, len(hyperparameter_experiment.cv), df_scores, sn)
        hyperparameter_experiment.update_df_scores(df_scores, fold_idx)
        shutil.rmtree(output_submission_path)
        now = pd.Timestamp.now()
        eta = start + (now - start) / (i_iter + 1 - start_iter) * (n_trials - start_iter)
        print(f"Done {i_iter + 1} / {n_trials} at {now}. ETA = {eta}.")
        hyperparameter_experiment.make_and_save_summary()
    scores_columns = ["valid_" + name for name in hyperparameter_experiment.score_names]
    for score in scores_columns:
        hyperparameter_experiment.df_scores_[score + "_max"] = (
            hyperparameter_experiment.df_scores_[score].rolling(n_trials, min_periods=1).max()
        )


def objective(config, run_params=None):
    hyperparameter_experiment = run_params["hyperparameter_experiment"]
    for h in hyperparameter_experiment.hyperparameters:
        h.default_index = config[h.name]
    hyper_indices = [h.default_index for h in hyperparameter_experiment.hyperparameters]
    hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
    output_submission_path = f"{hyperparameter_experiment.submission_path}_hyperopt_{hyper_hash}"
    output_submission = Path(output_submission_path).name
    os.chdir(run_params["current_dir"])
    write_hyperparameters(
        hyperparameter_experiment.submission_path,
        output_submission_path,
        hyperparameter_experiment.hypers_per_workflow_element,
    )
    # Calling the training script.
    hyperparameter_experiment.preprocess_data(output_submission_path)
    valid_scores = np.zeros(len(hyperparameter_experiment.cv))
    df_scores_list = []
    for fold_i in range(len(hyperparameter_experiment.cv)):
        df_scores = hyperparameter_experiment.run_next_experiment(
            output_submission_path, fold_i, run_params["save_output"]
        )
        sn = hyperparameter_experiment.score_names[0]
        valid_scores[fold_i] = df_scores.loc["valid", sn]
        df_scores_list.append(df_scores)
        hyperparameter_experiment.update_df_scores(output_submission, df_scores, fold_i)

    hyperparameter_experiment.make_and_save_summary(append=True)

    train.report(
        {
            "valid_score": valid_scores.mean(),
            "df_scores_list": df_scores_list,
            "output_submission": output_submission,
        }
    )


def run_tune(
    hyperparameter_experiment,
    n_trials,
    max_concurrent_runs,
    n_cpu_per_run,
    n_gpu_per_run,
    save_output,
    verbose,
):
    is_lower_the_better = hyperparameter_experiment.problem.score_types[0].is_lower_the_better
    engine_mode = "min" if is_lower_the_better else "max"

    if hyperparameter_experiment.engine.name == "ray_grid_search":
        warnings.warn("A full grid search is being used with RAY's default engine !")
        config = {
            h.name: tune.grid_search(range(h.start_value_index, h.stop_value_index))
            for h in hyperparameter_experiment.hyperparameters
        }
        num_samples = int(len(hyperparameter_experiment.cv))
    else:
        config = {
            h.name: tune.randint(h.start_value_index, h.stop_value_index)
            for h in hyperparameter_experiment.hyperparameters
        }
        num_samples = int(n_trials / len(hyperparameter_experiment.cv))

    run_params = {
        "current_dir": os.getcwd(),
        "hyperparameter_experiment": hyperparameter_experiment,
        "save_output": save_output,
    }
    tune_name = (
        f"{hyperparameter_experiment.engine.name}__"
        + f'{hyperparameter_experiment.submission_path.split("/")[-1]}__'
        + f"{hyperparameter_experiment.data_label}"
    )
    results = tune.run(
        tune.with_parameters(objective, run_params=run_params),
        max_concurrent_trials=max_concurrent_runs,
        metric="valid_score",
        mode=engine_mode,
        num_samples=num_samples,
        name=tune_name,
        search_alg=hyperparameter_experiment.engine.ray_engine,
        config=config,
        verbose=verbose,
        resources_per_trial={"cpu": n_cpu_per_run, "gpu": n_gpu_per_run},
        #        local_dir=f"./{(Path(hyperparameter_experiment.hyperopt_output_path) / 'ray_results').as_posix()}",
    )

    for _, row in results.results_df.iterrows():
        for h in hyperparameter_experiment.hyperparameters:
            h.default_index = int(row[f"config/{h.name}"])

    results.results_df.to_csv(os.path.join(Path(hyperparameter_experiment.hyperopt_output_path), "ray_summary.csv"))
    # the newly created submissions
    return list(results.results_df["output_submission"].to_numpy())


class RayEngine:
    # n_trials is only needed by zoopt at init time
    def __init__(
        self,
        engine_name,
        n_trials=None,
        points_to_evaluate=None,
        evaluated_rewards=None,
    ):
        self.name = engine_name
        if (engine_name[4:] == "random") or (engine_name[4:] == "grid_search"):
            self.ray_engine = None
        elif engine_name[4:] == "zoopt":
            try:
                from ray.tune.search.zoopt import ZOOptSearch

                self.ray_engine = ZOOptSearch(  # gets stuck often
                    algo="Asracos",  # only support ASRacos currently
                    budget=10,
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except("zoopt")
        elif engine_name[4:] == "ax":
            try:
                from ray.tune.search.ax import AxSearch

                self.ray_engine = AxSearch(
                    points_to_evaluate=points_to_evaluate,
                )
            except ModuleNotFoundError:
                self.raise_except("ax-platform sqlalchemy")
        elif engine_name[4:] == "skopt":
            try:
                from ray.tune.search.skopt import SkOptSearch

                self.ray_engine = SkOptSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except("scikit-optimize")
        elif engine_name[4:] == "hyperopt":
            try:
                from ray.tune.search.hyperopt import HyperOptSearch

                self.ray_engine = HyperOptSearch(
                    points_to_evaluate=points_to_evaluate,
                )
            except ModuleNotFoundError:
                self.raise_except("hyperopt")
        elif engine_name[4:] == "bayesopt":
            try:
                from ray.tune.search.bayesopt import BayesOptSearch

                self.ray_engine = BayesOptSearch(
                    points_to_evaluate=points_to_evaluate,
                )
            except ModuleNotFoundError:
                self.raise_except("bayesian-optimization")
        elif engine_name[4:] == "bohb":
            try:
                from ray.tune.search.bohb import TuneBOHB

                self.ray_engine = TuneBOHB(
                    points_to_evaluate=points_to_evaluate,
                )
            except ModuleNotFoundError:
                self.raise_except("hpbandster")
        elif engine_name[4:] == "nevergrad":
            try:
                from ray.tune.search.nevergrad import NevergradSearch
                import nevergrad as ng

                self.ray_engine = NevergradSearch(
                    optimizer=ng.optimizers.OnePlusOne,
                    points_to_evaluate=points_to_evaluate,
                )
            except ModuleNotFoundError:
                self.raise_except("nevergrad")
        elif engine_name[4:] == "hebo":
            try:
                from ray.tune.search.hebo import HEBOSearch

                self.ray_engine = HEBOSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except("hebo")
        elif engine_name[4:] == "optuna":
            try:
                from ray.tune.search.optuna import OptunaSearch

                self.ray_engine = OptunaSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except("optuna")
        else:
            raise ValueError(f"Engine {engine_name[4:]} not found in Ray Tune")

    def raise_except(library):
        raise EnvironmentError("Missing module: install it using pip install " + library)


def init_hyperopt(
    ramp_kit_dir,
    ramp_data_dir,
    ramp_submission_dir,
    submission,
    engine_name,
    workflow_element_names,
    fold_idxs,
    data_label,
    label,
    resume,
    test,
    n_trials=None,
):
    # n_trials is only needed by ray_zoopt at init time
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    submission_path = os.path.join(ramp_submission_dir, submission)
    hyperparameters = parse_all_hyperparameters(submission_path, problem.workflow, workflow_element_names)
    if engine_name == "random":
        engine = RandomEngine(hyperparameters)
    elif engine_name.startswith("ray_"):
        evaluated_rewards = None
        points_to_evaluate = None
        if resume:
            print("\n-------------- Resuming previous trials --------------")
            # dummy experiment just to load existing scores
            engine = RayEngine(engine_name, n_trials)
            hyperparameter_experiment = HyperparameterOptimization(
                hyperparameters,
                engine,
                ramp_kit_dir,
                ramp_data_dir,
                submission_path,
                workflow_element_names,
                fold_idxs,
                data_label,
                test,
            )
            hyperparameter_experiment.load_summary()
            valid_score_name = f"valid_{problem.score_types[0].name}"
            cols = ["hyperopt_submission", "fold_idx", valid_score_name]
            cols += [f"hyper_{h.name}_i" for h in hyperparameters]
            summary_df = hyperparameter_experiment.df_scores_[cols]
            prev_trials_group = summary_df.groupby("hyperopt_submission")
            mean_df = prev_trials_group.mean()
            # non-bulletproof test for submissions that have all folds trained
            mean_df = mean_df[mean_df["fold_idx"] == np.array(hyperparameter_experiment.fold_idxs).mean()]
            mean_df = mean_df.rename(columns={f"hyper_{h.name}_i": h.name for h in hyperparameters})
            mean_df = mean_df.astype({h.name: "int" for h in hyperparameters})

            evaluated_rewards = list(mean_df[valid_score_name].to_numpy())
            points_to_evaluate = mean_df[[h.name for h in hyperparameters]].to_dict("records")
            print(f"Found {len(points_to_evaluate)} existing subissions, resuming.")
            print("-------------- Done --------------\n")
        engine = RayEngine(engine_name, n_trials, points_to_evaluate, evaluated_rewards)
    else:
        raise ValueError(f"{engine_name} is not a valid engine name")
    hyperparameter_experiment = HyperparameterOptimization(
        hyperparameters,
        engine,
        ramp_kit_dir,
        ramp_data_dir,
        submission_path,
        workflow_element_names,
        fold_idxs,
        data_label,
        test,
    )

    return hyperparameter_experiment


def run_hyperopt(
    ramp_kit_dir,
    ramp_data_dir,
    ramp_submission_dir,
    data_label,
    submission,
    engine_name,
    n_trials,
    workflow_element_names,
    fold_idxs,
    save_output,
    test,
    label,
    resume,
    max_concurrent_runs,
    n_cpu_per_run,
    n_gpu_per_run,
    verbose,
):
    try:
        check_disk_space()
        # Continue with your Ray Tune experiment...
    except Exception as e:
        print(e)
        print("rm -rf /tmp/ray/*")
        print("rm -rf ~/ray_results/*")
        exit()

    ray.init(log_to_driver=False, ignore_reinit_error=True)
    if n_cpu_per_run is None:
        n_cpu_per_run = os.cpu_count() - 1

    if n_gpu_per_run is None:
        n_gpu_per_run = len(ray.get_gpu_ids())  # Get the number of GPUs available

    hyperparameter_experiment = init_hyperopt(
        ramp_kit_dir,
        ramp_data_dir,
        ramp_submission_dir,
        submission,
        engine_name,
        workflow_element_names,
        fold_idxs,
        data_label,
        label,
        resume,
        test,
    )
    if engine_name.startswith("ray_"):
        output_submissions = run_tune(
            hyperparameter_experiment,
            n_trials,
            max_concurrent_runs,
            n_cpu_per_run,
            n_gpu_per_run,
            save_output,
            verbose,
        )
        return output_submissions
    else:
        run(hyperparameter_experiment, n_trials, resume)
