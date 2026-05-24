import click
import click_config_file

from ..hyperopt import run_hyperopt

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--submission",
    default="starting_kit",
    show_default=True,
    help="The kit to hyperopt. It should be located in the "
    '"submissions" folder of the starting kit.',
)
@click.option(
    "--ramp-kit-dir",
    default=".",
    show_default=True,
    help="Root directory of the ramp-kit to hyperopt.",
)
@click.option(
    "--ramp-data-dir",
    default=".",
    show_default=True,
    help="Directory containing the data. This directory should "
    'contain a "data" folder.',
)
@click.option(
    "--data-label",
    default=None,
    show_default=True,
    help="A label specifying the data in case the same submissions "
    "are executed on multiple datasets. If specified, "
    "problem.get_train_data and problem.get_test_data should "
    "accept a data_label argument. Typically they can deal with "
    "multiple datasets containing the data within the directory "
    "specified by --ramp-data-dir (default: ./data), for example "
    "using subdirectories ./data/<data_label>/. It is also "
    "the subdirectory of submissions/<submission>/training_output "
    "where results are saved",
)
@click.option(
    "--ramp-submission-dir",
    default="submissions",
    show_default=True,
    help="Directory where the submissions are stored. It is the "
    'directory (typically called "submissions" in the ramp-kit) '
    "that contains the individual submission subdirectories.",
)
@click.option(
    "--engine",
    default="random",
    show_default=True,
    help='The name of the hyperopt engine, e.g., "random".',
)
@click.option(
    "--n-trials",
    default=10,
    show_default=True,
    help="The number of hyperopt iterations, inputted to the "
    "engine.",
)
@click.option(
    "--n-folds",
    default=3,
    show_default=True,
    help="The number of folds used in the hyperopt.",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click.option(
    "--workflow-elements-to-hyperopt",
    multiple=True,
    default=[],
    help="A list of workflow elements, typically python files in this folder "
    "(without the extension) which have a hyperparameter grid defined at the top.",
)
@click.option(
    "--save-output",
    is_flag=True,
    default=True,
    show_default=True,
    help="Specify this flag to create a "
    "<submission>_<data_label>_hyperopt_<timestamp> "
    "(or <submission>_hyperopt_<timestamp> if <data_label> is None)"
    'in the "submissions" dir with all the submissions.',
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    show_default=True,
    help="If true, the saving folder should be "
    "<submission>_<data_label>_<engine>_hyperopt",
)
@click.option("--label", is_flag=True, default=False, show_default=True, help="")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    show_default=True,
    help="To resume a broken run. If False, summary.csv will "
    "be created from scratch.",
)
@click.option(
    "--max-concurrent-runs",
    default=1,
    show_default=True,
    help="[ray] Maximum number of trials to run concurrently. Must be non-negative. If "
         "None or 0, no limit will be applied. default to 1.",
)
@click.option(
    "--n-cpu-per-run",
    default=1,
    show_default=True,
    help="[ray] Machine resources (cpu) to allocate per trial. default to 1",
)
@click.option(
    "--n-gpu-per-run",s
    default=0,
    show_default=True,
    help="[ray] Machine resources (gpu) to allocate per trial. default to 0",
)
@click.option(
    "--verbose",
    default=3,
    show_default=True,
    help="[ray] 0, 1, 2, or 3. Verbosity mode. 0 = silent, 1 = only status updates, "
         "2 = status and brief trial results, 3 = status and detailed trial results. "
         "Defaults to 3.",
)
@click_config_file.configuration_option()
def main(
    submission,
    ramp_kit_dir,
    ramp_data_dir,
    data_label,
    ramp_submission_dir,
    engine,
    n_trials,
    n_folds,
    first_fold_idx,
    workflow_elements_to_hyperopt,
    save_output,
    test,
    label,
    resume,
    max_concurrent_runs,
    n_cpu_per_run,
    n_gpu_per_run,
    verbose,
):
    """Hyperopt a submission."""
    run_hyperopt(
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        ramp_submission_dir=ramp_submission_dir,
        data_label=data_label,
        submission=submission,
        engine_name=engine,
        n_trials=n_trials * n_folds,
        workflow_element_names=workflow_elements_to_hyperopt,
        fold_idxs=range(first_fold_idx, first_fold_idx + n_folds),
        save_output=save_output,
        test=test,
        label=label,
        resume=resume,
        max_concurrent_runs=max_concurrent_runs,
        n_cpu_per_run=n_cpu_per_run,
        n_gpu_per_run=n_gpu_per_run,
        verbose=verbose,
    )


def start():
    main()


if __name__ == "__main__":
    start()
