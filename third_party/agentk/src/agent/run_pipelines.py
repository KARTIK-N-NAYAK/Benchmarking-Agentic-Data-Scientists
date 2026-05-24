import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path

from jsonlines import jsonlines
from pyrootutils import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from agent import PROJECT_ROOT
from third_party.data_preprocessing.env import DataPrepEnv
from agent.utils.utils import run_command
from ds_agent.results_processing.ramp_cot_builder import build_ramp_cot

MAX_SETUPS = 20


def check_for_success(run_dir: Path, match_patterns: list[str]) -> bool:
    """
    Check if any files matching given patterns exist within a directory.

    Args:
        run_dir: The directory path to search in.
        match_patterns: List of glob patterns to match files against.

    Returns:
        True if any file matching any of the patterns is found in run_dir, False otherwise.
    """
    any_ramp_files_found = any(
        glob.glob(str(run_dir / pattern)) for pattern in match_patterns
    )
    return any_ramp_files_found


def is_setup_pipeline_successful(setup_dir: Path, is_tabular: bool) -> bool:
    """
    Checks whether the final unit test passed and the submission.csv file was created.

    Args:
        setup_dir:  The setup directory where the final unit test creates its workspace.
        is_tabular: Is the competition tabular or not

    Returns:
        bool: Status of the setup pipeline.
    """
    ramp_pipeline_files = [
        "./final_unit_test_vtest_n0/submissions/starting_kit/training_output/bagged_scores.csv",
        "./final_unit_test_vtest_n0/submissions/starting_kit/training_output/submission_bagged_test.csv",
        "./final_unit_test_vtest_n0/submissions/starting_kit/training_output/submission_bagged_valid.csv"
    ]

    ds_pipeline_patterns = [
        "./*/submissions/*/submission.csv",
        "./*/submissions/*/submission_alt.csv"
    ]
    if is_tabular:
        is_successful = check_for_success(run_dir=setup_dir, match_patterns=ramp_pipeline_files)
    else:
        is_successful = check_for_success(run_dir=setup_dir, match_patterns=ds_pipeline_patterns)

    return is_successful


def is_main_pipeline_successful(run_dir: Path, is_tabular: bool) -> bool:
    """
    Checks whether the final unit test passed and the submission.csv file was created.

    Args:
        run_dir: The run directory where it generated files .
        is_tabular: Is the competition tabular or not

    Returns:
        bool: Status of the main pipeline.
    """
    ramp_pipeline_files = [
        "./ramp_kit_*/submissions/training_output/submission_bagged_then_blended_*.csv"
    ]

    ds_pipeline_patterns = [
        "./submissions/*/submission.csv",
        "./submissions/*/submission_alt.csv"
    ]

    if is_tabular:
        is_successful = check_for_success(run_dir=run_dir, match_patterns=ramp_pipeline_files)
    else:
        is_successful = check_for_success(run_dir=run_dir, match_patterns=ds_pipeline_patterns)

    return is_successful


def write_tab_cot_summary(setup_dir: Path) -> None:
    """
    Generate a Chain-of-Thought (CoT) summary for a ramp workspace and save it to a file.
    Args:
        setup_dir: The path to the setup directory containing ramp workspaces.

    Returns:
        None
    """
    ramp_workspace_patterns = [setup_dir / "ramp_kit_v*", setup_dir / "ramp_kitv*"]
    ramp_workspaces = [
        p for ramp_workspace_pattern in ramp_workspace_patterns for p in glob.glob(str(ramp_workspace_pattern))
    ]

    assert len(ramp_workspaces) > 0

    ramp_workspaces = Path(ramp_workspaces[0])
    cot_path = setup_dir / "summary.txt"
    cot_content = build_ramp_cot(ramp_workspaces)
    with open(cot_path, 'w') as f:
        f.write(cot_content)


def run_setup_pipeline(
        prep_task: str,
        prep_method: str,
        llm: str,
        code_llm: str,
        task_id: str,
        is_local_task: bool,
        alt_raw_data_root: str,
        total_time: float,
        max_setups: int,
        is_tabular: bool,
        exp_dir: Path,
        setup_log_dir: Path,
        use_final_unit_test: bool = True,
        default_response_path: str = None
) -> tuple[bool, float, Path]:
    """
    Runs the setup pipeline for a given task configuration by repeatedly attempting to set up
    the environment until success, time limit, or maximum setup attempts are reached.

    Args:
        prep_task: The data preparation task name
        prep_method: The data preparation method name
        llm: Config identifier of the language model used for reasoning.
        code_llm: Config identifier of the language model used for code generation.
        task_id: Data science task identifier.
        is_local_task:Whether the task is local rather than from kaggle
        total_time: Maximum total time allowed for all setup attempts (in seconds).
        max_setups:Maximum number of setup attempts allowed.
        is_tabular: Indicates if the task is fully tabular
        exp_dir: Base directory for experiment output and workspace.
        setup_log_dir:Directory to store logs for each setup attempt.
        use_final_unit_test: Whether to use the final unit test for validation. Defaults to True.
        alt_raw_data_root: Alternative data root path for raw task input.
        default_response_path:Path to a file containing pre-registered LLM responses.

    Returns:
        tuple[bool, float, Path]: A tuple containing:
             - is_setup_successful : Whether the setup was successful.
             - execution_time: Time taken to complete setup.
             - setup_dir: Path to the final setup directory.

    """

    print("Running setup...")
    setup_start_time = time.time()
    execution_time = 0
    setup_version = 0
    is_setup_successful = False
    setup_dir = exp_dir / f"seed_{setup_version}"

    while not is_setup_successful and execution_time < total_time and setup_version < max_setups:
        setup_dir = exp_dir / f"seed_{setup_version}"
        if setup_dir.exists() and setup_dir.is_dir():
            shutil.rmtree(setup_dir, ignore_errors=True)
        if setup_dir.parent.exists():
            print(f"workspace exists {setup_dir.parent} exists")
        setup_dir.mkdir(parents=True, exist_ok=True)

        setup_command = (
            f"HYDRA_FULL_ERROR=1 python ./src/agent/start.py "
            f"--config-name think_and_code_llm_sa_eval "
            f"task={prep_task} method={prep_method} "
            f"max_episodes=5 "
            f"llm@agent.llm={llm} "
            f"llm@agent.code_llm={code_llm} "
            f"task.task_id={task_id} "
            f"task.workspace_path={setup_dir} "
            f"task.use_final_unit_test={use_final_unit_test} "
            f"hydra.run.dir={setup_log_dir}/seed_{setup_version} "
            f"task.is_local_task={is_local_task} "
        )
        if alt_raw_data_root:
            setup_command = f"ALT_RAW_DATA_ROOT={alt_raw_data_root} " + setup_command

        if default_response_path:
            setup_command += f"+agent.read_answer_from_file_path={default_response_path} "

        print(setup_command, flush=True)
        setup_command_path = setup_dir / "setup_command.txt"
        with open(setup_command_path, 'w') as f:
            f.write(setup_command)
        run_output = run_command(setup_command)

        if run_output is None:
            print(f"[Warning] Setup failed at version {setup_version}")
            setup_version += 1
            continue

        path_to_chat_retrial_time = DataPrepEnv.get_path_to_chat_completion_retrial_time(
            workspace_path=str(setup_dir)
        )
        if os.path.exists(path_to_chat_retrial_time):
            with open(path_to_chat_retrial_time, "r") as f:
                chat_completion_retrial_time = float(f.readline())
        else:
            chat_completion_retrial_time = 0

        print(f"Spent {chat_completion_retrial_time:.0f} seconds trying to do chat completion without success.")
        if is_setup_pipeline_successful(setup_dir, is_tabular=is_tabular):
            print("Setup completed successfully.", flush=True)
            is_setup_successful = True
            with open(setup_dir / "setup_done.txt", 'w') as f:
                f.write("True")
        else:
            print(f"Setup attempt {setup_version} unsuccessful.", flush=True)
            setup_version += 1

    execution_time = time.time() - setup_start_time - chat_completion_retrial_time

    return is_setup_successful, execution_time, setup_dir


def run_ds_main_pipeline(
        llm: str,
        ds_method: str,
        task_id: str,
        setup_dir: Path,
        main_log_dir: Path,
        working_dir: Path,
        time_for_main_pipeline: float,
        use_ci_handling: bool,
        alt_raw_data_root: str,
        blend_after_n: int,
        max_time_per_submission: float,
        is_local_task: bool,
        allow_default_response: bool = False,
        debug_mode: bool = False,
        default_response_path: str = None,
        terminate_after_training: bool = False,
) -> bool:
    """
    Runs the main data science pipeline using the given setup directory
    Args:
        llm: Config identifier of the language model
        ds_method: The data science method name
        task_id: Data science task identifier.
        setup_dir: Directory where setup is available.
        main_log_dir: Directory for logging pipeline output
        working_dir: Workspace root path.
        time_for_main_pipeline: Time limit for main pipeline execution.
        use_ci_handling: whether to run with class imbalance handling
        alt_raw_data_root: Alternative data root path for raw task input.
        blend_after_n: number of submissions after which blending is triggered if it was not yet chosen
        max_time_per_submission: Time limit for each submission created during the main pipeline
        is_local_task: Whether the task is local rather than from kaggle
        allow_default_response: Whether to allow default responses.
        debug_mode: Whether debug mode is enabled.
        default_response_path: Path to a file containing pre-registered LLM responses.
        terminate_after_training: Terminate the pipeline after training a model.

    Returns:
        True if the main pipeline completed successfully and predictions were generated,False otherwise.

    """

    if "AGENT_DEBUG" in os.environ:
        debug_mode = os.getenv("AGENT_DEBUG")

    if "ALLOW_DEFAULT_RESPONSE" in os.environ:
        allow_default_response = os.getenv("ALLOW_DEFAULT_RESPONSE")

    ci_env_val = "True" if use_ci_handling else "False"

    if blend_after_n is None:
        blend_after_n = 500

    workspace_path = working_dir / setup_dir.name / 'main_pipeline'
    main_pipeline_command = (
        f"HYDRA_FULL_ERROR=1 "
        f"ALLOW_DEFAULT_RESPONSE={allow_default_response} "
        f"AGENT_DEBUG={debug_mode} "
        f"TOKENIZERS_PARALLELISM=0 "
        f"BLEND_AFTER_N={blend_after_n} "
        f"TTA=1 "
        f"MAX_TIME_PER_SUBMISSION={max_time_per_submission} "
        f"USE_CI_HANDLING={ci_env_val} "
        f"python ./src/agent/start.py "
        f"task=data_science_interact "
        f"llm@agent.llm={llm} "
        f"method={ds_method} "
        f"max_episodes=5 "
        f"task.task_id={task_id} "
        f"task.prepared_version={setup_dir.name} "
        f"task.prepared_setup_dir={setup_dir.parent.parent} "
        f"task.workspace_path={workspace_path} "
        f"hydra.run.dir={str(main_log_dir)} "
        f"task.is_local_task={is_local_task} "
        f"task.terminate_after_training={terminate_after_training} "
        f"+task.max_exec_time={time_for_main_pipeline} "
    )
    if alt_raw_data_root:
        main_pipeline_command = f"ALT_RAW_DATA_ROOT={alt_raw_data_root} " + main_pipeline_command

    if default_response_path:
        main_pipeline_command += f"+agent.read_answer_from_file_path={default_response_path} "

    main_command_path = setup_dir / "main_pipeline_command.txt"
    with open(main_command_path, 'w') as f:
        f.write(main_pipeline_command)

    print(f"Running main pipeline with a time limit of {time_for_main_pipeline} seconds...", flush=True)
    print(main_pipeline_command, flush=True)
    run_command(main_pipeline_command)
    return is_main_pipeline_successful(run_dir=workspace_path, is_tabular=False)


def run_ds_tabular_ramp(
        llm: str, task_id: str, setup_dir: Path, setup_version: int, time_for_main_pipeline: float, max_cpu: int
) -> bool:
    """
    Executes the tabular pipeline using the RAMP framework.
    Args:
        llm: Config identifier of the language model
        task_id: Unique identifier for the competition or task.
        setup_dir: Directory where setup is available.
        setup_version: Representing the setup version number.
        time_for_main_pipeline:
        max_cpu: Time limit for main pipeline execution.

    Returns:
        True if the main tabular pipeline completed successfully and predictions were generated,False otherwise.

    """
    time_for_main_pipeline = time_for_main_pipeline / 3600
    name_version = llm.split('/')[-1]
    agent_root = "."
    cmd_save_path = setup_dir / "main_pipeline_command.txt"
    ramp_pre_kit_command = (
        f"python {agent_root}/third_party/data_science/pre-kit-script.py "
        f"--root_folder {setup_dir} "
        f"--challenge_name {task_id} "
        f"--output_path {setup_dir}/ramp_kit"
    )
    ramp_setup_command = (
        f"ramp-setup --ramp-kit ramp_kit "
        f"--version {name_version} "
        f"--number {setup_version} "
        f"--kit-root {setup_dir} "
        f"--setup-root {setup_dir}"
    )
    # Subtracting the time taken to create the ramp setup from the total time
    ramp_commands = [ramp_pre_kit_command, ramp_setup_command]
    with open(cmd_save_path, 'w') as f:
        for c in ramp_commands:
            f.write(c)
            f.write("\n\n")

    start_time = time.time()
    run_command(" && ".join(ramp_commands))
    elapsed_secs = time.time() - start_time
    elapsed_hours = elapsed_secs / 3600
    remaining_hours = max(time_for_main_pipeline - elapsed_hours, 0)
    ramp_run_hyperopt_race_command = (
        f"ramp-hyperopt-race --ramp-kit ramp_kit "
        f"--version {name_version} "
        f"--number {setup_version} "
        f"--kit-root {setup_dir} "
        f"--n-rounds 1000 "
        f"--n-trials-per-round 1 "
        f"--n-folds-final-blend 30 "
        f"--max-time {remaining_hours} "
        f"--n-cpu-per-run {max_cpu} "
    )
    with open(cmd_save_path, 'a') as f:
        f.write(ramp_run_hyperopt_race_command + "\n\n")

    if remaining_hours == 0:
        print("No time left to run the full pipeline.")
        return False

    print(f"Running:\n{ramp_run_hyperopt_race_command}")
    run_command(ramp_run_hyperopt_race_command)

    final_predictions_path = setup_dir / f"ramp_kit_v{name_version}_n{setup_version}" / "final_test_predictions"
    if final_predictions_path.exists():
        selected_submissions = []
        for f in final_predictions_path.iterdir():
            if "valid" not in f.name and "bagged_then_blended" in f.name:
                selected_submissions.append(str(f))
            if "valid" not in f.name and "last_blend" in f.name:
                selected_submissions.append(str(f))
        with jsonlines.open(final_predictions_path.parent / "selected_submission.jsonl", "w") as writer:
            writer.write(
                {
                    "submission_path": selected_submissions[0] if selected_submissions else None,
                    "competition_id": task_id
                }
            )

    is_successful = is_main_pipeline_successful(run_dir=setup_dir, is_tabular=True)

    if is_successful:
        write_tab_cot_summary(setup_dir)

    return is_successful


def get_attempt_path(workspace_name: str, task_id: str, attempt: int | None, attempt_spec: str) -> Path:
    if Path(workspace_name).is_absolute():
        working_dir = Path(workspace_name)
    else:
        working_dir = PROJECT_ROOT / workspace_name

    if attempt is not None:
        exp_dir = Path(working_dir, f"attempt_{attempt}{attempt_spec}", task_id)
    else:
        exp_dir = Path(working_dir, task_id)

    return exp_dir


def run_setup_and_main_pipline(
        workspace_name: str,
        task_id: str,
        prep_task: str,
        prep_method: str,
        ds_method: str,
        llm: str,
        code_llm: str,
        is_local_task: bool,
        is_tabular: bool,
        total_time: float,
        max_time_per_submission: float,
        use_ci_handling: bool,
        blend_after_n: int,
        max_setups: int = MAX_SETUPS,
        alt_raw_data_root: str | None = None,
        setup_default_response_path: str | None = None,
        main_pipeline_default_response_path: str | None = None,
        max_cpu: int = 0,
        use_final_unit_test: bool = True,
        allow_default_response: bool = False,
        debug_mode: bool = False,
        terminate_after_training: bool = False,
        run_setup_only: bool = False,
        attempt: int | None = None,
        attempt_spec: str = ""
) -> dict[str, ...]:
    """
    Executes the setup and main pipeline for a given task.
    Args:
        workspace_name: Workspace directory name or path.
        task_id: Unique identifier for the competition or task.
        prep_task: The data preparation task name.
        prep_method: The data preparation method name.
        ds_method: The data science method name.
        llm: Config identifier of the language model.
        code_llm: Config identifier of the language model used for code generation.
        is_local_task: Whether the task is local rather than from kaggle.
        is_tabular: Indicates if the task is fully tabular.
        total_time: Maximum total time allowed for all setup attempts (and main pipeline execution time) (in seconds).
        max_time_per_submission: Time limit for each submission created during the main pipeline
        use_ci_handling: whether to run with class imbalance handling
        blend_after_n: number of submissions after which blending is triggered if it was not yet chosen
        max_setups: Maximum number of setup attempts allowed.
        alt_raw_data_root: Alternative data root path for raw task input.
        setup_default_response_path: Path to a file containing pre-registered LLM responses for setup.
        main_pipeline_default_response_path: Path to a file containing pre-registered LLM responses for main pipeline.
        max_cpu: Maximum CPU cores to use.
        use_final_unit_test: Whether to enable final unit tests in setup.
        allow_default_response: Allow default responses.
        debug_mode: Whether to run in debug mode.
        terminate_after_training: Whether to terminate after training phase.
        run_setup_only: If True, only runs setup pipeline.
        attempt : Attempt number.
        attempt_spec: Extra attempt specification.

    Returns:
        Dictionary with keys:
            - "setup" (bool): Indicates if the setup pipeline succeeded.
            - "main" (bool, optional): Indicates if the main pipeline succeeded (only present if setup passed).

    """
    exp_dir = get_attempt_path(
        workspace_name=workspace_name, task_id=task_id, attempt=attempt, attempt_spec=attempt_spec
    )
    setup_version = 0
    setup_log_dir = exp_dir / "logs" / "setup"

    pipeline_status = dict()

    # Run setup pipeline
    is_setup_successful, setup_time_taken, setup_dir = run_setup_pipeline(
        prep_task=prep_task,
        prep_method=prep_method,
        llm=llm,
        code_llm=code_llm,
        task_id=task_id,
        is_local_task=is_local_task,
        alt_raw_data_root=alt_raw_data_root,
        total_time=total_time,
        max_setups=max_setups,
        is_tabular=is_tabular,
        exp_dir=exp_dir,
        setup_log_dir=setup_log_dir,
        use_final_unit_test=use_final_unit_test,
        default_response_path=setup_default_response_path
    )
    if run_setup_only:
        return {'setup': is_setup_successful}

    # Run main pipline if setup is successful
    if is_setup_successful:
        time_for_main_pipeline = total_time - setup_time_taken
        main_pipeline_start_time = time.time()

        main_log_dir = exp_dir / "logs" / "main"
        if is_tabular:
            max_cpu = max_cpu
            if max_cpu <= 0:
                max_cpu = os.cpu_count()

            if debug_mode:
                time_for_main_pipeline = 180  # 3 minutes
            main_pipeline_status = run_ds_tabular_ramp(
                llm=llm,
                task_id=task_id,
                setup_dir=setup_dir,
                setup_version=setup_version,
                time_for_main_pipeline=time_for_main_pipeline,
                max_cpu=max_cpu
            )

        else:
            main_pipeline_status = run_ds_main_pipeline(
                llm=llm,
                ds_method=ds_method,
                task_id=task_id,
                setup_dir=setup_dir,
                main_log_dir=main_log_dir,
                working_dir=exp_dir,
                time_for_main_pipeline=time_for_main_pipeline,
                use_ci_handling=use_ci_handling,
                alt_raw_data_root=alt_raw_data_root,
                blend_after_n=blend_after_n,
                max_time_per_submission=max_time_per_submission,
                default_response_path=main_pipeline_default_response_path,
                allow_default_response=allow_default_response,
                debug_mode=debug_mode,
                is_local_task=is_local_task,
                terminate_after_training=terminate_after_training
            )

        main_pipeline_time_taken = time.time() - main_pipeline_start_time
        time_info = f"Time taken for setup: {setup_time_taken} \n Time taken for main pipeline: {main_pipeline_time_taken}"
        time_info_file = Path(exp_dir, "full_pipeline_time_info.txt")
        with open(time_info_file, 'w') as f:
            f.write(time_info)

        pipeline_status['setup'] = True
        if main_pipeline_status:
            pipeline_status['main'] = True
        else:
            pipeline_status['main'] = False
        print("🎯 Full pipeline run completed", flush=True)

    else:
        pipeline_status['setup'] = False
        print("Failed to setup successfully", flush=True)

    return pipeline_status


def main(
        workspace_name: str,
        task_id: str,
        llm: str,
        code_llm: str,
        is_local_task: bool,
        is_tabular: bool,
        total_time: float,
        max_time_per_submission: float,
        use_ci_handling: bool,
        blend_after_n: int,
        max_setups: int = MAX_SETUPS,
        alt_raw_data_root: str | None = None,
        max_cpu: int = 0,
        terminate_after_training: bool = False,
        run_setup_only: bool = False,
        attempt: int | None = None,
        attempt_spec: str = ""
) -> None:
    """
    Runs the setup and main pipeline stages for a task.

    Workflow:
    - Validates input arguments.
    - Reads relevant environment variables for pipeline configuration.
    - Executes the setup pipeline (and optionally the main pipeline).
    - Prints success or failure messages based on pipeline run status.

    Args:
        workspace_name: Workspace directory name or path.
        task_id: Unique identifier for the competition or task.
        llm: Config identifier of the language model.
        code_llm: Config identifier of the language model used for coding.
        is_local_task: Whether the task is local rather than from kaggle.
        is_tabular: Indicates if the task is fully tabular.
        total_time: Maximum total time allowed for all setup attempts (and main pipeline execution time) (in seconds).
        max_time_per_submission: Time limit for each submission created during the main pipeline
        use_ci_handling: whether to run with class imbalance handling
        blend_after_n: number of submissions after which blending is triggered if it was not yet chosen
        max_setups: Maximum number of setup attempts allowed.
        alt_raw_data_root: Alternative data root path for raw task input.
        max_cpu: Maximum CPU cores to use.
        terminate_after_training: Whether to terminate after training phase.
        run_setup_only: If True, only runs setup pipeline.
        attempt : Attempt number.
        attempt_spec: Extra attempt specification.

    Returns:
        None
    """
    debug_mode = os.environ.get("AGENT_DEBUG", False)
    allow_default_response = os.environ.get("ALLOW_DEFAULT_RESPONSE", False)

    run_status = run_setup_and_main_pipline(
        workspace_name=workspace_name,
        task_id=task_id,
        prep_task="data_preprocessing",
        prep_method="data-prep-flow",
        ds_method="agent-k-solve",
        llm=llm,
        code_llm=code_llm,
        is_local_task=is_local_task,
        is_tabular=is_tabular,
        total_time=total_time,
        max_time_per_submission=max_time_per_submission,
        use_ci_handling=use_ci_handling,
        blend_after_n=blend_after_n,
        max_setups=max_setups,
        setup_default_response_path=None,
        main_pipeline_default_response_path=None,
        max_cpu=max_cpu,
        terminate_after_training=terminate_after_training,
        run_setup_only=run_setup_only,
        attempt=attempt,
        attempt_spec=attempt_spec,
        alt_raw_data_root=alt_raw_data_root,
        allow_default_response=allow_default_response,
        debug_mode=debug_mode,
    )
    if run_status['setup']:
        if not run_setup_only:
            print(f"✅ Setup pipeline successful.")
            if 'main' in run_status.keys() and run_status['main']:
                print(f"✅ Main pipeline successful.")
            else:
                print(f"❌  Main pipeline failed")
    else:
        print(f"❌ Failed to setup successfully.", flush=True)

def add_shared_args(parser: argparse.ArgumentParser) -> None:
    # Core task config
    parser.add_argument("--task_id", type=str, required=True, help="Competition or task ID.")
    parser.add_argument("--llm", type=str, required=True, help="LLM for task execution.")
    parser.add_argument("--code_llm", type=str, required=True, help="LLM for code generation.")

    # Runtime control
    parser.add_argument("--total_time", type=int, required=True, help="Total run time in seconds.")
    parser.add_argument("--max_cpu", type=int, default=0, help="Max CPU usage.")
    parser.add_argument("--max_setups", type=int, default=MAX_SETUPS, help="Max number of setups retrials.")


    # Optional flags
    parser.add_argument("--workspace_name", type=str, default="workspace", help="Workspace folder or path.")
    parser.add_argument("--tabular_task", action='store_true', help="Use tabular pipeline variant.")
    parser.add_argument("--is_local_task", action='store_true', help="Use local data instead of Kaggle.")
    parser.add_argument("--attempt", type=int, required=False, default=None, help="Attempt number.")
    parser.add_argument("--attempt_spec", type=str, default="", help="Extra attempt identifier.")

    parser.add_argument("--max_time_per_submission", type=int, required=True, help="Maximum time per submission")
    parser.add_argument("--alt_raw_data_root", type=str, required=False, help="Alternate data root directory")
    parser.add_argument("--blend_after_n", type=int, required=False, help="Blend after n runs")
    parser.add_argument("--use_ci_handling", action='store_true', default=False, )


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for running the setup and main pipeline.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run setup and main pipeline with timeout.")
    add_shared_args(parser=parser)
    parser.add_argument("--run_setup_only", action='store_true', help="Only run the setup pipeline.")
    parser.add_argument("--terminate_after_training", action='store_true', help="Stop after training.")

    return parser.parse_args()


def validate_args(args) -> None:
    """
    Validates parsed command-line arguments for the pipeline runner.
    Args:
        args:Parsed arguments from argparse.

    Returns:
        None
    """
    if args.total_time < 0:
        print("❌ Error: <total_time> must be a non-negative integer.", flush=True)
        sys.exit(1)

    if args.max_cpu > os.cpu_count():
        print("❌ Error: <max_cpu> must be less than or equal to max CPUs available.", flush=True)
        sys.exit(1)
