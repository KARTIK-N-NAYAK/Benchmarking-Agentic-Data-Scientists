import os
import re
import shutil
import subprocess as sp
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from ds_agent.data_preprocessing.benchmarking.utils import get_task_scripted_answer_path
from ds_agent.data_preprocessing.log_utils import extract_task_error, load_log_as_df
from ds_agent.competition_ids import CompetitionID
from agent.tasks.datascience_task.utils import FileMap
from ds_agent.competition_struct import DataType
from ds_agent.competition_instances import ALL_COMPETITIONS_DICT


def run_ds_pipeline(
        task_dict: dict[CompetitionID | str: list[str]],
        output_dir: Path,
        experiment_name: str,
        python_path: str,
        prepared_setup_dir: Path,
        input_modalities: set[DataType],
        output_modalities: set[DataType],
        workspace_path: Path | None = None,
):
    successes = {}
    for task_id in task_dict:
        if isinstance(task_id, CompetitionID):
            task_id = task_id.value
            input_modalities = ALL_COMPETITIONS_DICT[task_id].input_types
            output_modalities = ALL_COMPETITIONS_DICT[task_id].target_types

        scripted_answer_path = get_task_scripted_answer_path(
            input_modalities=input_modalities, output_modalities=output_modalities
        )
        output_experiment_dir = Path(output_dir) / experiment_name / task_id

        successes[task_id] = {}
        print(f'Running ds main pipeline on task {task_id}', flush=True)
        for seed in tqdm(task_dict[task_id]):

            log_dir = output_experiment_dir / f'seed_{seed}'
            output_log = log_dir / 'output.txt'
            error_log = log_dir / 'error.txt'
            success_log = log_dir / 'success.txt'
            hydra_dir = log_dir / 'logs_ds_main'
            working_dir = output_experiment_dir / f'seed_{seed}' / 'final_unit_test'
            if output_log.is_file():
                print(f'\nTask {task_id} - seed {seed} already processed skipping', flush=True)
                continue

            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(str(log_dir.parent), 0o777)
                os.chmod(str(log_dir), 0o777)
            now = datetime.now()
            current_time = f"{now:%Y-%m-%d}_{now:%H-%M-%S}"
            if workspace_path:
                workspace_arg =  f"task.workspace_path={workspace_path}/{current_time} "
            else:
                workspace_arg = ""

            cmd = (
                f"TOKENIZERS_PARALLELISM=0 ALLOW_DEFAULT_RESPONSE=1 "
                f"HYDRA_FULL_ERROR=1 AGENT_DEBUG=1 "
                f"MAX_TIME_PER_SUBMISSION=3600 "
                f"{python_path} src/agent/start.py "
                f"task=data_science_interact "
                f"llm@agent.llm=hf_hub/qwen2.5-72b "  # LLM name doest matter as we read responses from file
                f"method=agent-k-solve "
                f"hydra.run.dir={hydra_dir} "
                f"max_episodes=5 "
                f"task.task_id={task_id} "
                f"task.prepared_setup_dir={prepared_setup_dir} "
                f"task.prepared_version=seed_{seed} "
                f"task.terminate_after_training=true "
                f"{workspace_arg}"
                f"+task.max_exec_time={3600 * 24} "
                f"+agent.read_answer_from_file_path={scripted_answer_path}"
            )

            with open(log_dir / "command.txt", "w") as f:
                f.write(cmd)

            r = sp.Popen([cmd], stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
            out, err = r.communicate()
            out, err = out.decode(), err.decode()

            success, submission_file_path = check_ds_pipeline_success(out)

            if submission_file_path:
                solve_py_error_src_log_path = submission_file_path.parent / FileMap.SOLVE_ERROR_LOG.value
                solve_py_error_dst_log_path = log_dir / FileMap.SOLVE_ERROR_LOG.value

                solve_py_output_src_log_path = submission_file_path.parent / FileMap.SOLVE_OUTPUT_LOG.value
                solve_py_output_dst_log_path = log_dir / FileMap.SOLVE_OUTPUT_LOG.value

                if solve_py_error_src_log_path.exists():
                    shutil.copy(str(solve_py_error_src_log_path), str(solve_py_error_dst_log_path))
                else:
                    print("No run_error.log found at path: ", solve_py_error_src_log_path)

                if solve_py_output_src_log_path.exists():
                    shutil.copy(str(solve_py_output_src_log_path), str(solve_py_output_dst_log_path))
                else:
                    print("No run_output.log found at path: ", solve_py_error_src_log_path)

                submission_path_log_path = log_dir / "workspace_path.txt"
                with open(submission_path_log_path, 'w') as f:
                    f.write(str(submission_file_path))

            else:
                print(f"\n No workspace  dir identify in the run TASK {task_id} - seed {seed}", flush=True)

            if success:
                with open(success_log, 'w') as f:
                    f.write(f"Success: {submission_file_path}")

            with open(output_log, 'w') as f:
                f.write(out)

            with open(error_log, 'w') as f:
                f.write(err)
                if not success:
                    f.write('Error\n')
                    try:
                        e = extract_task_error(load_log_as_df(hydra_dir / "output.jsonl"), use_final_test=False)
                        if e is not None:
                            f.write(e)
                        # else:
                        #     f.write(f"Could not extract error from {hydra_dir / 'output.jsonl'}\n")
                    except:
                        f.write("Error extraction also errored\n")

            if not success:
                print(f"\n TASK {task_id} - seed {seed} - failed", flush=True)

            successes[task_id][seed] = success
    return successes


def check_ds_pipeline_success(
        output_log: str,
) -> tuple[bool, Path | None]:
    regex_pattern = r"workspace: (.*?) \|"
    match = re.search(regex_pattern, output_log)

    if match:
        extracted_path = match.group(1)
        path_workspace = Path(extracted_path) / "submissions"
        path_submission = [f for f in path_workspace.iterdir() if f.is_dir()][0]
        submission_file_path = path_submission / 'submission.csv'

        return submission_file_path.is_file(), submission_file_path
    else:
        return False, None
