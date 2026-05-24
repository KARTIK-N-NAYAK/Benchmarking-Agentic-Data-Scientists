import json
from pathlib import Path
from typing import Collection

import pandas as pd
from tqdm import tqdm

from agent.loggers.base import Tag
from agent.memory import MemKey

LOG_ID_COLS = [t.value for t in Tag.universal_tags().keys()]

LOG_KEY_COL = "key"
LOG_VALUE_COL = "value"


def extract_task_id(log_df) -> str:
    task_id_df = log_df[log_df[LOG_KEY_COL] == "memory:store:task_id"]
    if not task_id_df.empty:
        return task_id_df.iloc[0][LOG_VALUE_COL]
    else:
        return ""


def extract_task_success(log_df, use_final_test=False) -> bool:
    if use_final_test:
        final_test_df = log_df[log_df[LOG_KEY_COL] == f"memory:store:{MemKey.FINAL_TEST_PASSED}"]
        if final_test_df.empty:
            return False
        return bool(final_test_df[LOG_VALUE_COL].iloc[-1])
    else:
        # Check success with no crash
        error_df = log_df[log_df[LOG_KEY_COL] == "error"]
        return error_df.empty


def extract_task_error(log_df, use_final_test=False) -> str | None:
    if use_final_test:
        # First check for final_test_error
        final_test_df = log_df[log_df[LOG_KEY_COL] == f"memory:store:{MemKey.FINAL_TEST_ERROR}"]
        if not final_test_df.empty:
            return final_test_df[LOG_VALUE_COL].iloc[-1]
    # Check for normal errors
    error_df = log_df[log_df[LOG_KEY_COL] == "error"]
    return None if error_df.empty else error_df[LOG_VALUE_COL].iloc[-1]


def load_log_as_df(log_path: Path):
    with log_path.open("r") as fp:
        processed_rows = []
        for line in fp.readlines():
            try:
                row_dict = json.loads(line.strip())
            except json.decoder.JSONDecodeError as e:
                print(f"Decoding error while loading row in log ({log_path}): {e}")
                print("Line:", line.strip())
                continue
            processed_row = {}
            for key in LOG_ID_COLS:
                processed_row[key] = row_dict.pop(key)

            key, value = next(iter(row_dict.items()))
            processed_row[LOG_KEY_COL] = key
            processed_row[LOG_VALUE_COL] = value
            processed_rows.append(processed_row)
        df = pd.DataFrame.from_records(processed_rows)

    return df


def make_experiment_filter(tasks: Collection[str] | None = None, seeds: Collection[int] | None = None):
    def path_filter(p: Path) -> bool:
        task, seed = p.parts[-4:-2]
        task_ok = tasks is None or task in tasks
        seed_ok = seeds is None or seed in seeds
        return task_ok and seed_ok

    return path_filter


def get_experiment_paths(
        benchmark_runs_dirs: Collection[Path], versions: Collection[str] | None = None
) -> dict[str, Path]:
    agent_config_paths = []
    for benchmark_runs_dir in tqdm(benchmark_runs_dirs):
        if not benchmark_runs_dir.exists():
            raise FileNotFoundError(f"The log path you provided does not exist: {benchmark_runs_dir}")

        agent_config_paths.extend(benchmark_runs_dir.glob("*/task_report.csv"))

    log_paths = [p.parent for p in agent_config_paths]

    log_paths = [p for p in log_paths if any(v in p.name.split("v")[-1] for v in versions)]

    if not log_paths:
        raise ValueError(f"No results loaded from any of the config paths: {' '.join(map(str, benchmark_runs_dirs))}")

    return {p.name: p for p in log_paths}


def read_experiment_logs(experiment_path: Path, path_filter=None) -> dict[Path, pd.DataFrame]:
    path_filter = path_filter if path_filter is not None else lambda p: True
    log_paths = list(filter(path_filter, experiment_path.glob("*/seed_*/logs/output.jsonl")))
    log_dfs = {}
    for log_path in tqdm(log_paths, desc="Reading experiment logs"):
        log_dfs[log_path] = load_log_as_df(log_path)
    return log_dfs
