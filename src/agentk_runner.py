#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# =========================
# CPU limits
# =========================
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"

# =========================
# Cache locations
# =========================
os.environ["HF_HOME"] = "/work/dlclarge1/mitran-AgentK/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]

# =========================
# API CONFIG (UPDATED)
# =========================
os.environ["OPENAI_BASE_URL"] = "https://openwebui.uni-freiburg.de/api/v1"
os.environ["OPENAI_API_KEY"] = "sk-494ce4dab1034d1e8965f07c445be4c8"   # <-- replace if needed

# =========================
# GPU stability
# =========================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================
# FORCE VENV
# =========================
os.environ.pop("UV_PROJECT_ENVIRONMENT", None)
os.environ.pop("UV_PROJECT", None)

VENV_PATH = "/home/mitran/miniconda3/envs/agent"

os.environ["VIRTUAL_ENV"] = VENV_PATH
os.environ["PATH"] = f"{VENV_PATH}/bin:" + os.environ.get("PATH", "")

print(f"[DEBUG] Using VENV: {VENV_PATH}")
print(f"[DEBUG] Python executable: {VENV_PATH}/bin/python")

# =========================
# LOGGING
# =========================
def log(msg: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# =========================
# GLOBAL PROCESS HANDLE
# =========================
pipeline_process = None
current_task_id = None

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/work/dlclarge1/mitran-AgentK/automated_data_scientists_benchmark")
AGENTK_ROOT = PROJECT_ROOT / "third_party" / "agentk"

MODEL_ID = "openai/gpt-oss-120b-llmlb"

ALT_RAW_DATA_ROOT_DEFAULT = AGENTK_ROOT / "data/raw_local_tasks"

BASE_WORKSPACE_DIR = Path(
    os.environ.get(
        "BASE_WORKSPACE_DIR",
        "/work/dlclarge1/mitran-AgentK/workspaces",
    )
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# =========================
# TIMEOUT HANDLER
# =========================
def timeout_handler(signum, frame):
    log("ERROR: Global timeout reached. Terminating pipeline.")

    try:
        if pipeline_process:
            pipeline_process.terminate()
    except:
        pass

    sys.exit(1)

# =========================
# SETUP
# =========================
def setup_env():
    log(f"OPENAI_BASE_URL = {os.environ['OPENAI_BASE_URL']}")

def unzip_if_needed():
    ramp_hyperopt = AGENTK_ROOT / "third_party/ramp-hyperopt"
    ramp_workflow = AGENTK_ROOT / "third_party/ramp-workflow"

    if not ramp_hyperopt.exists():
        log("Unzipping ramp-hyperopt...")
        subprocess.run(
            ["unzip", str(AGENTK_ROOT / "third_party/ramp-hyperopt.zip"),
             "-d", str(AGENTK_ROOT / "third_party")],
            check=True,
        )

    if not ramp_workflow.exists():
        log("Unzipping ramp-workflow...")
        subprocess.run(
            ["unzip", str(AGENTK_ROOT / "third_party/ramp-workflow.zip"),
             "-d", str(AGENTK_ROOT / "third_party")],
            check=True,
        )

# =========================
# PIPELINE
# =========================
def run_pipeline(task_id, max_time_per_submission, total_time):
    global pipeline_process

    log("Starting AgentK pipeline...")

    alt_raw_data_root = Path(
        os.environ.get("ALT_RAW_DATA_ROOT", str(ALT_RAW_DATA_ROOT_DEFAULT))
    )

    workspace_name = str(BASE_WORKSPACE_DIR / f"{task_id}_gptoss120b")
    
    # Create leaderboard directory (per task)
    leaderboard_dir = BASE_WORKSPACE_DIR / "leaderboards" / task_id
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    pythonpath_parts = [
        str(AGENTK_ROOT / "src"),
        str(AGENTK_ROOT / "third_party" / "ds-agent" / "src"),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    cmd = [
    f"{VENV_PATH}/bin/python",
    "script/run_complete_pipeline_with_react.py",
    "--task_id", task_id,
    "--is_local_task",
    "--tabular_task",
    "--alt_raw_data_root", str(alt_raw_data_root),
    "--llm", MODEL_ID,
    "--code_llm", MODEL_ID,
    "--total_time", str(total_time),
    "--max_time_per_submission", str(max_time_per_submission),
    "--attempt", "0",
    "--workspace_name", workspace_name,

    "--blend_after_n", "2",
    "--post_scaffold_top_n", "2",
    "--post_scaffold_timeout", "600",
    "--post_scaffold_llm", MODEL_ID,
    "--leaderboards_dir", str(leaderboard_dir),
    ]

    log("Executing:")
    log(" ".join(cmd))

    log_file = LOG_DIR / f"{task_id}.txt"

    with log_file.open("w") as f:
        pipeline_process = subprocess.Popen(
            cmd,
            cwd=AGENTK_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        for line in pipeline_process.stdout:
            print(line, end="")
            f.write(line)

        pipeline_process.wait()

    return pipeline_process.returncode

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="fctp")
    parser.add_argument("--max_time_per_submission", type=int, default=600)
    parser.add_argument("--total_time", type=int, default=18000)
    args = parser.parse_args()

    task_id = args.task_id

    global current_task_id
    current_task_id = task_id

    GLOBAL_TIMEOUT = int(os.environ.get("GLOBAL_TIMEOUT", "14400"))
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(GLOBAL_TIMEOUT)

    if not AGENTK_ROOT.exists():
        log(f"ERROR: AGENTK_ROOT missing: {AGENTK_ROOT}")
        sys.exit(1)

    setup_env()
    unzip_if_needed()

    log("Running without vLLM (direct API mode)")

    exit_code = run_pipeline(task_id, args.max_time_per_submission, args.total_time)

    log(f"Pipeline finished with exit code {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
