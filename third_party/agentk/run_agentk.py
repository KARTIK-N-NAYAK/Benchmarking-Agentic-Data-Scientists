#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Logging
def log(msg: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

log("Starting AgentK pipeline runner")

# Setting up the path configuration
AGENTK_ROOT = Path(
    "/work/dlclarge1/mitran-AgentK/automated_data_scientists_benchmark/third_party/agentk"
)

# Setting up the raw_local_tasks location
ALT_RAW_DATA_ROOT = Path(
    os.environ.get(
        "ALT_RAW_DATA_ROOT",
        "/work/dlclarge1/mitran-AgentK/automated_data_scientists_benchmark/third_party/agentk/data/raw_local_tasks",
    )
)

CPU_RANGE = os.environ.get("CPU_RANGE", "0-7")

# Task ID
TASK_ID = os.environ.get("TASK_ID", "rrp")
MODEL_ID = os.environ.get("MODEL_ID", "openai/qwen2.5-72b-instruct")

TIME_LIMIT_SECONDS = int(os.environ.get("TIME_LIMIT_SECONDS", "3600"))
MAX_TIME_PER_SUBMISSION = int(os.environ.get("MAX_TIME_PER_SUBMISSION", "3600"))
ATTEMPT = int(os.environ.get("ATTEMPT", "0"))

# Workspace name
WORKSPACE_NAME = os.environ.get(
    "WORKSPACE_NAME",
    "/work/dlclarge1/mitran-AgentK/workspaces/rrp_openai_qwen",
)

MAX_SETUPS = int(os.environ.get("MAX_SETUPS", "1"))

#Sanity checks
if not AGENTK_ROOT.exists():
    log(f"ERROR: AGENTK_ROOT does not exist: {AGENTK_ROOT}")
    sys.exit(1)

if not ALT_RAW_DATA_ROOT.exists():
    log(f"ERROR: ALT_RAW_DATA_ROOT does not exist: {ALT_RAW_DATA_ROOT}")
    sys.exit(1)

log(f"AGENTK_ROOT           = {AGENTK_ROOT}")
log(f"TASK_ID               = {TASK_ID}")
log(f"MODEL_ID              = {MODEL_ID}")
log(f"ALT_RAW_DATA_ROOT     = {ALT_RAW_DATA_ROOT}")
log(f"WORKSPACE_NAME        = {WORKSPACE_NAME}")
log(f"CPU_RANGE             = {CPU_RANGE}")
log(f"TIME_LIMIT_SECONDS    = {TIME_LIMIT_SECONDS}")
log(f"MAX_TIME_PER_SUB      = {MAX_TIME_PER_SUBMISSION}")
log(f"ATTEMPT               = {ATTEMPT}")
log(f"MAX_SETUPS            = {MAX_SETUPS}")

#chdir and uv sync
os.chdir(AGENTK_ROOT)
log(f"Changed directory to {Path.cwd()}")

log("Running: uv sync")
subprocess.run(
    ["uv", "sync"],
    check=True,
)

# Setup python path
pythonpath = (
    f"{AGENTK_ROOT}/src:"
    f"{AGENTK_ROOT}/third_party/ds-agent:"
    + os.environ.get("PYTHONPATH", "")
)

env = os.environ.copy()
env["PYTHONPATH"] = pythonpath

# Build command
cmd = [
    "taskset",
    "-c",
    CPU_RANGE,
    sys.executable,
    "-u",
    "run_complete_pipeline.py",
    "--task_id",
    TASK_ID,
    "--is_local_task",
    "--tabular_task",
    "--alt_raw_data_root",
    str(ALT_RAW_DATA_ROOT),
    "--llm",
    MODEL_ID,
    "--code_llm",
    MODEL_ID,
    "--total_time",
    str(TIME_LIMIT_SECONDS),
    "--max_time_per_submission",
    str(MAX_TIME_PER_SUBMISSION),
    "--attempt",
    str(ATTEMPT),
    "--workspace_name",
    WORKSPACE_NAME,
    "--max_setups",
    str(MAX_SETUPS),
]

log("Executing AgentK pipeline:")
log(" ".join(cmd))

# Run + Logs capture 
log_file = Path(f"{TASK_ID}.txt")
log(f"Logging stdout/stderr to: {log_file}")

with log_file.open("w") as f:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")
        f.write(line)

    process.wait()

exit_code = process.returncode
log(f"AgentK finished with exit code {exit_code}")
sys.exit(exit_code)
