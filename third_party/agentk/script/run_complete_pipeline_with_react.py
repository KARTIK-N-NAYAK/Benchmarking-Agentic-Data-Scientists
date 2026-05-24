import argparse
import glob
import os

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True, cwd=True)
#from agent.utils.utils import run_command
import subprocess
from agent.run_pipelines import main, get_attempt_path, add_shared_args

def run_command(command: str):
    """Run a shell command with bash and raise if it fails."""
    print(f"[run_command] {command}")
    return subprocess.run(command, shell=True, check=True, executable="/bin/bash")

def main_pipeline(args):
    if not args.code_llm:
        code_llm = args.llm
    else:
        code_llm = args.code_llm

    main(
        workspace_name=args.workspace_name,
        task_id=args.task_id,
        llm=args.llm,
        code_llm=code_llm,
        is_tabular=args.tabular_task,
        is_local_task=args.is_local_task,
        total_time=args.total_time,
        max_time_per_submission=args.max_time_per_submission,
        use_ci_handling=args.use_ci_handling,
        blend_after_n=args.blend_after_n,
        max_setups=args.max_setups,
        alt_raw_data_root=args.alt_raw_data_root,
        max_cpu=args.max_cpu,
        attempt=args.attempt,
        attempt_spec=args.attempt_spec
    )

    exp_dir = get_attempt_path(
        workspace_name=args.workspace_name, task_id=args.task_id, attempt=args.attempt, attempt_spec=args.attempt_spec
    )
    summary_paths = glob.glob(os.path.join(exp_dir, "seed_*", "summary.txt"))
    if len(summary_paths) == 0:
        use_agent_k_warm_start = "false"
        agent_k_scaffold_submission=""
    else:
        use_agent_k_warm_start = "true"
        agent_k_scaffold_submission=f"agent.agent_k_submissions=\"{summary_paths[0]}\""

    post_scaffold_workspace = exp_dir / "post_scaffold"

    post_scaffold_command = (
        f"source ~/.bashrc && "
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate reactagent && "
        f"aide "
        f"data_dir={args.alt_raw_data_root}/{args.task_id} "
        f"exp_name=\"{args.task_id}\" "
        f"top_n={args.post_scaffold_top_n} "
        f"agent.time_limit={args.total_time} "
        f"exec.timeout={args.post_scaffold_timeout} "
        f"copy_data=false "
        f"workspace_dir=\"{post_scaffold_workspace}\" "
        f"agent.code.model=\"{args.post_scaffold_llm}\" "
        f"agent.feedback.model=\"{args.post_scaffold_llm}\" "
        f"agent.use_agent_k_warm_start={use_agent_k_warm_start} "
        f"{agent_k_scaffold_submission} "

    )

    run_command(post_scaffold_command)

    if args.is_local_task:
        return

    submission_command = (
        f"python {root}/third_party/data_science/kaggle_submission/submit_kaggle.py "
        f"--task_id {args.task_id} "
        f"--workspace_root_agent {exp_dir.parent}/ "
        f"--workspace_root_react {post_scaffold_workspace} "
        f"--message {args.submission_message} "
        f"--leaderboards_dir {args.leaderboards_dir} "
        f"--team_name {args.team_name} "
    )

    run_command(submission_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run setup and main pipeline with timeout.")
    add_shared_args(parser=parser)

    parser.add_argument("--post_scaffold_top_n", type=int, default=1)
    parser.add_argument("--post_scaffold_timeout", type=int, required=True)
    parser.add_argument("--post_scaffold_llm", required=True)

    parser.add_argument("--submission_message", default="Submissionmade", type=str)
    parser.add_argument("--leaderboards_dir", default=None)
    parser.add_argument("--team_name", default="Abhineet", type=str)

    args_ = parser.parse_args()
    main_pipeline(args=args_)
