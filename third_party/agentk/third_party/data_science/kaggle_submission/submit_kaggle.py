import argparse
import fnmatch
import glob
import subprocess
from datetime import datetime
from pathlib import Path

from colorama import Fore, init

# Initialize colorama
init(autoreset=True)


# Function to run the grading command
def run_grading_command(competition: str, submission_file_path: str, message: str, result_file_name: str,
                        path_to_leaderboard_dir: str, team_name: str):
    command = [
        "python",
        "third_party/data_science/kaggle_submission/grade_submissions.py",
        "--competition", competition,
        "--submissions_file", submission_file_path,
        "--message", message,
        "--leaderboard_dir", path_to_leaderboard_dir,
        "--submit_and_validate", "True",
        "--result_file_name", result_file_name,
        "--team_name", team_name,
    ]

    try:
        print(f"Running command for {submission_file_path}: \n```\n{' '.join(command)}\n```")
        subprocess.run(command, check=True)
        return True
    except Exception as e:
        print(f"Failed to run command for {submission_file_path}: {e}")
        return False


def save_report(success_list: list, failure_list: list, already_submitted_list: list):
    # Create a timestamped report filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"submission_report_{timestamp}.txt"

    with open(report_filename, "w") as report_file:
        # Write the success and failure details to the report
        report_file.write(f"Submission Report - {timestamp}\n\n")

        # Write successful submissions
        if success_list:
            report_file.write(f"Successful Submissions:\n")
            for submission in success_list:
                report_file.write(f"- {submission}\n")
        else:
            report_file.write(f"No successful submissions.\n")

        # Write failed submissions
        if failure_list:
            report_file.write(f"\nFailed Submissions:\n")
            for submission in failure_list:
                report_file.write(f"- {submission}\n")
        else:
            report_file.write(f"\nNo failed submissions.\n")

        # Already submitted
        if already_submitted_list:
            report_file.write(f"\nAlready Submitted:\n")
            for submission in already_submitted_list:
                report_file.write(f"- {submission}\n")
        else:
            report_file.write(f"\nNo failed submissions.\n")

    print(f"\n{Fore.YELLOW}Submission report saved to {report_filename}")


def get_submission_dirs(root: str | None, task_id: str | None, pattern: str) -> list[str]:
    """Return submission directories matching a pattern for a given root and optional task_id."""
    if root is None:
        return []
    task = task_id if task_id is not None else "*"
    search_pattern = f"{root}/{task}/{pattern}"
    return glob.glob(search_pattern)


def main(args: argparse.Namespace):
    failure_list = []
    success_list = []
    already_submitted_list = []

    task_id = args.task_id if args.task_id is not None else "*"

    main_pipeline_dirs = []
    tabular_main_dirs = []
    aide_dirs = []
    if args.workspace_root_agent is not None:
        main_pipeline_dirs = get_submission_dirs(
            args.workspace_root_agent, task_id, "seed_*/main_pipeline/submissions/"
        )
        tabular_main_dirs = get_submission_dirs(
            args.workspace_root_agent, task_id, "seed_*/ramp_kit_*/final_test_predictions/"
        )
    if args.workspace_root_react is not None:
        aide_dirs = get_submission_dirs(args.workspace_root_react, task_id, "*/best_submissions/")

    # Combine them into one list of (dir, mode)
    all_submission_dirs = [(subdir, "main_pipeline") for subdir in main_pipeline_dirs] + \
                          [(subdir, "tabular_main") for subdir in tabular_main_dirs] + \
                          [(subdir, "aide") for subdir in aide_dirs]

    for subdir, mode in all_submission_dirs:
        subdir_path = Path(subdir)
        attempt = subdir_path.parent.parent.parent.parent.name
        competition = subdir_path.parent.parent.parent.name

        if mode == "main_pipeline":
            files_to_submit = [
                fpath for fpath in glob.glob(subdir + "*/*submission*.csv")
                if "sample" not in fpath
            ]
        elif mode == "tabular_main":
            files_to_submit = [
                fpath for fpath in glob.glob(subdir + "/*.csv")
                if fnmatch.fnmatch(fpath, "*blended_*.csv") or fnmatch.fnmatch(fpath, "*blend_*.csv")
                if fpath.endswith("30.csv") or fpath.endswith("030.csv")
            ]
        elif mode == "aide":
            files_to_submit = [
                fpath for fpath in glob.glob(subdir + "*.csv")
            ]
        else:
            continue  # skip unknown type

        for fpath in files_to_submit:
            fpath_path = Path(fpath)
            fname = fpath_path.name
            fname_score = fname.replace(".csv", "_scores.json")
            score_path = fpath_path.parent / fname_score

            if score_path.exists():
                print(f'\n{Fore.BLUE}Already submitted {fpath}')
                already_submitted_list.append(fpath)
            else:
                # Use different message suffix depending on mode
                suffix = attempt if mode == "main_pipeline" else fpath_path.parent.name
                submit_message = f"{args.message} - {suffix}"

                success = run_grading_command(
                    competition=competition,
                    submission_file_path=fpath,
                    message=submit_message,
                    result_file_name=fname_score,
                    path_to_leaderboard_dir=args.leaderboards_dir,
                    team_name=args.team_name,
                )

                if success:
                    success_list.append(fpath)
                else:
                    failure_list.append(fpath)

    if failure_list:
        print("\nThe following submissions failed:")
        for failed_submission in failure_list:
            print(f"\n{Fore.RED}- {failed_submission}")
    else:
        print(f"\n{Fore.GREEN}All submissions were graded successfully!")

    save_report(success_list=success_list, failure_list=failure_list, already_submitted_list=already_submitted_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_root_agent", default="", help="Workspace root directory")
    parser.add_argument("--workspace_root_react", default="", help="Workspace root directory for react agent")
    parser.add_argument("--task_id", required=False, help="If you only want to submit a particular competition")
    parser.add_argument("--message", required=True, help="Submission message")
    parser.add_argument("--team_name", required=True, help="Team name")
    parser.add_argument("--leaderboards_dir", required=True, help="Leaderboard directory")
    args = parser.parse_args()
    main(args)
