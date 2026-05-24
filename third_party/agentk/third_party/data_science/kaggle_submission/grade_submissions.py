import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import pyrootutils
from ds_agent.competition_ids import CompetitionID
from ds_agent.competition_instances import ALL_COMPETITIONS_LIST, get_competitions_from_ids
from ds_agent.results_processing.performance_results import get_candidate_leaderboard_path, get_quantiles_from_scores
from ds_agent.utils_kaggle import get_medal

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from third_party.data_science.utils import download_leaderboard
from kaggle.api.kaggle_api_extended import KaggleApi

ALL_COMPETITIONS_DICT = {c.competition_id: c for c in ALL_COMPETITIONS_LIST}

SUBMIT_WAIT_TIME = 10  # After submission, it waits for a few seconds to finish submission validation from Kaggle
api = KaggleApi()
api.authenticate()


def update_json(file_path: Path, results: dict):
    # Read existing data
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                result_report = json.load(f)
        except PermissionError:
            print(f"Permission denied while reading {file_path}")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in {file_path}, overwriting")
            result_report = {}
        result_report.update(results)
    else:
        result_report = results

    # Write data
    try:
        with open(file_path, "w") as f:
            json.dump(result_report, f, indent=2)
    except PermissionError:
        raise PermissionError(f"No permission to write {file_path}")


def submit_submission(competition_name: str, submission_file_path: str, message: str) -> tuple[bool, str]:
    status_message = ""
    if os.path.exists(submission_file_path):
        command = [
            'kaggle', 'competitions', 'submit',
            '-c', competition_name,
            '-f', submission_file_path,
            '-m', message
        ]
        print(" ".join(command[:-2]) + f" -m \"{message}\"", flush=True)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Submission result:", result.stdout)
            if "Could not submit to competition" in result.stdout:
                status = False
                status_message = result.stdout
            else:
                status = True
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit for {competition_name}: {e.stderr}")
            status = False
            status_message = e.stderr
    else:
        status = False
        status_message = "Submission file does not exist"

    return status, status_message


def get_competition_score(competition_name: str, submission_index=0) -> tuple[float, float, str, str]:
    public_score = private_score = status = description = None
    submission_status = 0
    while not submission_status:
        submissions = api.competition_submissions(competition_name)

        submission = submissions[submission_index]
        try:
            public_score = safe_float_conversion(submission.publicScore)
            private_score = safe_float_conversion(submission.privateScore)
            status = submission.status
            description = submission.description
            if status == "error":
                status += " : " + submission.errorDescription
                submission_status = 1
            elif status == "complete":
                print(f"Submission ID: {submission}")
                print(f"Submission Date: {submission.date}")
                print(f"Public Score: {submission.publicScore}")
                print(f"Private Score: {submission.privateScore}")
                print(f"Description: {submission.description}")
                print(f"Status: {submission.status}")
                submission_status = 1
            else:
                submission_status = 0
        except AttributeError:
            public_score = safe_float_conversion(submission.public_score)
            private_score = safe_float_conversion(submission.private_score)
            submission_status = submission.status.value
            description = submission.description
            if submission.status.PENDING:
                continue
            elif submission.status.COMPLETED:
                print(f"Submission ID: {submission}")
                print(f"Submission Date: {submission.date}")
                print(f"Public Score: {submission.public_score}")
                print(f"Private Score: {submission.private_score}")
                print(f"Description: {submission.description}")
                print(f"Status: 'completed")
                status = 'completed'
            elif submission.status.ERROR:
                print(f"Submission ID: {submission}")
                print(f"Submission Date: {submission.date}")
                print(f"Public Score: {submission.public_score}")
                print(f"Private Score: {submission.private_score}")
                print(f"Description: {submission.description}")
                print(f"Status: '{submission.error_description}'")
                status = submission.error_description

    return public_score, private_score, description, status


def safe_float_conversion(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def prepare_score_info(public_score: float, private_score: float, public_info: dict, private_info: dict,
                       description: str, status: str) -> dict:
    return {
        'total_submissions': public_info.get('total_submissions', None)
        if public_info.get('total_submissions', None) else private_info.get('total_submissions', None),
        'public_score': public_score,
        'private_score': private_score,
        'public_rank': public_info.get('public_rank', None),
        'private_rank': private_info.get('private_rank', None),
        'public_quantile': public_info.get('public_quantile', None),
        'private_quantile': private_info.get('private_quantile', None),
        'public_medal': public_info.get('public_medal', None),
        'private_medal': private_info.get('private_medal', None),
        'description': description,
        'status': status,
    }


def get_write_permission(dir_path: str | Path) -> None:
    if os.environ.get("SUDO_PASSWORD", None):
        os.system(f'echo {os.environ["SUDO_PASSWORD"]} | sudo -S chmod -R 777 {dir_path}')


def submit_and_rank_all(
        competition_name: str, submissions_file: str, message: str, leaderboard_dir: str,
        submission_score_file_name: str, team_name: str
) -> tuple[dict, dict]:
    submission_file = Path(submissions_file)
    score_info = {}
    failed_submissions = {}
    if submission_file.exists():
        print(f"Submitting {submission_file}")
        submission_status, submission_message = submit_submission(competition_name, str(submission_file), message)
        time.sleep(SUBMIT_WAIT_TIME)
        if submission_status:

            public_score, private_score, description, status = get_competition_score(
                competition_name=competition_name)

            public_info, private_info = rank_submission(
                competition_name, public_score, private_score, leaderboard_dir, team_name
            )

            score_info.update(
                {str(submission_file): prepare_score_info(
                    public_score=public_score, private_score=private_score, public_info=public_info,
                    private_info=private_info, description=description, status=status
                )}
            )
        else:
            raise TimeoutError

    get_write_permission(str(submission_file.parent))
    score_file = os.path.join(submission_file.parent, submission_score_file_name)

    update_json(Path(score_file), score_info)

    formatted_json = json.dumps(score_info, indent=4, sort_keys=True)
    print(formatted_json)

    return score_info, failed_submissions


def rank_all(competition_name: str, submissions_dir: str, leaderboard_dir: str, n_submissions: int,
             submission_score_file_name: str, team_name: str):
    print(f"competition_name : {competition_name}")
    score_info = {}
    for i in range(n_submissions):
        try:
            public_score, private_score, description, status = get_competition_score(
                competition_name=competition_name, submission_index=i
            )

            public_info, private_info = rank_submission(
                competition_name, public_score, private_score, leaderboard_dir, team_name
            )

            score_info.update(
                {i: prepare_score_info(public_score=public_score, private_score=private_score, public_info=public_info,
                                       private_info=private_info, description=description, status=status)}
            )
        except Exception as e:
            print(e)

    get_write_permission(submissions_dir)
    score_file = os.path.join(submissions_dir, submission_score_file_name)

    update_json(Path(score_file), score_info)

    formatted_json = json.dumps(score_info, indent=4, sort_keys=True)
    print(formatted_json)


def rank_one(competition_name: str, submissions_file: str, leaderboard_dir: str, n_submissions: int,
             submission_score_file_name: str, team_name: str):
    print(f"competition_name : {competition_name}")
    score_info = {}
    if os.path.exists(submissions_file):
        try:
            public_score, private_score, description, status = get_competition_score(
                competition_name=competition_name
            )

            public_info, private_info = rank_submission(
                competition_name, public_score, private_score, leaderboard_dir, team_name
            )

            score_info.update(
                {submissions_file: prepare_score_info(
                    public_score=public_score, private_score=private_score, public_info=public_info,
                    private_info=private_info, description=description, status=status
                )}
            )
        except Exception as e:
            print(e)
    else:
        print(f'File {submissions_file} not found.')
    submission_file = Path(submissions_file)
    get_write_permission(str(submission_file.parent))
    score_file = os.path.join(submission_file.parent, submission_score_file_name)
    update_json(Path(score_file), score_info)

    formatted_json = json.dumps(score_info, indent=4, sort_keys=True)
    print(formatted_json)


def handle_benchmark_entry(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    filtered_df = df[(df['Rank'] != 0) & (df['TeamName'] != team_name)]
    return filtered_df


def rank_submission(competition_name: str, public_score: float, private_score: float, leaderboard_dir: str,
                    team_name: str) \
        -> tuple[dict, dict]:
    public_info = {}
    private_info = {}
    competition = get_competitions_from_ids([CompetitionID.get_enum_element(competition_name)])[0]
    if public_score is not None:
        leader_boards = get_candidate_leaderboard_path(
            competition=competition, root_path_to_leaderboard=f"{leaderboard_dir}/{competition_name}"
        )
        if len(leader_boards) == 0:
            for phase in ["public", "private"]:
                download_leaderboard(
                    kaggle_api=api, competition=competition_name, zip_destination=leaderboard_dir, phase=phase
                )
            leader_boards = get_candidate_leaderboard_path(competition=competition,
                                                           root_path_to_leaderboard=f"{leaderboard_dir}/{competition_name}")

        for leaderboard_file in leader_boards:
            if "public" in leaderboard_file:
                lb_type = "public"
                scores = public_score
                info = public_info
            elif "private" in leaderboard_file:
                lb_type = "private"
                scores = private_score
                info = private_info
            else:
                continue  # skip files that are neither public nor private

            # Load and filter leaderboard
            leaderboard = pd.read_csv(leaderboard_file)
            filtered_lb = handle_benchmark_entry(leaderboard, team_name)
            total_submissions = filtered_lb.shape[0]

            # Compute rank, quantile, medal
            rank, quantile = get_quantiles_from_scores(
                scores=scores,
                leaderboard=filtered_lb,
                is_lower_better=False,
                return_rank=True
            )
            medal = get_medal(rank=rank, n_entries=total_submissions)

            # Update the corresponding info dictionary
            info.update({
                f"{lb_type}_rank": rank,
                f"{lb_type}_quantile": quantile,
                f"{lb_type}_medal": medal.name.value,
                "total_submissions": total_submissions,
            })

    return public_info, private_info


def main(arguments: argparse.Namespace):
    competition = arguments.competition
    submissions_file = arguments.submissions_file
    submissions_dir = arguments.submissions_dir
    leaderboard_dir = arguments.leaderboard_dir
    submission_result_file = arguments.result_file_name
    team_name = args.team_name
    if args.submit_and_validate:
        message = arguments.message
        if message is None:
            raise ValueError("Please provide a message using --message")
        submit_and_rank_all(
            competition_name=competition, submissions_file=submissions_file, message=message,
            leaderboard_dir=leaderboard_dir, submission_score_file_name=submission_result_file,
            team_name=team_name
        )
    elif args.rank_only:
        if arguments.get_n == 1 and arguments.submissions_file is not None:
            rank_one(
                competition_name=competition, submissions_file=arguments.submissions_file,
                leaderboard_dir=leaderboard_dir,
                n_submissions=arguments.get_n, submission_score_file_name=submission_result_file,
                team_name=team_name
            )
        else:
            rank_all(
                competition_name=competition, submissions_dir=submissions_dir, leaderboard_dir=leaderboard_dir,
                n_submissions=arguments.get_n, submission_score_file_name=submission_result_file,
                team_name=team_name
            )
    else:
        print("No option selected.Choose either --submit_and_validate or --rank_only. Terminating program.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle submission")
    parser.add_argument('--competition', type=str, required=True, help='Competition name')
    parser.add_argument('--submissions_file', type=str, required=False, default=None, help='Submission file name')
    parser.add_argument('--submissions_dir', type=str, required=False, default=None, help='Submissions dir')
    parser.add_argument('--message', type=str, required=False, default=None, help='Submission message')
    parser.add_argument('--leaderboard_dir', type=str, required=False,
                        help='Path of leader board csv files already downloaded')
    parser.add_argument('--submit_and_validate', type=bool, required=False, help='Submit file and then rank them')
    parser.add_argument('--rank_only', type=bool, required=False,
                        help='Get scores and rank them , it does not submit submissions')
    parser.add_argument('--get_n', type=int, required=False, default=3, help='Get "N" latest submissions from kaggle')
    parser.add_argument('--result_file_name', type=str, required=False,
                        help='Path of leader board csv files already downloaded', default="submission_scores.json")
    parser.add_argument("--team_name", required=True, help="Team name")

    args = parser.parse_args()
    main(args)
