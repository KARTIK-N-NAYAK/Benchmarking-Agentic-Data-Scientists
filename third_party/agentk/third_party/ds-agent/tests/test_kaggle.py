import json
import logging
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Callable

import kaggle.rest
import pytest
from kaggle import KaggleApi
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

from agent import PROJECT_ROOT
from agent.tools.fetch_tool import FetchTool
from ds_agent.competition_struct import Competition, SubmissionMode
#from ds_agent.utils_kaggle_submission import submit_notebook_file_upload

logger = logging.getLogger(__name__)


@pytest.fixture
def fetch_tool_factory(tmp_path) -> Callable[[str], FetchTool]:
    """Initialises an empty FetchTool that just needs a task_url"""
    return lambda task_id: FetchTool(
        task_url=f"https://www.kaggle.com/competitions/{task_id}",
        user_details=str(PROJECT_ROOT / "third_party" / "data_preprocessing" / "kaggle_login_details.json"),
        workspace_path=str(tmp_path),
        raw_data_dir=str(tmp_path),
        sample_submission_file=None,
        is_local_task=False
    )


def is_kaggle_TooManyRequests(e: BaseException):
    return isinstance(e, kaggle.rest.ApiException) and json.loads(e.body) == {"code": 429, "message": "TooManyRequests"}


def kaggle_rt_wrapper(func):
    return retry(
        retry=retry_if_exception(is_kaggle_TooManyRequests),
        wait=wait_exponential(multiplier=10, exp_base=4),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger=logger, log_level=logging.WARNING),
    )(func)


@pytest.fixture(scope="session")
def kaggle_api():
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    # Set relevant functions to retry TooManyRequest errors
    kaggle_api.competition_submissions = kaggle_rt_wrapper(kaggle_api.competition_submissions)
    kaggle_api.competition_submit = kaggle_rt_wrapper(kaggle_api.competition_submit)

    return kaggle_api


def n_competition_submissions(kaggle_api, competition_id):
    """Attempts to list and count competition submissions"""
    try:
        return len(kaggle_api.competition_submissions(str(competition_id)))
    except kaggle.rest.ApiException as e:
        body_json = json.loads(e.body)
        if body_json == {"code": 400, "message": "You do not have a Team in this Competition."}:
            # Some tasks (e.g. digit-recognizer) return this even when you successfully joined
            print(e)  # Log it because I'm not sure whether it always indicates successful joining
            return 0
        raise


def has_entered_competition(kaggle_api, competition_id) -> bool:
    """Attempts to list competition submissions to check if competition has been joined"""
    try:
        n_competition_submissions(kaggle_api, competition_id)
        return True
    except kaggle.rest.ApiException as e:
        body_json = json.loads(e.body)
        if body_json == {
            "code": 403,
            "message": "You must accept the rules for this competition to perform this action.",
        }:
            return False
        raise  # This can be because "You must Phone Verify"


class TestBenchmark:
    """Tests joining and submission using live kaggle endpoint over every task in TESTED_BENCHMARK"""

    MAX_SUBMISSIONS = 1
    if os.getenv("DS_PATH_RAW_DATA_ROOT") is None:
        raise RuntimeError("DS_PATH_RAW_DATA_ROOT must be set")
    RAW_DATA_ROOT = Path(os.getenv("DS_PATH_RAW_DATA_ROOT"))

    # @pytest.mark.skip
    def test_kaggle_joining(self, competition: Competition, kaggle_api, fetch_tool_factory):
        """Attempts to join a competition using FetchTool's selenium process"""
        task_id_str = competition.competition_id.value

        if self.MAX_SUBMISSIONS is None or not has_entered_competition(kaggle_api, task_id_str):
            fetch_tool = fetch_tool_factory(task_id_str)
            fetch_tool.join_competition(fetch_tool.task_url, implicit_wait_time=10, sleep_time=2)

            # Check the competition has been successfully joined
            assert self.MAX_SUBMISSIONS is None or has_entered_competition(kaggle_api, task_id_str)
        else:
            print(f"Kaggle user has already entered competition ({task_id_str})")


    def test_kaggle_submission(self, competition: Competition, kaggle_api, fetch_tool_factory):
        """Downloads and attempts to submit the sample_submission.csv file"""

        task_id_str = competition.competition_id.value

        if competition.submission_mode == SubmissionMode.FILE_UPLOAD:
            # We check that the total account submissions are low to avoid stacking up resubmission
            n_submissions = None if self.MAX_SUBMISSIONS is None else n_competition_submissions(kaggle_api, task_id_str)
            if self.MAX_SUBMISSIONS is None or n_submissions < self.MAX_SUBMISSIONS:
                sample_submission_path = self.RAW_DATA_ROOT / task_id_str / competition.sample_submission_filename
                if not sample_submission_path.exists():
                    raise FileNotFoundError(
                        "Sample submission file does not exist! "
                        "(Maybe RAW_DATA_ROOT is incorrect/incomplete or sample filename is wrong): "
                        + str(sample_submission_path)
                    )

                result = kaggle_api.competition_submit(
                    file_name=sample_submission_path,
                    message="test_submission_sample",
                    competition=task_id_str,
                )

                assert "Successfully submitted" in result.message, vars(result)

                # assert (
                #     self.MAX_SUBMISSIONS is None or n_competition_submissions(kaggle_api, task_id_str) == n_submissions + 1
                # )
            else:
                print(f"Kaggle user already has too many submissions ({n_submissions}) to this task ({task_id_str})")

        elif competition.submission_mode == SubmissionMode.NOTEBOOK_FILE_UPLOAD:
            task_id_str = competition.competition_id.value
            fetch_tool = fetch_tool_factory(task_id_str)
            fetch_tool.submit_notebook_file_upload(competition_slug=task_id_str, implicit_wait_time=15, sleep_time=5)

    def test_kaggle_leaderboard_view(self, competition, kaggle_api):
        """Checks whether the leaderboard can be retrieved using the api and is of minimum size"""
        task_id_str = competition.competition_id.value

        leaderboard = kaggle_api.competition_leaderboard_view(task_id_str)
        n_submissions = len(leaderboard)

        assert n_submissions >= 20, leaderboard

    def test_kaggle_leaderboard_ranking(self, competition, kaggle_api):
        """Try to find the rank of the last submission"""
        task_id_str = competition.competition_id.value

        leaderboard = kaggle_api.competition_leaderboard_view(task_id_str)
        last_submission = kaggle_api.competition_submissions(task_id_str)[0]

        private_score = last_submission.privateScoreNullable
        public_score = last_submission.publicScoreNullable
        score = private_score if private_score is not None else public_score
        assert (
            score is not None
        ), f"Both public and private score are None for the last submission: {vars(last_submission)}"
        score = float(score)

        leaderboard_scores = [float(entry.score) for entry in leaderboard if entry.hasScore]
        assert leaderboard_scores

        higher_is_better = leaderboard_scores[0] > leaderboard_scores[-1]

        def negate_if_higher_is_better(s):
            return -s if higher_is_better else s

        n_scores = len(leaderboard_scores)
        left_i = bisect_left(leaderboard_scores, score, key=negate_if_higher_is_better)
        right_i = bisect_right(leaderboard_scores, score, key=negate_if_higher_is_better)

        print(f"{'Higher' if higher_is_better else 'Lower'} is better")
        print(f"Rank: {left_i}-{right_i} out of {n_scores}")
        print(f"Score: {score} from range {leaderboard_scores[0]}-{leaderboard_scores[-1]}")
