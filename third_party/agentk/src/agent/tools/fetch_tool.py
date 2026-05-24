import glob
import hashlib
import io
import json
import os
import re
import subprocess as sp
import tarfile
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
import py7zr
import rarfile
import selenium
import textdistance
from agent.memory import MemKey
from agent.tools.base_tool import Tool
from ds_agent.utils import set_pd_options, reset_pd_options
from py7zr.exceptions import PasswordRequired
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm

from third_party.data_science.utils import get_agent_root_dir, get_raw_data_root_dir

DETECT_SAMPLE_SUBMISSION_WITH_LLM_FLAG = "@DETECT SAMPLE SUBMISSION WITH LLM@"
DETECT_PASSWORD_FILE_WITH_LLM_FLAG = "@DETECT PASSWORD FILE WITH LLM@"
SUPPORTED_COMPRESSED_FORMATS = [".zip", ".7z", ".tar", ".gz", ".tar.gz", ".rar", ".tgz"]

os.environ["MOZ_HEADLESS"] = "1"

RAW_DATA_DIR = get_raw_data_root_dir()


# declaring it here for test purposes
SCHEMA_DICT = {
    'aerial-cactus-identification': {
        "fields": [
            {
                "name": "id",
                "description": "",
                "type": "string"
            },
            {
                "name": "has_cactus",
                "description": "",
                "type": "decimal"
            },
        ]
    },
    'petfinder-pawpularity-score': {
        "fields": [
            {
                "name": "Id",
                "description": "",
                "type": "string"
            },
            {
                "name": "Pawpularity",
                "description": "",
                "type": "number"
            },
        ]
    },
    'icr-identify-age-related-conditions': {
        "fields": [
            {
                "name": "Id",
                "description": "",
                "type": "string"
            },
            {
                "name": "class_0",
                "description": "",
                "type": "number"
            },
            {
                "name": "class_1",
                "description": "",
                "type": "number"
            }
        ]
    },
    'quora-insincere-questions-classification': {
        "fields": [
            {
                "name": "qid",
                "description": "",
                "type": "string"
            },
            {
                "name": "prediction",
                "description": "",
                "type": "integer"
            },
        ]
    },
}


class SampleSubmissionIdentificationError(Exception):
    pass


class PasswordFileNotFoundError(Exception):
    pass


class ManyPasswordFilesError(Exception):
    pass


class FetchTool(Tool):
    name: str = "Fetch tool"
    requires_llm_prompt: bool = False

    def __init__(
            self,
            task_url: str,
            workspace_path: str,
            raw_data_dir: str,
            user_details: str,
            is_local_task: bool,
            sample_submission_file: str | None = None,
    ):
        """
        Args:
            sample_submission_file: name of the file containing the sample submission
            is_local_task: whether the task is defined in a local folder
        """
        self.task_url = task_url
        self.is_local_task = is_local_task
        self.workspace_path = workspace_path
        self.raw_data_dir = raw_data_dir
        self.user_details = user_details

        self.raw_task_description_path = os.path.join(self.raw_data_dir, "raw_task_description.txt")
        self.raw_data_description_path = os.path.join(self.raw_data_dir, "raw_data_description.txt")
        self.raw_metric_description_path = os.path.join(self.raw_data_dir, "raw_metric_description.txt")
        self.raw_table_view_path = os.path.join(self.raw_data_dir, "raw_table_view.txt")
        self.raw_table_info_path = os.path.join(self.raw_data_dir, "raw_table_info.txt")
        self.raw_data_view_path = os.path.join(self.raw_data_dir, "raw_data_view.txt")

        self.fetched_path = os.path.join(self.raw_data_dir, 'fetched_raw_data_done.json')
        if not self.is_local_task:
            if os.path.exists(self.fetched_path):
                self.already_fetched = json.load(open(self.fetched_path, "r"))["fetched_raw_data_done"]
            else:
                self.already_fetched = False
        else:
            self.already_fetched = True

        self.detected_file = None
        self.sample_submission_file = sample_submission_file

    def __call__(self, agent_input: str) -> dict[MemKey, str | bool]:
        """
        Uses Kaggle api cli to fetch a competition given an url.
        As fetching requires to accept the competition rules, this is done automatically.
        This Tool requires a valid kaggle login json file.
        """
        if not self.already_fetched:
            # first login and accept rules, so we can then fetch competition data
            self.join_competition(self.task_url, implicit_wait_time=15, sleep_time=5)
            self.get_dataset(name=os.path.basename(self.task_url), download_dir=self.raw_data_dir)
            self.check_raw_data_dir(download_dir=self.raw_data_dir)

        (detected_file, has_sample_submission,
         id_name, target_names, sample_submission_head) = self.review_sample_submission(
            raw_data_dir=self.raw_data_dir,
            workspace_path=self.workspace_path,
            task_id=os.path.basename(self.task_url),
            sample_submission_file=self.sample_submission_file,
        )
        # get metadata by scraping and summarize it
        if self.is_local_task:
            with open(Path(self.raw_data_dir) / "raw_data_description.txt", 'r') as f:
                raw_data_description = f.read()

            with open(Path(self.raw_data_dir) / "raw_metric_description.txt", 'r') as f:
                raw_metric_description = f.read()

            with open(Path(self.raw_data_dir) / "raw_task_description.txt", 'r') as f:
                raw_task_description = f.read()

        else:
            raw_task_description = self.scrape_task_description(self.task_url)
            raw_data_description = self.scrape_data_description(self.task_url)
            raw_metric_description = self.scrape_metric_description(self.task_url)

        raw_table_view = self.get_raw_table_view(
            download_dir=self.raw_data_dir,
            raw_table_view_path=self.raw_table_view_path,
            detected_file=detected_file,
            fetched_path=self.fetched_path
        )
        raw_table_info = self.get_raw_table_info(
            download_dir=self.raw_data_dir,
            raw_table_info_path=self.raw_table_info_path,
            detected_file=detected_file,
            fetched_path=self.fetched_path
        )
        raw_data_view = self.get_raw_data_view(
            download_dir=self.raw_data_dir,
            raw_data_view_path=self.raw_data_view_path,
            detected_file=detected_file,
        )

        print("Fetching complete!")

        if not self.already_fetched:
            json.dump({"fetched_raw_data_done": True}, open(self.fetched_path, 'w'))

        return {
            MemKey.RAW_TASK_DESCRIPTION: raw_task_description,
            MemKey.RAW_DATA_DESCRIPTION: raw_data_description,
            MemKey.RAW_METRIC_DESCRIPTION: raw_metric_description,
            MemKey.RAW_TABLE_VIEW: raw_table_view,
            MemKey.RAW_TABLE_INFO: raw_table_info,
            MemKey.RAW_DATA_VIEW: raw_data_view,
            MemKey.RAW_ID_COLUMN_NAME: id_name,
            MemKey.RAW_TARGETS_COLUMN_NAMES: target_names,
            MemKey.DETECT_SAMPLE_SUBMISSION_WITH_LLM: detected_file == DETECT_SAMPLE_SUBMISSION_WITH_LLM_FLAG,
            MemKey.HAS_SAMPLE_SUBMISSION: has_sample_submission,
            MemKey.SAMPLE_SUBMISSION_HEAD: sample_submission_head,
            MemKey.FETCHED_RAW_DATA: True,
        }

    @staticmethod
    def _find_button_and_click(driver, button_text) -> bool:
        buttons = driver.find_elements(By.CSS_SELECTOR, value="button[role=button]")
        for button in buttons:
            if button.text == button_text or button_text in button.text:
                driver.execute_script("arguments[0].click();", button)
                driver.implicitly_wait(10)
                print(f'Found and clicked on button "{button_text}"')
                return True
        return False
        # raise RuntimeError(f"Failed to find button with text '{button_text}'!")

    @staticmethod
    def review_sample_submission(
            raw_data_dir: str,
            workspace_path: str,
            task_id: str,
            sample_submission_file: str = None,
            detected_file: str = None
    ) -> Tuple[str, bool, str, str, str]:
        """
        Checks raw data dir for the file `sample_submission.csv` or a file with a similar name.
        If it does not find it, raise an error for now as some unit tests depend on its existence.

        The Id column name is then extracted as well as the target name(s) from the column names of
        `sample_submission.csv` and returned, to be saved for later use.

        Returns a flag True/False to notify the presence of the `sample_submission.csv` and the names.
        """
        if detected_file is None:
            # find all files that are similar to 'sample_submission'
            # searching in all subdirectories, limiting to depth of 1
            files = []
            # scan_dir = os.path.join(raw_data_dir, "./**/*")
            scan_dir = os.path.join(raw_data_dir, "./*")
            for path in tqdm(glob.iglob(scan_dir, recursive=True), desc=f"scanning {scan_dir}"):
                if path.endswith(".csv"):
                    files.append(Path(path))

            if len(files) == 0:
                raise ValueError(f"No possible sample submission found with this path {scan_dir}!")
            file_names = [x.name for x in files]
            complete_file_names = {x.name: str(x) for x in files}
            min_leven_dist_file = min(file_names,
                                      key=lambda x: textdistance.levenshtein("sample_submission.csv", x.lower()))
            min_bag_dist_file = min(file_names, key=lambda x: textdistance.bag("sample_submission.csv", x.lower()))
            longest_subseq_file = max(file_names, key=lambda x: textdistance.lcsseq("sample_submission.csv", x.lower()))
            longest_substr_file = max(file_names, key=lambda x: textdistance.lcsstr("sample_submission.csv", x.lower()))
            detected_files = [min_leven_dist_file, min_bag_dist_file, longest_substr_file, longest_subseq_file]
            votes, max_votes = {}, 0
            for x in detected_files:
                if x in votes:
                    votes[x] += 1
                else:
                    votes[x] = 1
                max_votes = max(max_votes, votes[x])
            most_voted_files = [x for x in votes if votes[x] == max_votes]
            if len(most_voted_files) > 1:
                detected_file = DETECT_SAMPLE_SUBMISSION_WITH_LLM_FLAG
            else:
                detected_file = complete_file_names[most_voted_files[0]]

            if detected_file == DETECT_SAMPLE_SUBMISSION_WITH_LLM_FLAG:
                return detected_file, False, '', '', ''

        if sample_submission_file is not None and sample_submission_file != detected_file:
            raise SampleSubmissionIdentificationError(
                f"Expected sample submission file {sample_submission_file} but got {detected_file}"
            )

        matched_words_sample_submissions = ["sample", "entry", "submission", "benchmark", "submit"]
        words_re = re.compile("|".join(matched_words_sample_submissions))

        # copy sample submission into workspace and save id name and target names to JSON file
        if words_re.search(detected_file.lower()):
            os.system(f"cp {os.path.join(raw_data_dir, detected_file)} {workspace_path}/data/sample_submission.csv")
            print(f"Copied {detected_file} to workspace {workspace_path}/data/sample_submission.csv")

            df = pd.read_csv(os.path.join(raw_data_dir, detected_file))
            id_name, target_names = str(df.columns[0]), list(df.columns[1:])

            _table_view = FetchTool.df_formated_head_view(
                input_df_path=os.path.join(raw_data_dir, detected_file))

            json.dump(
                obj={"id_name": id_name, "target_names": target_names},
                fp=open(os.path.join(workspace_path, "metadata/submission_names.json"), "w"),
                indent=True,
            )
            return detected_file, True, id_name, ",".join(target_names), _table_view

        else:
            raise RuntimeError(
                f"Warning: sample_submission.csv does not exist as part of the raw competition data.\n"
                f"Review your unit tests accordingly as the default ones may assume the existence of that file.\n"
                f"TASK_ID: {task_id}\n"
                f"FILES CONSULTED: {os.listdir(raw_data_dir)}"
            )

    @staticmethod
    def create_dataset(
            dataset_id: str,
            username: str,
            dataset_dir: Path,
            title: str,
            submission_path: Path,
            schema: dict[str, Any],
            subtitle: str = "",
            description: str = "",
            id_no: int = 12345,
    ):
        dataset_metadata = {
            "title": title,
            "subtitle": subtitle,
            "description": description,
            "id": f"{username}/{dataset_id}",
            "id_no": id_no,
            "licenses": [{"name": "CC0-1.0"}],
            "resources": [
                {
                    "path": str(submission_path),
                    "description": "submission data",
                    "schema": schema
                },
            ],
        }

        json.dump(dataset_metadata, open(dataset_dir / f"dataset-metadata.json", "w"), indent=4)

    def sign_in_competition(self, driver: webdriver.Firefox, competition_slug: str, implicit_wait_time: int = 10,
                            sleep_time: int = 2, **kwargs) -> None:
        page = kwargs.get("page", "rules")
        driver.get(f"https://www.kaggle.com/competitions/{competition_slug}/{page}")
        self._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)
        user_details = json.load(open(self.user_details))
        self._sign_in(
            driver=driver,
            user_email=user_details["email"],
            user_pwd=user_details["pwd"],
            implicit_wait_time=implicit_wait_time,
            sleep_time=sleep_time
        )

    @staticmethod
    def _sign_in(driver: webdriver.Firefox, user_email: str, user_pwd: str,
                 implicit_wait_time: int = 10, sleep_time: int = 2) -> None:
        _start_url = driver.current_url

        FetchTool._find_button_and_click(driver, button_text="Sign In")
        FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)

        FetchTool._find_button_and_click(driver, button_text="Email")
        FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)

        # add email and password in boxes
        email_box = driver.find_element(By.XPATH, '//input[@placeholder="Enter your email address or username"]')
        email_box.send_keys(user_email)
        password_box = driver.find_element(By.XPATH, '//input[@placeholder="Enter password"]')
        password_box.send_keys(user_pwd)

        FetchTool._find_button_and_click(driver, button_text="Sign In")

        # wait until sign in is done
        _counter = 0
        while driver.current_url != _start_url and _counter < 10:
            FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)
            _counter += 1
        if driver.current_url == _start_url:
            print("Sign In complete")
            # now that we are logged in, join competition
            FetchTool._find_button_and_click(driver, button_text="I Understand and Accept")
            FetchTool._find_button_and_click(driver, button_text="Late Submission")
            FetchTool._find_button_and_click(driver, button_text="I Understand and Accept")

        else:
            raise RuntimeError("Could not Sign In")

    def join_competition(self, url: str, implicit_wait_time: int = 10, sleep_time: int = 3) -> None:
        """
        Uses Selenium and FireFox webdriver headless to parse the competition website and join the competition
        given by the URL if not already joined.
        This method requires the user to have set up a kaggle account and have its credentials saved the same way as
        required by the Kaggle API
        """
        kaggle_login = json.load(open(self.user_details))
        kaggle_login_mail = kaggle_login.get("email", None)
        kaggle_login_pwd = kaggle_login.get("pwd", None)
        if kaggle_login_mail is None:
            kaggle_login_mail = kaggle_login.get("login_email", None)
        if kaggle_login_pwd is None:
            kaggle_login_pwd = kaggle_login.get("login_pwd", None)

        assert kaggle_login_mail is not None, f"Please provide a kaggle `email` and `pwd` in {self.user_details}"
        assert kaggle_login_pwd is not None, f"Please provide a kaggle `email` and `pwd` in {self.user_details}"

        opts = FirefoxOptions()
        opts.headless = True

        try:
            with webdriver.Firefox(options=opts) as driver:
                # go to competition Rules page
                driver.get(f"{url}/rules")
                driver.implicitly_wait(implicit_wait_time)
                wait = WebDriverWait(driver, implicit_wait_time)
                time.sleep(sleep_time)

                # choose to Sign in
                self._find_button_and_click(driver, button_text="Sign In")
                # select to sign in with email
                # we need to wait for the sign-in options form to load before trying to find the Email sign in
                FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)
                self._find_button_and_click(driver, button_text="Email")

                # add email and password in boxes
                emails = driver.find_elements(By.NAME, "email")

                while len(emails) == 0:
                    FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)
                    # go to competition Rules page
                    driver.get(f"{url}/rules")
                    driver.implicitly_wait(implicit_wait_time)
                    wait = WebDriverWait(driver, implicit_wait_time)
                    time.sleep(sleep_time)

                    # choose to Sign in
                    self._find_button_and_click(driver, button_text="Sign In")
                    # select to sign in with email
                    # we need to wait for the sign-in options form to load before trying to find the Email sign in
                    self._find_button_and_click(driver, button_text="Email")

                    emails = driver.find_elements(By.NAME, "email")

                email = emails[0]
                email.send_keys(kaggle_login_mail)

                password = driver.find_element(By.NAME, "password")
                password.send_keys(kaggle_login_pwd)

                # select button to sign in and click on it
                self._find_button_and_click(driver, button_text="Sign In")

                # wait until sign in is done
                FetchTool._driver_wait(driver=driver, implicit_wait_time=implicit_wait_time, sleep_time=sleep_time)
                wait.until(EC.url_matches(f"{url}/rules"))
                print("Sign In complete")
                # now that we are logged in, join competition

                accept = self._find_button_and_click(driver, button_text="I Understand and Accept")
                late_submission = self._find_button_and_click(driver, button_text="Late Submission")

                if not accept and not late_submission:
                    join_competition = self._find_button_and_click(driver, button_text="Join Competition")
                    if not join_competition:
                        raise selenium.common.exceptions.NoSuchElementException(
                            "Neither 'Join Competition' nor 'Late Submission' button found "
                            "-- it's likely you have already joined"
                        )
                    # Wait for rules accept popup
                    wait.until(
                        EC.presence_of_element_located(
                            (By.XPATH, "//div[@id='kaggle-portal-root-global']/div[@role='presentation']")
                        )
                    )
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    accept2 = self._find_button_and_click(driver, button_text="I Understand and Accept")

                    if not accept2:
                        raise RuntimeError(
                            f"Could not accept: {accept=}, {late_submission=}, {join_competition=} {accept2=}"
                        )

        except selenium.common.exceptions.NoSuchElementException as e:
            print(e)

        except RuntimeError as e:
            print(e)

        finally:
            # Close the browser window
            if "driver" in locals():
                driver.quit()

    def _datetime_hash(self):
        return hashlib.sha1(str(time.time()).encode("utf-8")).hexdigest()

    @staticmethod
    def _driver_wait(driver: webdriver.Firefox, implicit_wait_time: int = 10, sleep_time: int = 2):
        driver.implicitly_wait(implicit_wait_time)
        time.sleep(sleep_time)

    @staticmethod
    def get_dataset(name: str, download_dir: str) -> None:
        """
        Uses the Kaggle API to pull the data of a competition given its URL
        Args:
            name: (str) Kaggle competition name from the URL
            download_dir: (str) directory where to save competition data
        """
        if os.path.exists(download_dir) and len(os.listdir(download_dir)) > 0:
            print(
                f"Directory {download_dir} already exists and is not empty! "
                f"Please check files and remove if you want to overwrite."
            )
            FetchTool.decompress_all_files(download_dir=download_dir)
        else:
            os.makedirs(download_dir, mode=0o777, exist_ok=True)
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            print(download_dir)
            kaggle_api.competition_download_files(name, path=download_dir, quiet=False)
            FetchTool.decompress_all_files(download_dir=download_dir)

    @staticmethod
    def decompress_all_files(download_dir: str) -> None:
        # unzip files
        decompressed_names = []
        compressed_names = [
            z for z in os.listdir(download_dir) if
            os.path.splitext(z)[1] in SUPPORTED_COMPRESSED_FORMATS
        ]
        _step_limit = 1000
        _counter = 0
        while len(compressed_names) > 0 and _counter < _step_limit:
            for compressed_name in compressed_names:
                decompressed_name = FetchTool.decompress_file(directory=download_dir, file=compressed_name)
                if decompressed_name is not None:
                    decompressed_names.append(decompressed_name)

            compressed_names = [
                z for z in os.listdir(download_dir) if
                (os.path.splitext(z)[1] in SUPPORTED_COMPRESSED_FORMATS and z not in decompressed_names)
            ]
            _counter += 1

        if _counter == _step_limit:
            raise RuntimeError(f"Tried to unzip {_step_limit} times, there are still compressed files to inflate:"
                               f" {compressed_names}. Check if there is a problem or if there are indeed more than"
                               f" {_step_limit} compressed files to inflate, increase step limit.")
        print("Unzipped all files.")

    @staticmethod
    def _view_zip_contents(file_path, password=None) -> list[str] | bool:
        """
        Returns the file(s) that would be decompressed.
        If no password is passed but the file is password-protected, returns a bool
        """
        if password is None:
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    return zip_ref.namelist()
            except RuntimeError:
                requires_password = True
                return requires_password
        else:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.setpassword(password)
                return zip_ref.namelist()

    @staticmethod
    def _view_7z_contents(file_path, password=None) -> list[str] | bool:
        if password is None:
            try:
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    return archive.getnames()
            except PasswordRequired:
                requires_password = True
                return requires_password
        else:
            with py7zr.SevenZipFile(file_path, mode='r', password=password) as archive:
                return archive.getnames()

    @staticmethod
    def _view_tar_contents(file_path, password=None) -> list[str] | bool:
        with tarfile.open(file_path, 'r') as tar_ref:
            return tar_ref.getnames()

    @staticmethod
    def _view_rar_contents(file_path, password=None) -> list[str] | bool:
        if password is None:
            try:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    return rar_ref.namelist()
            except rarfile.PasswordRequired:
                requires_password = True
                return requires_password
        with rarfile.RarFile(file_path) as rar_ref:
            rar_ref.setpassword(password)
            return rar_ref.namelist()

    @staticmethod
    def retrieve_password(directory: str) -> str | None:
        """Tries to find password file if there is any, otherwise returns None"""
        extension_list = ['txt']
        password_files = [file_name for file_name in os.listdir(directory) if
                          any(file_name.endswith(ext) for ext in extension_list)]
        print(f"The length of password files: {len(password_files)}")
        if len(password_files) == 1:
            password_file = password_files[0]
        elif len(password_files) > 1:
            min_leven_dist_file = min(password_files,
                                      key=lambda x: textdistance.levenshtein("password.txt", x.lower()))
            min_bag_dist_file = min(password_files,
                                    key=lambda x: textdistance.bag("password.txt", x.lower()))
            longest_subseq_file = max(password_files,
                                      key=lambda x: textdistance.lcsseq("password.txt",
                                                                        x.lower()))
            longest_substr_file = max(password_files,
                                      key=lambda x: textdistance.lcsstr("password.txt",
                                                                        x.lower()))
            detected_files = [min_leven_dist_file, min_bag_dist_file, longest_substr_file,
                              longest_subseq_file]
            votes, max_votes = {}, 0
            for x in detected_files:
                if x in votes:
                    votes[x] += 1
                else:
                    votes[x] = 1
                max_votes = max(max_votes, votes[x])
            most_voted_files = [x for x in votes if votes[x] == max_votes]
            if len(most_voted_files) > 1:
                raise ManyPasswordFilesError("Many password files.")
            else:
                password_file = most_voted_files[0]

        else:
            password_file = None

        return password_file

    @staticmethod
    def decompress_file(directory: str, file: str) -> str | None:
        _, extension = os.path.splitext(file)
        file_path = Path(directory, file)
        requires_password = False
        if extension == ".zip":
            req = FetchTool._view_zip_contents(file_path)
            if isinstance(req, bool) and req:
                requires_password = True
        elif extension == ".7z":
            req = FetchTool._view_7z_contents(file_path)
            if isinstance(req, bool) and req:
                requires_password = True
        elif extension in [".tar", ".gz", ".tar.gz", ".tgz"]:
            req = FetchTool._view_tar_contents(file_path)
            if isinstance(req, bool) and req:
                requires_password = True
        elif extension == ".rar":
            req = FetchTool._view_rar_contents(file_path)
            if isinstance(req, bool) and req:
                requires_password = True

        if requires_password:
            password_file = FetchTool.retrieve_password(directory)
            if password_file is not None:
                with open(Path(directory, password_file), 'r') as password_file:
                    password = password_file.read().strip()
                return FetchTool._decompress_file_with_password(directory=directory, file=file, password=password)
            else:
                raise PasswordFileNotFoundError(f"Password file not found related to {file}")
        else:
            return FetchTool._decompress_file_no_password(directory=directory, file=file)

    @staticmethod
    def _decompress_file_with_password(directory: str, file: str, password: str) -> str | None:
        filename, extension = os.path.splitext(file)
        file_path = Path(directory, file)
        new_directory = Path(directory, filename)

        if extension == ".zip":
            file_list = FetchTool._view_zip_contents(file_path, password)
            if len(file_list) > 1:
                os.makedirs(new_directory, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.setpassword(bytes(password, 'utf-8'))
                    zip_ref.extractall(new_directory)
            else:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.setpassword(bytes(password, 'utf-8'))
                    zip_ref.extractall(directory)
        elif extension == ".7z":
            file_list = FetchTool._view_7z_contents(file_path, password)
            if len(file_list) > 1:
                os.makedirs(new_directory, exist_ok=True)
                with py7zr.SevenZipFile(file_path, mode='r', password=password) as archive:
                    archive.extractall(path=new_directory)
            else:
                with py7zr.SevenZipFile(file_path, mode='r', password=password) as archive:
                    archive.extractall(path=new_directory)
        elif extension in [".tar", ".gz", ".tar.gz", ".tgz"]:
            file_list = FetchTool._view_tar_contents(file_path)
            if len(file_list) > 1:
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(new_directory)
            else:
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(directory)
        elif extension == ".rar":
            file_list = FetchTool._view_rar_contents(file_path, password)
            if len(file_list) > 1:
                with rarfile.RarFile(file_path, mode='r') as rar_ref:
                    rar_ref.setpassword(password)
                    rar_ref.extractall(path=new_directory)
            else:
                with rarfile.RarFile(file_path, mode='r') as rar_ref:
                    rar_ref.setpassword(password)
                    rar_ref.extractall(path=new_directory)
        else:
            raise ValueError(f"Compressed file {file} has unsupported format {extension}")
        os.remove(file_path)
        return file

    @staticmethod
    def _decompress_file_no_password(directory: str, file: str) -> str | None:
        filename, extension = os.path.splitext(file)
        file_path = Path(directory, file)
        new_directory = Path(directory, filename)

        if extension == ".zip":
            file_list = FetchTool._view_zip_contents(file_path)
            if len(file_list) > 1:
                os.makedirs(new_directory, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(new_directory)
            else:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(directory)
        elif extension == ".7z":
            file_list = FetchTool._view_7z_contents(file_path)
            if len(file_list) > 1:
                os.makedirs(new_directory, exist_ok=True)
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(path=new_directory)
            else:
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(path=directory)
        elif extension in [".tar", ".gz", ".tar.gz", ".tgz"]:
            file_list = FetchTool._view_tar_contents(file_path)
            if len(file_list) > 1:
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(new_directory)
            else:
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(directory)
        elif extension == ".rar":
            file_list = FetchTool._view_rar_contents(file_path)
            if len(file_list) > 1:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(path=new_directory)
            else:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(path=directory)
        else:
            raise ValueError(f"Compressed file {file} has unsupported format {extension}")
        os.remove(file_path)
        return file

    @staticmethod
    def check_raw_data_dir(download_dir: str) -> None:
        """
        Check if all files are contained in a subfolder or `self.raw_data_dir/task_id` and
        if yes, then peel of one level, until all files are not contained in a subfolder.
        """
        while len(os.listdir(download_dir)) == 1:
            single_folder_name = os.listdir(download_dir)[0]
            os.system(f"mv {download_dir}/{single_folder_name}/* {download_dir}/")
            os.system(f"rm -rf {download_dir}/{single_folder_name}")
            print(f"All files are contained in {download_dir}/{single_folder_name} -> "
                  f"moving everything out to {download_dir}", flush=True)

    def scrape_task_description(self, url: str) -> str:
        """
        scrape the competition Overview page and extract the text using Seleniun through a Firefox Driver.
        Make sure to have downloaded the driver and copied it into /usr/local/bin or /usr/bin
        """
        if os.path.exists(self.raw_task_description_path):
            with open(self.raw_task_description_path, "r") as f:
                raw_task_description = f.read()
            return raw_task_description
        else:
            opts = FirefoxOptions()
            opts.headless = True

            try:
                with webdriver.Firefox(options=opts) as driver:
                    driver.get(url)
                    driver.implicitly_wait(10)
                    time.sleep(2)
                    try:
                        description_element = driver.find_element(By.ID, "description")
                    except selenium.common.exceptions.NoSuchElementException:
                        description_element = driver.find_element(By.ID, "abstract")

                    try:
                        evaluation_element = driver.find_element(By.ID, "evaluation")
                    except selenium.common.exceptions.NoSuchElementException:
                        evaluation_element = None

                    description_text = description_element.text
                    if evaluation_element is not None:
                        evaluation_text = evaluation_element.text
                        task_description = description_text + "\n----------\n" + evaluation_text
                    else:
                        task_description = description_text

                    with open(self.raw_task_description_path, "w") as f:
                        f.write(task_description.strip())
                    return task_description.strip()

            except Exception as e:
                raise Exception(f"An error occurred when scrapping Competition {url} for its description:\n{e}")

            finally:
                # Close the browser window
                driver.quit()

    def scrape_data_description(self, url: str) -> str:
        """
        scrape the competition Overview page and extract the text using Seleniun through a Firefox Driver.
        Make sure to have downloaded the driver and copied it into /usr/local/bin or /usr/bin
        """
        if os.path.exists(self.raw_data_description_path):
            with open(self.raw_data_description_path, "r") as f:
                raw_data_description = f.read()
            return raw_data_description
        else:
            opts = FirefoxOptions()
            opts.headless = True

            try:
                with webdriver.Firefox(options=opts) as driver:
                    driver.get(f"{url}/data")
                    driver.implicitly_wait(10)
                    time.sleep(2)
                    self._find_button_and_click(driver, button_text="See More")
                    element = driver.find_element(By.XPATH, "//*[contains(text(), 'Dataset Description')]")
                    parent_element = element.find_element(By.XPATH, "../../../../..")
                    description = parent_element.text
                    with open(self.raw_data_description_path, "w") as f:
                        f.write(description.strip())
                    return description.strip()

            except Exception as e:
                print(e)

            finally:
                # Close the browser window
                driver.quit()

    def scrape_metric_description(self, url: str) -> str:
        """
        scrape the competition Overview page and extract the text using Seleniun through a Firefox Driver.
        Make sure to have downloaded the driver and copied it into /usr/local/bin or /usr/bin
        """
        if os.path.exists(self.raw_metric_description_path):
            with open(self.raw_metric_description_path, "r") as f:
                raw_metric_description = f.read()
            return raw_metric_description
        else:
            opts = FirefoxOptions()
            opts.headless = True

            try:
                with webdriver.Firefox(options=opts) as driver:
                    driver.get(url)
                    driver.implicitly_wait(10)
                    time.sleep(2)
                    try:
                        evaluation_element = driver.find_element(By.ID, "evaluation")
                    except selenium.common.exceptions.NoSuchElementException:
                        evaluation_element = driver.find_element(By.ID, "description")
                    evaluation_text = evaluation_element.text
                    evaluation_description = evaluation_text
                    with open(self.raw_metric_description_path, "w") as f:
                        f.write(evaluation_description.strip())
                    return evaluation_description.strip()

            except Exception as e:
                raise Exception(f"An error occurred when scrapping Competition {url} for its description:\n{e}")

            finally:
                # Close the browser window
                driver.quit()

    @staticmethod
    def save_discussion_to_json(data, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the data to a JSON file
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    @staticmethod
    def get_kaggle_competition_level(user_id: str) -> str | None:
        """
        Fetches the competition level of a Kaggle user.
        """
        url = f"https://www.kaggle.com/{user_id}/competitions"
        opts = FirefoxOptions()
        opts.add_argument("--headless")

        with webdriver.Firefox(options=opts) as driver:
            driver.get(url)
            wait = WebDriverWait(driver, 10)

            try:
                target_element = wait.until(
                    EC.visibility_of_element_located(
                        (By.XPATH,
                         '/html/body/main/div[1]/div/div[5]/div[2]/div/div[2]/div/div/div[5]/div[2]/div[2]/div[2]/div[1]/div[1]/a')
                    )
                )
                t = target_element.text
                level = t.split("\n")[1]

                if level not in ["Expert", "Master", "Grandmaster", "Novice", "Contributor"]:
                    raise RuntimeError(f"Unexpected level found: {level}")

                return level
            except Exception as e:
                print(f"Error retrieving competition level for {user_id}: {e}")
                return None

        # Close the browser window
        driver.quit()

    @staticmethod
    def scrape_comments(url: str, driver: webdriver.Firefox, **kwargs) -> list[Dict]:
        """
        scrapes the comments of a particular discussion thread
        returns the user id and comments except for the main comment
        """
        driver.get(url)

        # on the discussion page try to scrape the first comment
        comments = driver.find_elements(
            By.XPATH, '//div[@data-testid="discussions-comment"]'
        )  # driver.find_elements(By.CSS_SELECTOR, "div.sc-dnOVKr.dYwnhY")
        print(len(comments))

        comment_list = []

        for i in range(1, len(comments[1:21]) + 1):
            regex_pattern = r'(?<=\nmore_vert\n)(.*?)(?=\nreply)'

            match = re.search(regex_pattern, comments[i].text, re.DOTALL)
            if match:
                cleaned_comment = match.group(1)
            else:
                cleaned_comment = None

            user_url = comments[i].find_element(By.TAG_NAME, "a").get_attribute("href")
            user_id = user_url.split("/")[-1]
            user_level = FetchTool.get_kaggle_competition_level(user_id)
            comment_dict = {
                'user_url': user_url,
                'comment date': comments[i].find_element(By.CSS_SELECTOR, "div p span").get_attribute("title"),
                'comment': cleaned_comment,
                'user_level': user_level
            }

            comment_list.append(comment_dict)

        return comment_list

    @staticmethod
    def scrape_discussion(url: str, driver: webdriver.Firefox, **kwargs) -> Dict:
        """
        The function scrapes discussion separating main heading and the
            following comments
            """

        # go to the discussion which has to be scrapped
        # driver.get(href_value)

        # scrapping the main comment and its meta data
        wait = WebDriverWait(driver, 10)
        heading = driver.find_element(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[1]/div[1]/h3").text
        user = driver.find_element(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[1]/div[1]/div[1]/div/a").get_attribute(
            "href")  # instead get the link

        # for the main comment
        div = driver.find_element(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[1]/div[1]/div[3]/div/div")  # the div containing tha main comment
        para = div.find_elements(By.CSS_SELECTOR, "p")
        comment = ' '.join([p.text for p in para])

        # for the links in the comment
        links = div.find_elements(By.XPATH, "//a[@rel='noreferrer nofollow']")
        href_in_main = [href.get_attribute('href') for href in links]

        # for date and time of the comment
        date_element = driver.find_element(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[1]/div[1]/div[1]/div/span/span")
        date_and_time = date_element.get_attribute('title')

        user_comments = FetchTool.scrape_comments(url, driver, num_comments=5)

        # creating the dict with the details
        # Creating the dictionary with the details
        comment_details = {
            'heading': heading,
            'user': user,
            'main_comment': comment,
            'date_and_time': date_and_time,
            'links_in_main': href_in_main,
            'user_comment': user_comments
        }

        # Output the dictionary
        # print(comment_details)

        return comment_details

    def kaggle_scrape_all_discussions(self, competition_slug: str, **kwargs) -> None:
        '''
        The function scrapes all the discussion for a give competition slug
        '''
        # Sign in and go to the discussion page
        opts = FirefoxOptions()
        opts.headless = False
        driver = webdriver.Firefox(options=opts)
        self.sign_in_competition(
            driver=driver, competition_slug=competition_slug, user_details=self.user_details,
            page="discussion?sort=votes"
        )

        print("on the discussion page")

        # list of the discussions on the fist page sorted according to the most votes
        # ignore the pinned discussions
        list_items = driver.find_elements(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[2]/div/div[4]/ul[1]/h3[2]/following-sibling::li[contains(@class, 'MuiListItem-root') and contains(@class, 'MuiListItem-gutters') and contains(@class, 'MuiListItem-divider')]")
        print("to next page")
        # go_to_next_page(driver=driver, competition_slug=competition_slug, curr_page=2)
        print("on the next page")
        # list_items.append(driver.find_elements(By.XPATH, "//li[contains(@class, 'MuiListItem-root') and contains(@class, 'MuiListItem-gutters') and contains(@class, 'MuiListItem-divider')]"))
        print(len(list_items))

        # for the competitions that do not have pinned discussions just scrap all the discussions from the top
        if len(list_items) == 0:
            list_items = driver.find_elements(
                By.XPATH, "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div/div/div[2]/div/div[4]/ul[1]/li"
            )

        # print(list_items[0])
        wait = WebDriverWait(driver, 10)

        list_items = list_items[:5]  # only selecting the first few discussions
        urls = []

        for i in range(len(list_items)):
            a_element = list_items[i].find_element(By.TAG_NAME, "a")
            href_value = a_element.get_attribute("href")
            urls.append(href_value)

        for i in range(len(urls)):
            # taking out the discussion
            try:
                driver.get(urls[i])
                discussion = self.scrape_discussion(urls[i], driver)
                print("on the discussion page")

                self.__class__.save_discussion_to_json(discussion,
                                                       f"{self.raw_data_dir}/discussions/{competition_slug}/discussion{i}.json")

                print("driver.current_url", driver.current_url)
                driver.back()
                # _sign_in(driver=driver, competition_slug=competition_slug, user_details= user_details)
                print("driver.current", driver.current_url)


            except Exception as e:
                print(f"Error clicking on item: {e}")

        driver.quit()

    @staticmethod
    def df_formated_head_view(input_df_path: str, n_rows: int = 5, max_columns: int = 100, width: int = 100) -> str:
        # avoid truncating output of df.head()
        set_pd_options(max_rows=None, max_columns=max_columns, float_format="{:20,.2f}".format, width=width)

        if ".csv" in input_df_path:
            read_func = pd.read_csv
        elif ".tsv" in input_df_path:
            read_func = partial(pd.read_csv, sep='\t')
        elif ".json" in input_df_path:
            read_func = pd.read_json
        else:
            read_func = None

        if read_func is None:
            raise ValueError(f"{input_df_path} is not a CSV or TSV or JSON file.")

        raw_table_view = ""
        try:
            table = next(read_func(input_df_path, chunksize=n_rows))
            _table_view = f"{os.path.join(input_df_path)}\n" + str(table.head()) + "\n\n"
            raw_table_view += _table_view
        except ValueError:
            print(f"Skipping view of {input_df_path} as it it probably not a table.", flush=True)

        # reset pandas options
        reset_pd_options()
        return raw_table_view

    @staticmethod
    def get_raw_table_view(
            download_dir: str,
            raw_table_view_path: str,
            detected_file: str | None,
            fetched_path: str,
            max_columns: int = 100,
            width: int = 100
    ) -> str:
        """
        Args:
            download_dir: (str) the path to the directory where data is downloaded
            max_columns: (int) the max number of columns allowed to be displayed when displaying the dataframe
            width: (int) the max number of characters allowed ber column when displaying the dataframe
        """
        if os.path.exists(raw_table_view_path):
            with open(raw_table_view_path, "r") as f:
                raw_table_view = f.read()
            return raw_table_view

        files = os.listdir(download_dir)
        excluded_files = {detected_file, os.path.basename(fetched_path)}
        files = [f for f in files if f not in excluded_files and f[-4:] == ".csv"]
        raw_table_view = ""
        for file in files:
            _table_view = FetchTool.df_formated_head_view(
                input_df_path=os.path.join(download_dir, file), max_columns=max_columns, width=width,
            )
            raw_table_view += _table_view
        if len(raw_table_view) > 0:
            with open(raw_table_view_path, "w") as f:
                f.write(raw_table_view)

        return raw_table_view

    @staticmethod
    def get_raw_table_info(
            download_dir: str,
            raw_table_info_path: str,
            detected_file: str | None,
            fetched_path: str
    ) -> str:
        """
        Args:
            download_dir: directory where all raw data is downloaded
            raw_table_info_path: path to file where to save raw table information
            detected_file: name of the sample submission file or detected equivalent
            fetched_path: path to the json file containing the flag saying if fetching is already done
        """
        if os.path.exists(raw_table_info_path):
            with open(raw_table_info_path, "r") as f:
                raw_table_info = f.read()

            return raw_table_info

        files = os.listdir(download_dir)
        excluded_files = {detected_file, os.path.basename(fetched_path)}
        files = [f for f in files if f not in excluded_files]
        raw_table_info = ""
        for file in files:
            if ".csv" in file:
                table = pd.read_csv(os.path.join(download_dir, file))
            elif ".tsv" in file:
                table = pd.read_csv(os.path.join(download_dir, file), sep='\t')
            elif ".json" in file:
                try:
                    table = pd.read_json(os.path.join(download_dir, file))
                except ValueError:
                    print(f"Skipping view of {os.path.join(download_dir, file)}", flush=True)
                    continue
            else:
                table = None

            if table is not None:
                buffer = io.StringIO()
                table.info(buf=buffer)
                s = buffer.getvalue()
                _table_info = f"{os.path.join(download_dir, file)}\n" + s
                for c in table.columns[1:]:
                    if (table[c].dtype == 'object' and isinstance(table[c].iloc[0], str)
                            and len(table[c].unique().tolist()) < 200):
                        _table_info += f"\n- column {c} contains strings with values in {table[c].unique().tolist()}"
                raw_table_info += "\n\n" + _table_info
        with open(raw_table_info_path, "w") as f:
            f.write(raw_table_info)

        return raw_table_info

    @staticmethod
    def tree(
            directory: str,
            padding: str = ' ',
            filelimit: int = 100_000,
            is_subdir: bool = False,
            files_to_ignore: list[str] = None,
            folder_limit: int = 5  # Limit for folders at each level
    ) -> str:
        """
        Prints the tree structure for the path specified.
        - directory: Root directory path to display the tree.
        - padding: Padding for the subdirectories.
        - filelimit: Limit for the number of files to display.
        - is_subdir: Internal flag to track if it's a subdirectory.
        - files_to_ignore: List of files to ignore when building the tree.
        - folder_limit: Maximum number of folders to display at each level.
        """
        end_line = '\n'
        tree_str = padding[:-1] + '+-' + os.path.basename(os.path.abspath(directory)) + '/' + end_line
        padding = padding + ' '

        sub_folders = []
        other_files = []
        with os.scandir(directory) as it:
            for entry in tqdm(it, desc="Scanning directory"):
                if entry.name.startswith('.') or (files_to_ignore and entry.name in files_to_ignore):
                    continue
                if entry.is_dir():
                    sub_folders.append(entry.name)
                else:
                    other_files.append(entry.name)

        # Apply the folder limit and append <additional folders> if limit is exceeded
        if len(sub_folders) > folder_limit:
            sub_folders = sub_folders[:folder_limit]
            sub_folders.append('... <additional folders>')

        files = sub_folders + other_files

        limit = int(filelimit)
        count = 0
        for file in files:
            count += 1
            path = os.path.join(directory, file)
            if os.path.isdir(path) and file != '... <additional folders>':
                tree_str += padding + '|' + end_line
                if count == len(files):
                    tree_str += FetchTool.tree(
                        directory=path, padding=padding + ' ', filelimit=filelimit,
                        is_subdir=True, files_to_ignore=files_to_ignore, folder_limit=folder_limit
                    )
                else:
                    tree_str += FetchTool.tree(
                        directory=path, padding=padding + '|', filelimit=filelimit,
                        is_subdir=True, files_to_ignore=files_to_ignore, folder_limit=folder_limit
                    )
            else:
                if file == '... <additional folders>':
                    tree_str += padding + '|' + end_line if not is_subdir else ''
                    tree_str += padding + '+-' + file + end_line
                elif limit == 100_000:
                    tree_str += padding + '|' + end_line if not is_subdir else ''
                    tree_str += padding + '+-' + file + end_line
                elif limit == 0:
                    tree_str += padding + '|' + end_line if not is_subdir else ''
                    tree_str += padding + '+-' + '... <additional files>' + end_line
                    limit -= 1
                elif limit <= 0:
                    break
                else:
                    tree_str += padding + '|' + end_line if not is_subdir else ''
                    tree_str += padding + '+-' + file + end_line
                    limit -= 1
        return tree_str

    @staticmethod
    def get_raw_data_view(download_dir: str, raw_data_view_path: str, detected_file: str | None) -> str:
        """
        Args:
            download_dir: directory where all raw data is downloaded
            raw_data_view_path: path to file where to save raw data view
            detected_file: name of the sample submission file or detected equivalent
        """
        if os.path.exists(raw_data_view_path):
            with open(raw_data_view_path, "r") as f:
                raw_data_view = f.read()
            if len(raw_data_view) > 0:
                return raw_data_view
        # ignore scrapped files and other added ones
        files_to_ignore = [
            'fetched_raw_data_done.json', 'tree.txt', 'raw_data_description.txt', 'raw_data_view.txt',
            'raw_metric_description.txt', 'raw_table_view.txt', 'raw_table_info.txt', 'raw_task_description.txt',
            'recursive_chmod_done.txt', 'description.md'
        ]
        # ignore sample sample_submission
        if detected_file is not None:
            files_to_ignore.append(detected_file.split('/')[-1])
        for file in os.listdir(download_dir):
            if file.endswith(".zip") or file.endswith(".tar.gz") or file.endswith(".tgz") or file.endswith(".7z"):
                files_to_ignore.append(file)
        tree_structure = FetchTool.tree(directory=download_dir, filelimit=3, files_to_ignore=files_to_ignore)
        tree_structure = str(Path(download_dir).parent) + "\n" + tree_structure

        with open(raw_data_view_path, 'w') as f:
            f.write(tree_structure)

        return tree_structure

    @staticmethod
    def get_raw_data_view_old(download_dir: str, raw_data_view_path: str, detected_file: str | None) -> str:
        """
        Args:
            download_dir: directory where all raw data is downloaded
            raw_data_view_path: path to file where to save raw data view
            detected_file: name of the sample submission file or detected equivalent
        """
        if os.path.exists(raw_data_view_path):
            with open(raw_data_view_path, "r") as f:
                raw_data_view = f.read()
            if len(raw_data_view) > 0:
                return raw_data_view

        # create top-level shallow tree view
        detected_file_str = f'{detected_file}|' if detected_file is not None else ''
        top_level_tree_command = (f"tree   -I 'raw_*|{detected_file_str}fetched_raw_data_done.json|tree.txt'"
                                  f" -L 1 {download_dir}  -o {os.path.join(download_dir, 'tree.txt')}")
        r = sp.Popen([top_level_tree_command], stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        out, err = r.communicate()

        with open(os.path.join(download_dir, 'tree.txt'), 'r') as f:
            top_level_tree = f.read()
        os.remove(os.path.join(download_dir, 'tree.txt'))
        tree = top_level_tree

        # create sublevel deep tree view
        all_files = os.listdir(download_dir)
        all_files = [f for f in all_files if "raw_" not in f]  # do not add raw descriptions in the view
        folders = [f for f in all_files if "." not in f]
        for folder in folders:
            sub_level_tree_command = (f"tree -L 3 -I 'tree.txt' {os.path.join(download_dir, folder)} --filelimit=20"
                                      f" | shuf -n 10 -o {os.path.join(download_dir, 'tree.txt')}")
            r = sp.Popen([sub_level_tree_command], stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
            out, err = r.communicate()

            with open(os.path.join(download_dir, 'tree.txt'), 'r') as f:
                sub_level_tree_lines = f.readlines()
            os.remove(os.path.join(download_dir, 'tree.txt'))
            sub_level_tree = f"{os.path.join(download_dir, folder)}\n"
            for line in sub_level_tree_lines:
                sub_level_tree += f"\t{line}"

            tree += "\n\n" + sub_level_tree + "\n\t..."

        with open(raw_data_view_path, 'w') as f:
            f.write(tree)

        return tree
