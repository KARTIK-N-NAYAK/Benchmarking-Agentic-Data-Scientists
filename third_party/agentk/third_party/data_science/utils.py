from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple
import glob

import numpy as np
import pandas as pd
import requests
import tqdm
import json
import os
import time
import subprocess

from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except OSError:
    KaggleApi = None

from ds_agent.utils_kaggle import check_is_lower_better


def get_agent_root_dir() -> Path:
    """Read and return the root path where repo code is located"""
    return Path(__file__).parent.parent.parent


def get_raw_data_root_dir() -> Path:
    """Read and return the root path where raw DS data are stored"""

    aux_path_to_raw_ds_data = f"{get_agent_root_dir()}/root_path_to_raw_ds_data.txt"
    if os.environ.get('ALT_RAW_DATA_ROOT', None):
        aux_path_to_raw_ds_data = os.environ['ALT_RAW_DATA_ROOT']
        print(f"Using raw data dir: {aux_path_to_raw_ds_data}", flush=True)
        return Path(aux_path_to_raw_ds_data)
    while not os.path.exists(aux_path_to_raw_ds_data):
        message = (f"\n/!\/!\/!\ PLEASE READ /!\/!\/!\ \n"
                   f"file {os.path.abspath(aux_path_to_raw_ds_data)} should contain"
                   f" the absolute path to the directory where the raw data"
                   f" used in datascience tasks will be saved.\n"
                   f"Enter here the root path (not subtask specific) to the "
                   f"folder containing all data-science data\nPath to save raw data: ")
        path_to_raw_ds_data = ""
        while path_to_raw_ds_data == "":
            path_to_raw_ds_data = input(message)

        os.system(f'echo "{path_to_raw_ds_data}" > {aux_path_to_raw_ds_data}')
    with open(aux_path_to_raw_ds_data) as f:
        path_to_raw_ds_data = f.readline().replace("\n", "")
    return Path(path_to_raw_ds_data)


# def get_path_to_ds_python() -> str:
#     """Get path to the python executable to use for data-science"""
#     path_to_python = str(Path(__file__).parent.parent / "agent_k_python_path.txt")
#     if not os.path.exists(path_to_python):
#         import sys
#         python_exe = sys.executable
#     else:
#         with open(path_to_python) as f:
#             python_exe = f.readline().replace("\n", "")
#     return python_exe

def get_path_to_ds_python() -> str:
    import sys
    return sys.executable

def download_file(url: str, destination: Path):
    """Downloads a file from the specified URL

    Saves the file to the provided destination path.

    Args:
        url (str): The URL to download the file from.
        destination (str): The local path to save the downloaded file.
    """
    try:
        response = requests.get(url, stream=True, verify=False)
        breakpoint()
        response.raise_for_status()  # raises an HTTPError for bad responses

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Download complete. File saved to:", destination)
    except requests.RequestException as e:
        print("Error downloading the file:", e)


def download_leaderboard(
        kaggle_api: KaggleApi,
        competition: str,
        zip_destination: str | Path,
        phase: str
):
    """Downloads leaderboard.

    This is not supported by the Kaggle API so we implement it ourselves as there is
    a link on the leaderboard webpage to download it.

    Args:
        kaggle_api (object): KaggleAPI instance
        competition (str): Identifier of the competition
        zip_destination (Path): where the zip file is
        phase (str): public or private
    """

    zip_destination_path = Path(zip_destination)
    zip_destination_ = str(zip_destination_path.absolute() / competition)

    os.makedirs(zip_destination_, exist_ok=True)
    result = kaggle_api.competitions_list(group="entered", search=competition)

    competition_url = f"https://www.kaggle.com/competitions/{competition}"
    for c in result:
        url = getattr(c, "url", f"https://www.kaggle.com/competitions/{c.ref}")
        if url == competition_url:
            break
    num_id = getattr(c, "id", f"https://www.kaggle.com/competitions/{c.ref}")

    try:
        subprocess.run(
            ["wget", "-q", "--timeout=2", "https://www.example.com", "-O", "/dev/null"],
            check=True
        )
    except subprocess.CalledProcessError as _:
        return

    user_details: str = "./third_party/data_preprocessing/kaggle_login_details.json"
    kaggle_login = json.load(open(user_details))
    kaggle_login_mail = kaggle_login.get("email", None)
    kaggle_login_pwd = kaggle_login.get("pwd", None)
    if kaggle_login_mail is None:
        kaggle_login_mail = kaggle_login.get("login_email", None)
    if kaggle_login_pwd is None:
        kaggle_login_pwd = kaggle_login.get("login_pwd", None)

    os.environ["MOZ_HEADLESS"] = "1"

    # Firefox profile for auto-download
    fp = webdriver.FirefoxProfile()
    fp.set_preference("browser.download.folderList", 2)
    print(f"The zip destination is set to {zip_destination_}")
    fp.set_preference("browser.download.dir", zip_destination_)
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip")
    fp.set_preference("pdfjs.disabled", True)

    # Firefox options
    opts = FirefoxOptions()

    opts.profile = fp
    opts.headless = False

    # Start driver
    with webdriver.Firefox(options=opts) as driver:
        # Login page
        driver.get("https://www.kaggle.com/account/login?phase=emailSignIn&returnUrl=%2F")
        time.sleep(2)

        # Enter email and password
        driver.find_element(By.NAME, "email").send_keys(kaggle_login_mail)
        driver.find_element(By.NAME, "password").send_keys(kaggle_login_pwd + Keys.RETURN)

        # Wait for login to complete
        time.sleep(5)

        # Go to download page
        download_url = f"https://www.kaggle.com/competitions/{num_id}/leaderboard/download/{phase}"
        driver.execute_script(f"window.location.href='{download_url}'")
        # Wait for download to finish
        time.sleep(10)
    pattern = os.path.join(zip_destination_, f"{competition}-{phase}leaderboard*.zip")

    zip_file = glob.glob(pattern)
    with zipfile.ZipFile(str(zip_file[-1]), "r") as zip_ref:
        zip_ref.extractall(zip_destination_)


def recursive_chmod(path: str, mode: int) -> None:
    """Change permission of all files in path"""
    if os.path.exists(os.path.join(path, 'recursive_chmod_done.txt')):
        print("Recursive chmod already done, skipping.")
    else:
        for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(path)):
            os.chmod(path=dirpath, mode=mode)
            for filename in filenames:
                os.chmod(path=os.path.join(dirpath, filename), mode=mode)
        with open(os.path.join(path, 'recursive_chmod_done.txt'), 'w') as f:
            f.write(f"Recursive chmod done for all files in {path}")
