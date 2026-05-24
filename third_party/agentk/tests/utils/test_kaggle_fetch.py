import argparse
import os
import shutil
import subprocess
import tempfile

from pyrootutils import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from agent.tools.fetch_tool import FetchTool


def check_network() -> bool:
    """
    Check internet connection and Fix Firefox webdriver stuck issue
    Returns:
        True if internet connection exists else False
    """
    try:
        subprocess.run(
            ["wget", "-q", "--timeout=2", "https://www.example.com", "-O", "/dev/null"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        return False


def test_kaggle_fetch(task_id: str, keep: bool, dir_name: str | None = None) -> bool:
    """
    Test fetch tool can access kaggle portal
    Args:
        task_id: Kaggle competition name
        keep: Whether to keep fetched files or not
        dir_name: Directory to save fetched files

    Returns:
        Status of Kaggle fetch - True if success else False
    """

    if dir_name is None:
        # create temp directory
        with tempfile.TemporaryDirectory() as tmp_dirname:
            raw_data_dir = tmp_dirname
    else:
        raw_data_dir = dir_name

    workspace_path = os.path.join(raw_data_dir, 'workspace')
    competition_dir = os.path.join(raw_data_dir, task_id)
    shutil.rmtree(competition_dir, ignore_errors=True)  # Delete folder if exists

    fetch_tool = FetchTool(
        task_url=f"https://www.kaggle.com/competitions/{task_id}",
        user_details="./third_party/data_preprocessing/kaggle_login_details.json",
        is_local_task=False,
        workspace_path=workspace_path,
        raw_data_dir=raw_data_dir,
    )

    print(f"Kaggle URL : {fetch_tool.task_url}")

    fetch_tool.join_competition(url=fetch_tool.task_url)
    fetch_tool.get_dataset(name=task_id, download_dir=fetch_tool.raw_data_dir)
    if os.path.exists(competition_dir):
        if not keep:
            shutil.rmtree(raw_data_dir, ignore_errors=True)  # Delete folder if exists
        else:
            print(f"Data saved to {competition_dir}")
        return True

    return False


if __name__ == "__main__":
    """
    Fetch Kaggle competition data.

    Example:
        python tests/utils/test_kaggle_fetch.py
        python tests/utils/test_kaggle_fetch.py --task_id sp-society-camera-model-identification
    """
    parser = argparse.ArgumentParser(description="Fetch Kaggle competition data.")
    parser.add_argument("--task_id", type=str, default="playground-series-s3e5", help="Kaggle competition task ID")
    parser.add_argument("--dir_name", type=str, required=False, default=None, help="Directory to download data to")
    parser.add_argument('--keep', action='store_true', help="Keep downloaded data")
    args = parser.parse_args()

    if check_network():
        print("✅ Internet test Successful!")
        status = test_kaggle_fetch(args.task_id, keep=args.keep, dir_name=args.dir_name)
        if status:
            print("✅ Kaggle fetch Successful!")
        else:
            print("❌ Kaggle fetch Failed!")
    else:
        print("❌ Aborting fetch due to no network.")
