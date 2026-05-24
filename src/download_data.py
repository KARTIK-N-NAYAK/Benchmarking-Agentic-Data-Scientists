from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import zipfile
from requests.exceptions import HTTPError

comp_list = Path("config/agentk.txt")
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# Load list
competitions = [
    c.strip() for c in comp_list.read_text().splitlines() if c.strip()
]

competitions = competitions

api = KaggleApi()
api.authenticate()

for comp in competitions:
    print(f"⏬ downloading {comp} ...")
    try:
        api.competition_download_files(
            comp,
            path=str(out_dir/"zip"),
        )
    except HTTPError as e:
        if e.response.status_code == 403:
            print(f"⚠️  SKIPPING {comp}: You have not accepted the rules for this competition yet.")
            print(f"   Go to https://www.kaggle.com/c/{comp}/rules to accept them.")

        raise e
    
    zip_path = out_dir / (comp + ".zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir/"extracted"/comp)
