import argparse
import json
from datetime import timezone, datetime
from functools import cache
from pathlib import Path

import pandas as pd


def get_comp_info_path(ds_root: Path, community: bool) -> Path:
    competition_info_path = ds_root / "data" / "competition_info.csv"
    community_competition_info_path = competition_info_path.with_stem("community_" + competition_info_path.stem)
    if community:
        return community_competition_info_path
    return competition_info_path


"""Instructions for collecting .har file
1. Navigate to https://www.kaggle.com/competitions/ in Google Chrome
2. Select desired filter (e.g. "All Competitions" or "Community + Open + Sort by num teams")
3. Press F12 to open dev console, navigate to network tab and filter for "ListCompetitions"
4. Go to Network tab and tick "Preserve log"
5. Go to console and use the following code to repeatedly click through pages
```javascript
interval = setInterval(function(){
  document.querySelector('[aria-label="Go to next page"]').click()
},1000);
```
6. When you have enough, cancel the loop with `clearInterval(interval)`
7. Download .har file using button near the top of the Network tab
"""


def _extract_competition_info(har_path: Path):
    """Extracts kaggle competition data from a chrome network log '.har' file"""
    with har_path.open("r") as fp:
        network_log = json.load(fp)["log"]

    assert "https://www.kaggle.com/competitions" in (p["title"].split("?")[0].strip("/") for p in network_log["pages"])

    competitions = []
    for entry in network_log["entries"]:
        content = entry["response"]["content"]
        if content["mimeType"] != "application/json":
            print("Content is not a json:", content)
            continue

        response_json = json.loads(content["text"])
        response_competitions = response_json.get("competitions", [])
        competitions.extend(response_competitions)

    return competitions


COLUMNS_TO_KEEP = [
    "competitionHostSegmentId",
    "competitionType",
    "deadline",
    "id",
    "medalsAllowed",
    "title",
    "totalTeams",
    "forumId"
]

# Manually collected info
DATE_COLS = ["deadline"]

NULLABLE_INT_COLS = ["totalTeams"]

COMPETITION_TYPE_ID_TO_LABEL = {
    1: "Featured",
    2: "Research",
    3: "Recruitment",
    5: "Getting Started",
    6: "Masters",
    8: "Playground",
    10: "Community",
    11: "Analytics",
}


def _process_har(har_path: Path) -> pd.DataFrame:
    competitions = _extract_competition_info(har_path)

    df = pd.json_normalize(competitions, max_level=1).set_index("competitionName")
    df.drop_duplicates("id", inplace=True)

    # Translate competitionHostSegment to label (e.g. "Featured", "Playground")
    df["competitionType"] = df["competitionHostSegmentId"].map(COMPETITION_TYPE_ID_TO_LABEL)

    return df[COLUMNS_TO_KEEP]


def medal_awarding_competitions(comp_df) -> pd.DataFrame:
    df = comp_df
    # Unfinished competitions have not given medals
    df = df[df["deadline"] < datetime(2025, 7, 1, 0, 0, 0, microsecond=0, tzinfo=timezone.utc)]

    # Competitions with no participants are invalid
    df = df.dropna(subset="totalTeams")

    # Filter for Featured / Research competitions
    df = df[df["medalsAllowed"] == True]

    return df


@cache
def get_competition_info(ds_root: Path, include_community: bool = False) -> pd.DataFrame:
    read_csv_kwargs = dict(
        index_col=0, parse_dates=DATE_COLS, date_format="ISO8601", dtype={col: "Int64" for col in NULLABLE_INT_COLS}
    )
    try:
        comp_df = pd.read_csv(get_comp_info_path(ds_root=ds_root, community=False), **read_csv_kwargs)

        if include_community:
            community_comp_df = pd.read_csv(get_comp_info_path(ds_root=ds_root, community=True), **read_csv_kwargs)
            comp_df = pd.concat([comp_df, community_comp_df], axis=0)
            comp_df.drop_duplicates("id", inplace=True)

    except FileNotFoundError:
        print("competition_info.csv not found, please see instructions for processing .har file in competition_info.py")
        raise

    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a HAR file.")
    parser.add_argument("--community", action="store_true", help="Process a community HAR file.")
    parser.add_argument("--har_path", type=str, required=True, help="Path to the HAR file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    output_path = args.output_path

    df_ = _process_har(Path(args.har_path))
    df_.to_csv(output_path)

    print(df_)
