import argparse
from pathlib import Path

import pandas as pd

from ds_agent.competition_info import medal_awarding_competitions, get_competition_info


def latest_leaderboard_path(competition_name: str, leaderboard_dir: Path, try_public=False) -> Path | None:
    phases = ["private"]
    if try_public:
        phases.append("public")

    for phase in phases:
        leaderboard_csvs = list(leaderboard_dir.glob(f"{competition_name}-{phase}leaderboard-*.csv"))
        if leaderboard_csvs:
            break
    else:
        # The loop was not broken so leaderboard_csvs is empty
        return None

    return max(leaderboard_csvs)


def load_leaderboard(leaderboard_path: Path, split_team_members=True, drop_sample_submission=True):
    df = pd.read_csv(leaderboard_path, index_col=0)

    if drop_sample_submission:
        df.drop(index=0, errors="ignore", inplace=True)

    df["TeamMemberUserNames"] = df["TeamMemberUserNames"].str.split(",")
    df["Solo"] = df["TeamMemberUserNames"].str.len() == 1
    if split_team_members:
        df.reset_index(inplace=True)
        df = df.explode("TeamMemberUserNames").rename(columns={"TeamMemberUserNames": "User"})
        df.set_index("User", verify_integrity=True, inplace=True)

    return df


def missing_leaderboards(ds_root: Path, require_medal_awarding=True) -> list[str]:
    df = get_competition_info(ds_root=ds_root)
    if require_medal_awarding:
        df = medal_awarding_competitions(df)
    leaderboard_dir = ds_root / "leaderboards"
    return [c for c in df.index if latest_leaderboard_path(competition_name=c, leaderboard_dir=leaderboard_dir) is None]


if __name__ == "__main__":
    # Simple script to print the URLs of missing leaderboards for manual collection
    parser = argparse.ArgumentParser(description="Run setup and main pipeline.")
    parser.add_argument(
        "--ds_root", type=str, required=True, help='Path to the folder where data-science info will be stored'
    )
    args = parser.parse_args()
    ds_root_ = Path(args.ds_root)
    missing_comps = missing_leaderboards(require_medal_awarding=True, ds_root=ds_root_)

    for comp_id in missing_comps:
        print(f"https://www.kaggle.com/competitions/{comp_id}/leaderboard")

    if not missing_comps:
        print("All available leaderboards are present!")
