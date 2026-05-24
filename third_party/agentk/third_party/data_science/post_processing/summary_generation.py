import glob
import os
from pathlib import Path


def generate_cot_summary(competition_workspace: Path | str) -> str | None:
    """
    Aggregates the summaries from each submission in the competition submission folder
    """
    files = []
    scan_dir = os.path.join(competition_workspace, "./**/*")
    for path in glob.iglob(scan_dir, recursive=True):
        if path.endswith("summary.txt"):
            files.append(Path(path))

    if len(files) > 0:
        print(f"Found {len(files)} summaries in {competition_workspace}")
    else:
        return None

    # concatenate summaries
    competition_summaries = []
    for i, file in enumerate(files):
        # open file, read it and append its text to a string
        with open(file, "r") as f:
            summary = f.read()
            competition_summaries.append(
                f"### Submission {i} summary and metric value on a validation set:\n" + summary
            )

    competition_summary = "\n-----\n".join(competition_summaries)
    return competition_summary


def generate_and_save_cot_summaries(competition_workspaces: list[Path], cot_summaries_dir: Path) -> None:
    for competition_workspace in competition_workspaces:
        attempt = competition_workspace.parent.parent.parent.name
        competition_slug = competition_workspace.parent.parent.name
        competition_summary = generate_cot_summary(competition_workspace)
        if competition_summary is not None:
            os.makedirs(cot_summaries_dir / attempt / competition_slug, exist_ok=True)
            with open(cot_summaries_dir / attempt / competition_slug / "summary.txt", "w") as f:
                f.write(competition_summary)
            print(f"Wrote summary.txt at {cot_summaries_dir / attempt / competition_slug}")


def generate_summaries_of_all_competitions(workspace: Path, cot_summaries_dir: Path) -> None:
    competition_workspaces = [
        Path(p) for p in glob.glob(str(workspace / "attempt_*-2days_limit/*/seed_*/main_pipeline/"))
    ]
    generate_and_save_cot_summaries(
        competition_workspaces=sorted(competition_workspaces), cot_summaries_dir=cot_summaries_dir,
    )
