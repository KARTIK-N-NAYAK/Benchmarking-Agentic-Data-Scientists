import click
import pandas as pd
from pathlib import Path
import ramphy as rh
rh.actions.EXECUTE_PLAN = True
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to set up.",
)
@click.option(
    "--ramp-kits-dir",
    default=".",
    show_default=True,
    help="The root folder where the ramp kits are.",
)
@click.option(
    "--version",
    help="The program version",
)
@click.option(
    "--number",
    help="The program number (repeated within version)",
)
@click.option(
    "--ip",
    help="The IP adress of the machine",
)
def main(
    ramp_kit,
    ramp_kits_dir,
    version,
    number,
    ip,
):
    results_summary_df = pd.read_csv(Path(ramp_kits_dir) / "results_summary.csv")
    for col in results_summary_df.columns:
        if col[:14] == "contributivity" or col[:6] == "rounds":
            results_summary_df[col] = results_summary_df[col].fillna(0)
            results_summary_df[col] = results_summary_df[col].astype("int64")
        if col[:7] == "runtime":
            results_summary_df[col] = results_summary_df[col].astype("timedelta64[ns]")
    row = {
        "ramp_kit": ramp_kit,
        "version": version,
        "number": number,
        "server": ip,
        "run_finished": 0,
        "kaggle_finished": 0,
    }
    results_summary_df.loc[len(results_summary_df)] = row
    results_summary_df = results_summary_df.sort_values(["ramp_kit", "version", "number"])
    results_summary_df.to_csv("results_summary.csv", index=False)
    print(results_summary_df)

def start():
    main()

if __name__ == "__main__":
    start()

