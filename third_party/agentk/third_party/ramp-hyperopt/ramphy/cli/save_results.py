import click
import click_config_file
import pandas as pd
import ramphy as rh
import ramphy.ramp_setup as rs
from pathlib import Path
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

def get_validation_results(all_actions, first_fold_idx, n_folds_final_blend, blend_type):
    stop_fold_idx = first_fold_idx + n_folds_final_blend
    if blend_type == "last_blend":
        blend_actions = [ra for ra in all_actions if ra.name == "blend" and
                         ra.kwargs["fold_idxs"] == range(first_fold_idx, stop_fold_idx)]
        final_blend_actions = [ra for ra in all_actions if ra.name == "final_blend_then_bag"]
    elif blend_type == "bag_then_blend":
        blend_actions = [ra for ra in all_actions if ra.name == "bag_then_blend" and
                         ra.kwargs["fold_idxs"] == range(first_fold_idx, stop_fold_idx)]
        final_blend_actions = [ra for ra in all_actions if ra.name == "final_bag_then_blend"]
    else:
        raise ValueError(f"Unknown blend_type {blend_type}")
    if len(blend_actions) != len(final_blend_actions):
        raise ValueError("The rest of the script relies on calling blend only from "
                         "final_blend, so the number of actions should be the same.\n"
                         f"Instead len(blend_actions) = {len(blend_actions)} "
                         f"!= {len(final_blend_actions)} = len(final_blend_actions)"
                         f"blend_type = {blend_type}")
    results_df = pd.DataFrame()
    results_df["round_time"] = [ra.kwargs["round_datetime"] for ra in final_blend_actions]
    results_df["valid_score"] = [ra.blended_score for ra in blend_actions]
    results_df["contributivities"] = [ra.contributivities for ra in blend_actions]
    results_df["runtime"] =  [ra.runtime.total_seconds() for ra in blend_actions]
    results_df = results_df.sort_values("round_time")
    results_df["round_idx"] = [ra.kwargs["n_rounds"]  if "n_rounds" in ra.kwargs else -1 for ra in final_blend_actions]
    results_df = results_df[results_df["round_idx"] > 0]
    results_df = results_df.sort_values("round_idx")
    return results_df


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    default=None,
    help="The kit to update.",
)
@click.option(
    "--version",
    default=None,
    help="The program version",
)
@click.option(
    "--number",
    default=None,
    help="The program number (repeated within version)",
)
@click.option(
    "--n-folds-hyperopt",
    default=3,
    show_default=True,
    help="The number of folds used in hyperopt.",
)
@click.option(
    "--n-folds-final-blend",
    default=30,
    show_default=True,
    help="The number of folds to bag",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    version,
    number,
    n_folds_hyperopt,
    n_folds_final_blend,
    first_fold_idx,
):
    
    round_idxs = [20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

    ramp_kit_dir = f"{ramp_kit}_v{version}_n{number}"    
    all_actions = rh.actions.get_all_actions(ramp_kit_dir)
    blend_actions = [ra for ra in all_actions if ra.name == "blend" and
                     ra.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt)]
    # adding possible final blend, not in the scheduled round_idx's
    if len(blend_actions) not in round_idxs:
        round_idxs = round_idxs + [len(blend_actions)]
        round_idxs = [idx for idx in round_idxs if idx <= len(blend_actions)]

    results_path = Path(ramp_kit_dir) / "results"
    results_path.mkdir(exist_ok=True)

    results_df = get_validation_results(
        all_actions, first_fold_idx, n_folds_final_blend, blend_type="last_blend")
    results_df = results_df[results_df["round_idx"].isin(round_idxs)]
    results_df = results_df.drop_duplicates(subset='round_idx', keep='last')
    results_df["ramp_kit"] = ramp_kit
    results_df["version"] = version
    results_df["number"] = number
    results_df.to_csv(results_path / "results_blended_then_bagged.csv", index=False)
    print(f"saved blended_then_bagged round idxs: {list(results_df['round_idx'])}")

    results_df = get_validation_results(
        all_actions, first_fold_idx, n_folds_final_blend, blend_type="bag_then_blend")
    results_df = results_df[results_df["round_idx"].isin(round_idxs)]
    results_df = results_df.drop_duplicates(subset='round_idx', keep='last')
    results_df["ramp_kit"] = ramp_kit
    results_df["version"] = version
    results_df["number"] = number
    results_df.to_csv(results_path / "results_bagged_then_blended.csv", index=False)
    print(f"saved bagged_then_blended round idxs: {list(results_df['round_idx'])}")

    for ra in all_actions:
        if ra.name == "blend":
            if ra.kwargs["fold_idxs"] == range(0, 3):
                ra.name = "blend_hyperopt"
            else:
                ra.name = "blend_then_bag"
    all_actions_df = pd.DataFrame([{"name": ra.name, "runtime": ra.runtime} for ra in all_actions])
    runtimes = all_actions_df.groupby('name')['runtime'].sum()
    runtimes.to_csv(results_path / "runtimes.csv")

def start():
    main()

if __name__ == "__main__":
    start()

