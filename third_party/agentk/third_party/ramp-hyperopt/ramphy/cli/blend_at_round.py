import click
import click_config_file
import ramphy.ramp_setup as rs
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

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
    "--n-folds-final-blend",
    default=30,
    show_default=True,
    help="The number of folds to bag",
)
@click.option(
    "--n-folds-hyperopt",
    default=3,
    show_default=True,
    help="The number of folds used in hyperopt.",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click.option(
    "--round-idx",
    default=-1,
    show_default=True,
    help="The round at which we should blend.",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    version,
    number,
    n_folds_final_blend,
    n_folds_hyperopt,
    first_fold_idx,
    round_idx,
):
    rs.blend_at_round.blend_at_round(
        ramp_kit,
        version,
        number,
        n_folds_final_blend,
        n_folds_hyperopt,
        first_fold_idx,
        round_idx,
    )

def start():
    main()

if __name__ == "__main__":
    start()

