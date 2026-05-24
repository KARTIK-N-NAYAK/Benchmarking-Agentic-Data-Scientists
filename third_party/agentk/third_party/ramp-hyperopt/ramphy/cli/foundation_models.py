import ramphy.ramp_setup as rs
import click
import click_config_file

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to hyperopt.",
)
@click.option(
    "--kit-root",
    default="./",
    show_default=True,
    help="The root folder where the kits are",
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
    "--n-folds-hyperopt",
    default=3,
    show_default=True,
    help="The number of folds used in hyperopt. Needed so hyperopt race can resume.",
)
@click.option(
    "--n-folds-final-blend",
    default=30,
    show_default=True,
    help="The number of folds used in the final blend.",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click.option(
    "--base-predictors",
    multiple=True,
    default=["lgbm", "xgboost", "catboost"],
    help="A list of base predictors.",
)
@click.option(
    "--data-preprocessors",
    multiple=True,
    default=["drop_id", "drop_columns", "base_columnwise", "col_in_train_only", "rm_constant_col"],
    help="A list of data_preprocessors to use. base_columnwise are the base col encoding and inputing",
)
@click.option(
    "--foundation-predictors-dir",
    default="best_predictor_arms/hand_selection",
    help="Directory where the foundation models are stored",
    show_default=True,
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    kit_root,
    version,
    number,
    n_folds_hyperopt,
    n_folds_final_blend,
    first_fold_idx,
    base_predictors,
    data_preprocessors,
    foundation_predictors_dir,
):
    print(version, number)
    rs.foundation.foundation_models(
        ramp_kit=ramp_kit,
        kit_root=kit_root,
        version=version,
        number=number,
        n_folds_hyperopt=n_folds_hyperopt,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        base_predictors=list(base_predictors),
        data_preprocessors=list(data_preprocessors),
        foundation_predictors_dir=foundation_predictors_dir,
    )


def start():
    main()


if __name__ == "__main__":
    start()
