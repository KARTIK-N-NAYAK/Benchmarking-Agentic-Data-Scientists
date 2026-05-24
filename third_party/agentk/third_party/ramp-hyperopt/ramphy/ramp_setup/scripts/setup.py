import json
from pathlib import Path
import ramphy.ramp_setup as rs
import ramphy as rh


def setup(
    ramp_kit: str,
    setup_root: str,
    kit_root: str,
    version: str,
    number: str | int,
):
    """Sets up a ramp kit from a ramp setup kit, and submits and trains starting kit.

    Takes input from <setup_dir>/<ramp_kit>, and produces the kit in
    <ramp_kit>_v<version>_n<number>. <setup_dir>/<ramp_kit> can be produced
    semi-manually (e.g. calling "kaggle competitions download" and manually editing
    basic metadata, examples are in the test folder), or automatically using an LLM.

    Parameters
    ----------
    ramp_kit : str
        The name of the ramp-kit.
    setup_root : str
        The folder where the ingredients are found is <setup_root>/<ramp_kit>.
        Typically contains test, train, and metadata.
    kit_root : str
        The folder where kits are found.
    version : str
        The version tag of ramphy and rampwf.
    number : str | int
        A suffix, typically a number, differentiating between different runs
        of the same kit.
    """
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{kit_root}/{ramp_kit}_{kit_suffix}"

    rs.scripts.tabular.tabular_setup(
        download_dir=f"{setup_root}/{ramp_kit}",
        ramp_kit_dir=ramp_kit_dir,
    )

    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))

    if "regression" in metadata["prediction_type"]:
        rs.scripts.tabular.tabular_regression_ordered_submit(
            ramp_kit_dir=ramp_kit_dir,
            submission="starting_kit",
            regressor="lgbm",
        )
    elif "classification" in metadata["prediction_type"]:
        rs.scripts.tabular.tabular_classification_ordered_submit(
            ramp_kit_dir = ramp_kit_dir,
            submission = 'starting_kit',
            classifier = 'lgbm',
        )

    rh.actions.train(
        ramp_kit_dir = ramp_kit_dir,
        submission = 'starting_kit',
#        fold_idxs = range(900, 903),
        fold_idxs = range(3),
        force_retrain = True,
    )
