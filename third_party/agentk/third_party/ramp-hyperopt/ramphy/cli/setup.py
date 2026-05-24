import json
import click
from pathlib import Path
import ramphy as rh
import ramphy.ramp_setup as rs
rh.actions.EXECUTE_PLAN = True
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to set up.",
)
@click.option(
    "--setup-root",
    default="../ramp-setup-kits",
    show_default=True,
    help="The root folder where the kits with original metadata and "
    "train/test files are",
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
def main(
    ramp_kit,
    setup_root,
    kit_root,
    version,
    number,
):
    rs.scripts.setup.setup(
        ramp_kit = ramp_kit,
        setup_root = setup_root,
        kit_root = kit_root,
        version = version,
        number = number,
    )    

def start():
    main()

if __name__ == "__main__":
    start()

