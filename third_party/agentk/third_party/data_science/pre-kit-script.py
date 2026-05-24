import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT_PROJECT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from agent.tasks.datascience_task.ramp_utils import prepare_for_ramp_setup

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_folder", help="root path containing the metadata folder and tables")
    parser.add_argument("--challenge_name", help="Name of the challenge")
    parser.add_argument("--output_path", help="Path where the json and csvs will be saved")
    parser.add_argument(
        "--post-setup", action='store_true', default=False,
        help="If active, do the ramp setup in the context of the post-benchmark. "
             "This means that some tasks are sub-sampled because their dataset is too large."
        )

    args = parser.parse_args()

    prepare_for_ramp_setup(
        info_path=args.root_folder,
        data_path=args.root_folder,
        challenge_name=args.challenge_name,
        output_path=args.output_path,
        post_setup=args.post_setup,
    )
