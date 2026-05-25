import argparse
import subprocess
import shutil
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input data directory passed to mlzero (e.g. data/)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory created by mlzero (default: output)"
    )
    parser.add_argument(
        "-n", "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations to pass to mlzero (default: 10)"
    )
    args = parser.parse_args()

    # 1. Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build mlzero command
    cmd = [
        "mlzero",
        "-i", args.input_dir,
        "-o", str(output_dir),
        "-c", "third_party/ml_zero/src/autogluon/assistant/configs/mllab_test.yaml",
        "-n", str(args.max_iterations),
        "--initial-instruction",
        (
            "Train on 'data/train.csv' and predict labels for 'data/test.csv' and save them in 'submissions.csv', refer to 'data/sample-submission.csv' for format"
        )
    ]

    # 3. Run command
    subprocess.run(cmd, check=True)

    # # 4. Cleanup output directory
    # for item in output_dir.iterdir():
    #     if item.name == "results.csv":
    #         continue

    #     try:
    #         if item.is_symlink():
    #             # Remove symlink without following it
    #             item.unlink()
    #         elif item.is_file():
    #             item.unlink()
    #         elif item.is_dir():
    #             shutil.rmtree(item)
    #     except FileNotFoundError:
    #         # Symlink target disappeared between checks – safe to ignore
    #         pass      

    # 5. Rename results.csv → submissions.csv
    results_file = output_dir / "results.csv"
    submissions_file = output_dir / "submissions.csv"

    if not results_file.exists():
        logging.warning("Warning: results.csv not found in output directory. Skipping rename.")
    else:
        results_file.rename(submissions_file)
        logging.info(f"Renamed results.csv to submissions.csv")

if __name__ == "__main__":
    main()
