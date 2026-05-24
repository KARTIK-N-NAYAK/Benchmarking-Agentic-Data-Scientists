import argparse
import logging
import pandas as pd
import os
from autogluon.tabular import TabularPredictor


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def main(args):
    logging.info("Loading training data...")
    train_data = pd.read_csv(args.train_path)

    logging.info(f"Training shape: {train_data.shape}")

    # Unique model directory per job
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    model_path = f"{args.model_dir}_{job_id}"

    logging.info(f"Using model directory: {model_path}")

    logging.info("Starting AutoGluon training...")
    predictor = TabularPredictor(
        label=args.target,
        path=model_path,
        eval_metric="roc_auc"
    ).fit(
        train_data,
        time_limit=args.time_limit,
        presets="best_quality",
        num_gpus=args.num_gpus,
    )

    logging.info("Training complete.")

    logging.info("Loading test data...")
    test_data = pd.read_csv(args.test_path)

    logging.info(f"Test shape: {test_data.shape}")

    logging.info("Generating predictions...")
    predictions = predictor.predict_proba(test_data)

    # Take probability of positive class
    predictions = predictions.iloc[:, 1]

    logging.info(f"Saving predictions to: {args.output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    output_df = pd.DataFrame({
        args.output_column_name: predictions
    })

    output_df.to_csv(args.output_path, index=False)

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    parser.add_argument("--output_column_name", type=str, default="prediction")
    parser.add_argument("--time_limit", type=int, default=1800)
    parser.add_argument("--presets", type=str, default="best_quality")
    parser.add_argument("--model_dir", type=str, default="autogluon_model")
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--log_path", type=str, default="training.log")

    args = parser.parse_args()

    setup_logging(args.log_path)
    main(args)