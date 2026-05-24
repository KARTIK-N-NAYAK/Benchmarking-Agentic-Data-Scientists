import os
import sys
import warnings

import pandas as pd
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from tabicl import TabICLClassifier
from tqdm import tqdm

from metadata import MetaData
from utils import get_data_paths, write_training_points, preprocess_dataset, prepare_test_set_chunks, create_submission, \
    prepare_dataset, normalize_predictions, get_dataset_path, check_necessary_files_exists, parse_args


def run_inference(competition: str, workspace_path: str, setup_path: str, context_length: int) -> None:
    meta_data = MetaData(dataset_path=setup_path)
    workspace_path = os.path.join(workspace_path, competition)
    prediction_type = meta_data.prediction_type
    target_cols = meta_data.target_cols
    id_field = meta_data.id_field

    assert len(target_cols) == 1, "More than one target columns"
    train_data_path, test_data_path, problem_py_path = get_data_paths(dataset_path=setup_path)

    print(competition)
    print(f"prediction_type : {prediction_type}")
    is_classification = prediction_type != 'regression'

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    all_classes = []
    if is_classification:
        all_classes = train_data[target_cols[0]].unique().tolist()

    write_training_points(output_path=workspace_path, num_training_points=train_data.shape[0])

    train_data = train_data.astype({col: float for col in train_data.select_dtypes(include='int').columns})
    test_data = test_data.astype({col: float for col in test_data.select_dtypes(include='int').columns})

    train_data, test_data = preprocess_dataset(
        train_data=train_data,
        test_data=test_data,
        is_classification=is_classification,
        id_field=id_field,
        target_cols=target_cols
    )

    test_samples = prepare_test_set_chunks(test_data, chunk_size=context_length)
    print(f"Number of test splits : {len(test_samples)}")
    all_predictions = []
    for test_sample in tqdm(test_samples, desc=f"Predicting", leave=False, position=2):
        x_train, y_train = prepare_dataset(
            train_data=train_data,
            id_field=id_field,
            target_cols=target_cols,
            test_sample=test_sample,
            context_length=context_length
        )
        test_sample = test_sample.drop(columns=target_cols + [id_field], errors='ignore')

        clf = TabICLClassifier(
            n_estimators=32,  # number of ensemble members
            norm_methods=["none", "power"],  # normalization methods to try
            feat_shuffle_method="latin",  # feature permutation strategy
            class_shift=True,  # whether to apply cyclic shifts to class labels
            outlier_threshold=4.0,  # z-score threshold for outlier detection and clipping
            softmax_temperature=0.9,  # controls prediction confidence
            average_logits=True,  # whether ensemble averaging is done on logits or probabilities
            use_hierarchical=True,  # enable hierarchical classification for datasets with many classes
            batch_size=8,  # process this many ensemble members together (reduce RAM usage)
            use_amp=True,  # use automatic mixed precision for faster inference
            model_path=None,  # where the model checkpoint is stored
            allow_auto_download=True,  # whether automatic download to the specified path is allowed
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",  # the version of pretrained checkpoint to use
            device=None,  # specify device for inference
            random_state=42,  # random seed for reproducibility
            n_jobs=None,  # number of threads to use for PyTorch
            verbose=False,  # print detailed information during inference
            inference_config=None,  # inference configuration for fine-grained control
        )

        pipeline = make_pipeline(TableVectorizer(), clf)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            pipeline.fit(x_train, y_train)  # X should be a DataFrame

            predictions = pipeline.predict_proba(test_sample)

            batch_classes = pipeline.classes_  # usually available in scikit-learn-like pipelines

            # Normalize to global class shape
            normalized_preds = normalize_predictions(predictions, batch_classes, all_classes)

            all_predictions.append(normalized_preds)

    create_submission(predictions=all_predictions, dataset_path=str(setup_path), output_path=workspace_path)


def main(competition_name: str, workspace_path: str, setup_path: str, context_length: int) -> None:
    dataset_path = get_dataset_path(dataset_path=setup_path)
    print(f"Setups found at {dataset_path}")
    if not check_necessary_files_exists(dataset_path=dataset_path):
        print(f"[❌ ERROR] Could not find a successful setup {setup_path}.")
        sys.exit(1)

    run_inference(
        competition=competition_name,
        workspace_path=workspace_path,
        setup_path=dataset_path,
        context_length=context_length
    )


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(
        competition_name=args.competition_name,
        workspace_path=args.workspace_path,
        setup_path=args.setup_path,
        context_length=args.context_length
    )
