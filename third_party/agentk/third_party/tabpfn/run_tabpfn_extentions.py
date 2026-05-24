import os
import sys

import pandas as pd
from tabpfn_extensions.many_class import ManyClassClassifier
from tqdm import tqdm

from metadata import MetaData
from utils import parse_args, prepare_test_set_chunks, get_data_paths, write_training_points, \
    prepare_dataset, fit_and_predict, get_tabpfn_class, create_submission, preprocess_dataset, get_dataset_path, \
    check_necessary_files_exists


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

    kwargs = dict(
        ignore_pretraining_limits=True,  # (bool) Allows the use of datasets larger than pretraining limits.
        n_estimators=32,  # (int) Number of estimators for ensembling; improves accuracy with higher values.
        inference_config={
            "SUBSAMPLE_SAMPLES": context_length,
            # (int) Maximum number of samples per inference step to manage memory usage.
        },
    )

    for test_sample in tqdm(test_samples, desc=f"Predicting", leave=False, position=2):
        x_train, y_train = prepare_dataset(
            train_data=train_data,
            id_field=id_field,
            target_cols=target_cols,
            test_sample=test_sample,
            context_length=context_length
        )
        test_sample = test_sample.drop(columns=target_cols + [id_field], errors='ignore')

        model = get_tabpfn_class(is_classification=is_classification, **kwargs)

        if is_classification:
            model = ManyClassClassifier(
                estimator=model,
                alphabet_size=10,
                n_estimators_redundancy=4,
                random_state=42,
            )

        predictions = fit_and_predict(
            model=model,
            x_train=x_train,
            y_train=y_train,
            test_sample=test_sample,
            is_classification=is_classification,
            all_classes=all_classes
        )

        all_predictions.append(predictions)

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
