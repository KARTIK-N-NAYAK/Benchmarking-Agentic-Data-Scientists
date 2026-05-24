RAMP AutoDS
=============

Automated Tabular Data Scientist based on the RAMP ecosystem.

# Installation

# Setup

1. Clone a RAMP setup kit, for example, [`kaggle_abalone`](https://github.com/ramp-setup-kits/kaggle_abalone), place it in `./ramp-setup-kits` (by convention), and download the data from the kaggle site.
```
ramp-setup-kits
└───kaggle_abalone
    ├───metadata.json
    ├───sample_submission.csv
    ├───test.csv
    └───train.csv
```

2. If you haven't, create the `ramp-kits` folder
```
mkdir ramp-kits
```

3. Set up the first run on your new kit:
```
cd ramp-kits
ramp-setup --ramp-kit kaggle_abalone --version 1_1 --number 1
```
`version` and `number` can be any string. Conventionally we use `version` to mark either a version of `ramp-autods` or a config file specifying command-line parameters, and `number` to mark an execution, like a seed.

If your setup folder is not `../ramp-setup-kits`, you can specify it with `--setup-root`.

The result is a functional RAMP kit (also available [here](https://github.com/ramp-kits/kaggle_abalone), for reference), with the starting kit submission (an LGBM) trained, tested, and scored.

<details><summary>Explanation of the resulting folder structure</summary>

```
ramp-kits
└───kaggle_abalone_v1_1_n1
    ├───actions
    │   ├───'2025-02-24 16:57:58.900437.pkl'
    │   └───...
    ├───data
    │   ├───metadata.json
    │   ├───sample_submission.csv
    │   ├───test.csv
    │   └───train.csv
    ├───problem.py
    └───submissions
        └───starting_kit
            ├───data_preprocessor_0_drop_id.py
            ├───data_preprocessor_1_drop_columns.py
            ├───data_preprocessor_2_col_in_train_only.py
            ├───data_preprocessor_3_Sex_1_cat_col_encoding.py
            ├───data_preprocessor_4_rm_constant_col.py
            ├───feature_extractor.py
            ├───regressor.py
            └───training_output
                └───starting_kit
                    ├───bagged_scores.csv
                    ├───fold_0
                    │   ├───scores.csv
                    │   ├───y_pred_test.npz
                    │   └───y_pred_train.npz
                    ├───...
                    ├───submission_bagged_test.csv 
                    └───submission_bagged_valid.csv
      
```
1. `actions` contains tiny python pickles storing all parameters and return values of function called during the run, marked with the `@ramp_action` decorator in the code. We use this heterogeneous database to recover timings, scores, and other statistics about the run, as well as to resume a crashed run.
2. `data` is a copy of the setup kit, except than `metadata.json` is augmented with stats computed after reading the training and test data.
3. `problem.py` is the [config file that describes the kit](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html). You can modify it after the setup, if you know what you are doing.
4. `submissions` is the folder with all the submissions (data preprocessors and a predictor, inserted into the [classification]() or [regression]() workflow.). After setup, a single `starting_kit` submission is submitted, containing an LGBM `regressor.py`, and a list of `data_preprocessor`s (some of them fixed, some of them depending on the data). The `feature_extractor.py` is currently blank; data preprocessors are executed once, before the folds are created, while the feature extractor is called for every fold.
5. `submissions/<submission>/training_output` stores all the results, including a table `bagged_scores.csv` for all the foldwise scores and runtimes, and `submission_bagged_test.csv` that is a valid submission file that can be submitted to Kaggle. For each fold, we store `y_pred_train.npz` (training + validation predictions) and `y_pred_test.npz` (test predictions). These can be large files, but we need to keep them until the final blend.
</details>

<details><summary>Optional RAMP test</summary>

The kit is a valid RAMP kit which means that all RAMP commands can be used on it. You can test
```
ramp-test
```
which will re-train the starting kit, but on all 30 CV folds instead of the three folds we use to check in the setup. If you modify `problem.py`, we suggest that you unit test it using `ramp-test`.
</details>
