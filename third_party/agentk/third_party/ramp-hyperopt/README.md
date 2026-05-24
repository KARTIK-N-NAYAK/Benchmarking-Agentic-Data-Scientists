RAMP hyperopt
=============

Hyperopt package for ramp-workflow.

# Installation

# How to use

You need a RAMP kit set up, for example, from [here](https://github.com/ramp-kits/kaggle_abalone), with a submission to be hyperopted. Each workflow element can have a section describing the hyperparameters and grids of values, see for example the [regressor](https://github.com/ramp-kits/kaggle_abalone/blob/master/submissions/starting_kit/regressor.py). You can hyperopt each workflow element separately, or at the same time. You can start by 
```
ramp-hyperopt --config config_hyperopt_1.ini
```
using the config file [here](https://github.com/ramp-kits/kaggle_abalone/blob/master/config_hyperopt_1.ini). It will run three trials on the first three folds from the CV object defined in [`problem.py`](https://github.com/ramp-kits/kaggle_abalone/blob/master/problem.py#L30), using the [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO) hyperopt engine. It will create three new submissions `submissions/starting_kit_hyperopt_<hash>` with the hyperparameters proposed by the engine, train and score them. These are ordinary RAMP submissions that can also be tested using `ramp-test`. They are identical to the [original regressor](https://github.com/ramp-kits/kaggle_abalone/blob/master/submissions/starting_kit/regressor.py) except for the `default` values in the `Hyperparameter` objects (used by `ramp-test`) that are replaced by the values proposed by HEBO.

In addition, `ramp-hyperopt` will also create or update `submissions/starting_kit/hyperopt_output/summary.csv` containing a list of the selected hyperparameter values, and the corresponding train and valid scores and runtimes.

The summary file is also used to resume (warm start) the hyperopt run using
```
ramp-hyperopt --config config_hyperopt_1.ini --resume
```

You can also hyperopt the [`data_preprocessor_1_drop_columns`](https://github.com/ramp-kits/kaggle_abalone/blob/master/submissions/starting_kit/data_preprocessor_1_drop_columns.py) workflow element (boolean hypers of which column to drop) by using 
```
ramp-hyperopt --config config_hyperopt_2.ini --resume
```
Resuming means that the three new submissions will be concatenated to `submissions/starting_kit/hyperopt_output/summary.csv`, and subsequent resumes will start form the joint space of the hyperparameters of these two workflow elements.

You can also jointly optimize the two by setting
```
workflow_elements_to_hyperopt = ['regressor', 'data_preprocessor_1_drop_columns']
```
in the congfig file, but we found that alternate optimizatioon works better. When only `data_preprocessor_1_drop_columns` is optimized, the regressor will be fixed to the one in the starting kit. If you want to use another regressor, you need to replace
```
submission = 'starting_kit_hyperopt_<hash>'
```
When you do this, `ramp-hyperopt` will create new submissions `starting_kit_hyperopt_<hash>` (and not, recursively, `starting_kit_hyperopt_<hash1>_hyperopt_<hash2>`), and concatenate the new hypers to `submissions/starting_kit/hyperopt_output/summary.csv`. So you can continue this alternate optimization. This, of course, works with any number of hyperoptable workflow elements, as long as you don't change the joint hyperparameter space. 

## Command-line parameters

`--submission`: the submission in `submissions` to hyperopt. Workflow elements that are not hyperopted, will be copied into the new submissions from this submission.

`--ramp-kit-dir`: the folder of the ramp kit, default `./`.

`--ramp-data-dir`: the folder of the ramp data (data should be in `<ramp-data-dir>/data`), default `./`.

`--engine`:
- `ray_hebo` (suggested)
- `ray_zoopt`
- `ray_optuna` (does not work with `--resume`)
- `ray_ax` (does not work: AssertionError: Experiment not set on Ax client)
- `ray_skopt` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)
- `ray_hyperopt` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)
- `ray_bayesopt` (does not work: AttributeError: module 'bayes_opt' has no attribute 'UtilityFunction')
- `ray_bohb` (does not work, can't install: ERROR: Could not build wheels for netifaces)
- `ray_nevergrad` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)

`--n-trials`: the number of hyperopt trials, default 10.

`--n-folds`: the number of hyperopt folds (in case you don't want to use all the CV folds specified in `problem.py`), default 3.

`--first-fold-idx`: the index of the first fold, default 0.

`--workflow-elements-to-hyperopt`: the workflow elements to hyperopt, default [].

`--save-output`: whether to save the output of the hyperopt, default `True`.

`--resume`: to resume hyperopt, default `False`.

`--max-concurrent-runs`: Maximum number of trials to run concurrently. Must be non-negative. None or 0 means no limit. Default 1.

`--n-cpu-per-run`: the number of CPUs per trial, default 1.

`--n-gpu-per-run`: the number of GPUs per trial, default 0.

`--verbose`: 0, 1, 2, or 3. Verbosity mode. 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results. Defaults to 3.



