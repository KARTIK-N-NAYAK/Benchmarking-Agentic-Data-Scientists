import os
import glob
import pytest
import shutil
import hashlib
import numpy as np
import ramphy as rh
import rampwf as rw
from pathlib import Path

PATH = os.path.dirname(__file__)

def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'ramp_setup_kits', '*'))):
        grid.append(os.path.abspath(path_kit))
    return grid

def _preprocessor_tester(path_kit: str, preprocessor_name: str):
    ramp_kit_dir = Path(PATH) / 'ramp_kits' / 'test_kit'
    ramp_data_dir = ramp_kit_dir
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    # Setup and starting kit
    rh.ramp_setup.kit_setup(
        download_dir = path_kit,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_data_dir,
    )

    submission = f'xgboost_{preprocessor_name}'
    rh.ramp_setup.tabular_regression_ordered_submit(
        submission=submission,
        regressor="xgboost",
        feature_extractor="empty",
        data_preprocessors=["drop_id", preprocessor_name],
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )

    rh.actions.train(
        submission=submission,
        fold_idxs=[900, 901],
        ramp_kit_dir=str(ramp_kit_dir),
        ramp_data_dir=str(ramp_data_dir),
    )

    n_trials = 9

    # hyperopt
    rh.actions.hyperopt(
        submission=submission,
        n_trials=n_trials,
        fold_idxs=range(900, 903),
        ramp_kit_dir=str(ramp_kit_dir),
        ramp_data_dir=str(ramp_data_dir),
        resume=True
    )

    # cleaning up
    shutil.rmtree(ramp_kit_dir)
    shutil.rmtree('cache')

@pytest.mark.xfail
@pytest.mark.parametrize("path_kit", _generate_grid_path_kits())
def test_l1_selector(path_kit: str):
    # Expected to fail cause only for classification
    _preprocessor_tester(path_kit=path_kit, preprocessor_name='l1_selector')

@pytest.mark.xfail
@pytest.mark.parametrize("path_kit", _generate_grid_path_kits())
def test_rf_selector(path_kit: str):
    # Expected to fail cause only for classification
    _preprocessor_tester(path_kit=path_kit, preprocessor_name='rf_selector')

@pytest.mark.xfail
@pytest.mark.parametrize("path_kit", _generate_grid_path_kits())
def test_select_kbest(path_kit: str):
    _preprocessor_tester(path_kit=path_kit, preprocessor_name='select_kbest')

@pytest.mark.xfail
@pytest.mark.parametrize("path_kit", _generate_grid_path_kits())
def test_variance_threshold(path_kit: str):
    _preprocessor_tester(path_kit=path_kit, preprocessor_name='variance_threshold')