import os
import glob
import pytest
import shutil
import hashlib
import numpy as np
import ramphy.ramp_setup as rs
from pathlib import Path

PATH = os.path.dirname(__file__)


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'ramp_setup_kits', '*'))):
        grid.append(Path(path_kit).name)
    return grid


@pytest.mark.parametrize(
    "ramp_kit",
    _generate_grid_path_kits()
)
def test_submission(ramp_kit):
    kit_root = Path(PATH) / 'ramp_kits'
    setup_root = Path(PATH) / 'ramp_setup_kits'
    ramp_kit_dir = kit_root / f"{ramp_kit}_v0_n0"

    # cleaning up
    if ramp_kit_dir.is_dir():
        shutil.rmtree(ramp_kit_dir)
    if Path('cache').is_dir():
        shutil.rmtree('cache')
    if Path('catboost_info').is_dir():
        shutil.rmtree('catboost_info')

#    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    rs.scripts.setup.setup(
        ramp_kit = ramp_kit,
        setup_root = setup_root,
        kit_root = kit_root,
        version = "0",
        number = 0,
    )

    rs.orchestration.hyperopt_race(
        ramp_kit=ramp_kit,
        kit_root=kit_root,
        version="0",
        number=0,
        resume=False,
        n_rounds=1,
        n_trials_per_round=3,
        n_folds_hyperopt=3,
        n_folds_final_blend=7,
        base_predictors=["lgbm", "xgboost", "catboost"],
        top_n_for_mean=2,
        n_sigma=1.0,
    )

    rs.orchestration.hyperopt_race(
        ramp_kit=ramp_kit,
        kit_root=kit_root,
        version="0",
        number=0,
        resume=True,
        n_rounds=1,
        n_trials_per_round=3,
        n_folds_hyperopt=3,
        n_folds_final_blend=7,
        base_predictors=["lgbm", "xgboost", "catboost"],
        top_n_for_mean=2,
        n_sigma=1.0,
    )

    # cleaning up
    if ramp_kit_dir.is_dir():
        shutil.rmtree(ramp_kit_dir)
    if Path('cache').is_dir():
        shutil.rmtree('cache')
    if Path('catboost_info').is_dir():
        shutil.rmtree('catboost_info')
