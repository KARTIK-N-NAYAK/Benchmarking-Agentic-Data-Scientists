# AutoGluon Local Setup & Training Guide

This guide explains how to set up AutoGluon locally and run the training + prediction pipeline.

---

## 1. Requirements

- Python 3.9 – 3.11
- (Optional) NVIDIA GPU with CUDA drivers installed
- Conda or Python venv

## 2. Create an AutoGluon environment
conda create -n autogluon_env python=3.10 -y
conda activate autogluon_env

## 3. Install AutoGluon
pip install autogluon.tabular

## 4. Install common dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

### Notes
For small datasets, medium_quality preset is sufficient.

For larger datasets, increase training time:
--time_limit 1200


For stronger models:
--presets best_quality