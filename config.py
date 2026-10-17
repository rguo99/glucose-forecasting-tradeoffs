"""
Central path configuration.

All data and output locations are resolved here from environment variables,
so no script contains machine-specific paths.

Environment variables
---------------------
T1DEXI_DATA_DIR  (required for real-data runs)
    Output root of the t1dexi-preprocessing pipeline. Expected layout:
        netiob_cleaned_data/user_device_data.parquet
        processed_data/non_exercise_8h/          (instances + by_user/ features)
        data_filtering/non_exercise_8h/          (train/test instance lists)
T1DEXI_RAW_DIR   (needed only by heterogeneity_drivers.py)
    Folder with the raw T1DEXI domains converted to parquet (DM, LB, PR).
RESULTS_DIR      (optional, default: <repo>/output)
    Where model runs and all derived tables/figures are written.
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.abspath(os.environ.get('RESULTS_DIR', os.path.join(REPO_ROOT, 'output')))
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')
MAIN_RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'main_results')
EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'eval_results')
CACHE_DIR = os.path.join(RESULTS_DIR, 'cache')

COHORT_FOLDER = 'non_exercise_8h'


def _data_root():
    root = os.environ.get('T1DEXI_DATA_DIR')
    if not root:
        raise RuntimeError(
            'T1DEXI_DATA_DIR is not set. Point it at the output root of the '
            't1dexi-preprocessing pipeline (see README).')
    return os.path.abspath(os.path.expanduser(root))


def data_dir():
    """Instance table and per-user feature files."""
    return os.path.join(_data_root(), 'processed_data', COHORT_FOLDER)


def filter_dir():
    """Curated train/test instance lists."""
    return os.path.join(_data_root(), 'data_filtering', COHORT_FOLDER)


def device_parquet():
    return os.path.join(_data_root(), 'netiob_cleaned_data', 'user_device_data.parquet')


def raw_dir():
    root = os.environ.get('T1DEXI_RAW_DIR')
    if not root:
        raise RuntimeError('T1DEXI_RAW_DIR is not set (folder with raw T1DEXI DM/LB/PR parquet files).')
    return os.path.abspath(os.path.expanduser(root))
