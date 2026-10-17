"""
real_data_evaluation.py
=======================
Per-user model fitting entry point. Loads one user's curated instances
(--data-path, --curation-dir), fits the requested model through
InferenceRunner.run_user_level_inference_unified, and writes the user's
results bundle (posterior/predictive samples, diagnostics, provenance) under
--output-path. Normally invoked by run_cohort.py / run_experiments.py, one
subprocess per user.
"""

import os
import sys

# JAX backend choice, decided before JAX is imported. All models in the paper
# run JAX on CPU: Bayesian Loop and Pure AR1 sample with numpyro on CPU, AR1-RLS
# is closed-form, and the neural models train in PyTorch (which uses the GPU
# itself), so JAX is only used for bookkeeping arrays. Keeping JAX off the GPU
# also avoids CUDA/cuDNN version conflicts between the JAX and PyTorch wheels.
_JAX_CPU_ONLY_MODELS = {'ar1_rls', 'transformer_mc', 'lstm_ensemble', 'lstm_conformal',
                        'transformer_conformal', 'transformer_ensemble',
                        'lstm_mc', 'ar1', 'ar1_pure'}
# Models whose Ray worker should not request a GPU (pure CPU).
_NO_GPU_RESOURCE_MODELS = {'ar1_rls'}                  # (2) Ray GPU allocation
_early_model = None
if '--model' in sys.argv:
    _idx = sys.argv.index('--model')
    if _idx + 1 < len(sys.argv):
        _early_model = sys.argv[_idx + 1]
_USE_GPU = _early_model not in _JAX_CPU_ONLY_MODELS
_JAX_PLATFORM = "gpu" if _USE_GPU else "cpu"
# Kept as the name InferenceRunner.py / downstream code already expects.
_CPU_ONLY_MODELS = _NO_GPU_RESOURCE_MODELS

# PLACE THESE AT THE VERY TOP, BEFORE ANY JAX IMPORTS
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Ray workers (spawned for the SVI parallel path) don't inherit this
# process's sys.argv, so InferenceRunner.py's own --model sniffing sees
# nothing in the worker and would default back to "gpu" even when the
# driver correctly resolved "cpu" above. Env vars (which Ray DOES propagate
# to workers) make the platform choice survive into the worker's fresh jax
# import. JAX_PLATFORMS (plural) additionally prevents JAX from even
# attempting to discover/initialize the CUDA plugin -- JAX_PLATFORM_NAME
# alone only sets the *default* backend but JAX still probes CUDA at
# import time, which is what was hitting the cuDNN mismatch.
os.environ["JAX_PLATFORM_NAME"] = _JAX_PLATFORM
if _JAX_PLATFORM == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_platform_name", _JAX_PLATFORM)
# Use Float32 for 2x speed and memory efficiency on GPU (highly recommended)
jax.config.update("jax_enable_x64", False)

import numpyro
numpyro.set_platform(_JAX_PLATFORM)
print(f">>> real_data_evaluation.py: JAX/numpyro platform = {_JAX_PLATFORM} "
      f"(model={_early_model!r}, jax_cpu_only_models={sorted(_JAX_CPU_ONLY_MODELS)}, "
      f"no_gpu_resource_models={sorted(_NO_GPU_RESOURCE_MODELS)})")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import argparse
import json
import pickle

# Import unified inference function and ModelType enum
from InferenceRunner import run_user_level_inference_unified, ModelType
from models.loop_model import *

import ray


# exp_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# os.environ['EXPERIMENT_TIMESTAMP'] = exp_date
exp_date = os.environ.get('EXPERIMENT_TIMESTAMP', 
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.environ['EXPERIMENT_TIMESTAMP'] = exp_date
print(f"EXPERIMENT TIMESTAMP: {exp_date}")

# At the top of real_data_evaluation.py, add:
from models.loop_model import USE_HEURISTICS
print(f">>> real_data_evaluation.py: USE_HEURISTICS = {USE_HEURISTICS}")


def _build_dataset_label(dataset_mode: str, use_carbs_only: bool) -> str:
    if dataset_mode == 'single_user':
        if use_carbs_only:
            return 'single_user_carbs_only'
        else:
            return 'single_user_all_instances'
    else:  # all_users
        if use_carbs_only:
            return 'all_users_carbs_only'
        else:
            return 'all_users_all_instances'

def run_evaluation_remote(user_datasets, default_settings, target_user_id, 
                          output_base_path, model_type_str, dataset_mode, use_carbs_only,
                          use_carb_uncertainty, carb_uncertainty_type, 
                          per_event_method, mc_carb_samples, 
                          use_user_param_prior,
                          inference_method, num_warmup, num_samples, num_chains, num_svi_steps, svi_lr,
                          plot_subset_instances=None, initial_params=None,
                          max_train_instances=None,
                          fixed_test_size=None,
                          sampling_mode='chronological',
                          train_instance_ids=None,
                          test_instance_ids=None):
    
    from InferenceRunner import ModelType
    model_type = ModelType(model_type_str)
    
    evaluator = SequentialLoopEvaluator(
        default_settings, 
        target_user_id=target_user_id,
        model_type=model_type,
        dataset_mode=dataset_mode,
        use_carbs_only=use_carbs_only,
        use_carb_uncertainty=use_carb_uncertainty,
        carb_uncertainty_type=carb_uncertainty_type,
        per_event_method=per_event_method,
        mc_carb_samples=mc_carb_samples,
        use_user_param_prior=use_user_param_prior,
        inference_method=inference_method,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        num_svi_steps=num_svi_steps,
        svi_lr=svi_lr,
        plot_subset_instances=plot_subset_instances,
        initial_params=initial_params,
        max_train_instances=max_train_instances,
        fixed_test_size=fixed_test_size,
        sampling_mode=sampling_mode,
        train_instance_ids=train_instance_ids,
        test_instance_ids=test_instance_ids,
    )
    
    dataset_label = _build_dataset_label(dataset_mode, use_carbs_only)
    try:
        evaluator.run_evaluation(user_datasets, output_base_path)
    except Exception as e:
        print(f"ERROR: Error evaluating user {target_user_id} with {model_type_str}: {e}")
        import traceback
        traceback.print_exc()


class SequentialLoopEvaluator:
    def __init__(self, default_patient_settings: Dict[str, float], 
                 target_user_id: Optional[str] = None,
                 model_type: ModelType = None,
                 dataset_mode: str = 'single_user',
                 use_carbs_only: bool = False,
                 use_carb_uncertainty: bool = False,
                 carb_uncertainty_type: str = 'global',
                 per_event_method: str = 'direct',
                 mc_carb_samples: int = 30,
                 use_user_param_prior: bool = False,
                 inference_method: str = 'svi',
                 num_warmup: int = 500,
                 num_samples: int = 1000,
                 num_chains: int = 4,
                 num_svi_steps: int = 5000,
                 svi_lr: float = 0.001,
                 plot_subset_instances: int = None,
                 initial_params=None,
                 max_train_instances: int = None, 
                 fixed_test_size: int = None,
                 sampling_mode: str = 'chronological',
                 train_instance_ids: set = None,
                 test_instance_ids: set = None):
        
        self.default_settings = default_patient_settings
        self.isf_cr_data = None
        self.target_user_id = target_user_id
        self.model_type = model_type
        self.dataset_mode = dataset_mode
        self.use_carbs_only = use_carbs_only
        self.use_carb_uncertainty = use_carb_uncertainty
        self.carb_uncertainty_type = carb_uncertainty_type
        self.per_event_method = per_event_method
        self.mc_carb_samples = mc_carb_samples
        self.use_user_param_prior = use_user_param_prior
        self.inference_method = inference_method
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_svi_steps = num_svi_steps
        self.svi_lr = svi_lr
        self.plot_subset_instances = plot_subset_instances
        self.initial_params = initial_params
        self.max_train_instances = max_train_instances
        self.fixed_test_size = fixed_test_size
        self.sampling_mode = sampling_mode
        self.train_instance_ids = train_instance_ids
        self.test_instance_ids = test_instance_ids
        
        # Logging setup...
        carbs_label = "carbs-only" if use_carbs_only else "all-data"
        carb_config = f"carb_uncertainty={carb_uncertainty_type}" if use_carb_uncertainty else "no_carb_uncertainty"
        if use_carb_uncertainty and carb_uncertainty_type == 'per_event':
            carb_config += f", method={per_event_method}"
        
        print(f"Initialized evaluator with:")
        print(f"  Model: {model_type.value}")
        print(f"  Dataset mode: {dataset_mode}")
        print(f"  Dataset type: {carbs_label} filtered")
        
    
    def load_derived_isf_cr(self, isf_cr_path: str) -> bool:
        try:
            self.isf_cr_data = pd.read_csv(isf_cr_path)
            return True
        except FileNotFoundError:
            print(f"Warning: ISF/CR file not found at {isf_cr_path}")
            return False

    def filter_df(self, df, user_col='user_id', instance_col='ex'):
        """
        Robustly filter dataframe to ONLY include curated instances from curate_data.py.
        Uses a fast set-based lookup to avoid KeyError and performance issues.
        """
        if df.empty:
            return df

        # 1. Resolve path to the curated dictionary
        carbs_suffix = "_with_carbs" if self.use_carbs_only else ""
        if self.dataset_mode == 'all_users':
            rel_folder = f"expanded_non_exercise_all_users{carbs_suffix}"
        else:
            rel_folder = f"expanded_non_exercise_single_user{carbs_suffix}"
        
        possible_pickle_paths = [
            f"data_preprocessing/data_filtering/{rel_folder}/filtered_user_instances.pkl",
            f"data_filtering/{rel_folder}/filtered_user_instances.pkl",
            # New pipeline — absolute path via curation_dir attribute if set
            os.path.join(self._curation_dir, "filtered_user_instances.pkl")
            if hasattr(self, '_curation_dir') and self._curation_dir else None,
        ]
        possible_pickle_paths = [p for p in possible_pickle_paths if p]
        
        pickle_path = next((p for p in possible_pickle_paths if os.path.exists(p)), None)
        
        if not pickle_path:
            # If the file doesn't exist, we don't filter. 
            # This allows the script to run on raw data if curation hasn't been done.
            return df
        
        # 2. Load and Normalize Dictionary
        with open(pickle_path, 'rb') as f:
            raw_dict = pickle.load(f)
        
        # Force all keys to strings for matching (e.g., 37.0 -> "37")
        curated_lookup = {}
        for k, v in raw_dict.items():
            try:
                norm_k = str(int(float(k)))
                curated_lookup[norm_k] = set(v)
            except:
                curated_lookup[str(k)] = set(v)

        # 3. Apply Filtering
        # We process each user in the dataframe separately to handle type mismatches
        filtered_chunks = []
        
        # Normalize the user column in the DF to strings for grouping
        # Use a separate series (not a DF column) to avoid a KeyError
        df_user_ids = df[user_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
        
        for norm_uid in df_user_ids.unique():
            if norm_uid in curated_lookup:
                # Get all rows for this specific user
                user_mask = (df_user_ids == norm_uid)
                user_group = df[user_mask]
                
                # Filter this user's rows by the allowed instance IDs
                allowed_ids = curated_lookup[norm_uid]
                valid_rows = user_group[user_group[instance_col].isin(allowed_ids)]
                filtered_chunks.append(valid_rows)
        
        if not filtered_chunks:
            # Return empty DF with same columns if nothing matched
            return pd.DataFrame(columns=df.columns)
            
        return pd.concat(filtered_chunks).copy()
    
    def _read_feature(self, data_path, filename_base):
        """Load parquet if available, fall back to csv."""
        parquet_path = os.path.join(data_path, filename_base.replace('.csv', '.parquet'))
        csv_path     = os.path.join(data_path, filename_base)
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None
    
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        by_user_dir = os.path.join(data_path, 'by_user')
        use_by_user = (
            self.target_user_id is not None
            and os.path.exists(by_user_dir)
        )

        feature_map = {
            'cgm':        'cgm',
            'basal':      'basal',
            'bolus':      'bolus',
            'meals':      'meals',
            'heart_rate': 'verily_hr',
            'step_count': 'step_count',
        }

        datasets = {}
        for key, feat_name in feature_map.items():
            if use_by_user:
                # Load just this user's parquet — fast, small
                fp = os.path.join(by_user_dir, feat_name,
                                f'{self.target_user_id}.parquet')
                if os.path.exists(fp):
                    df = pd.read_parquet(fp)
                    print(f"  Loaded {key} (by_user): {len(df)} records")
                else:
                    print(f"  WARNING: Missing by_user/{feat_name}/{self.target_user_id}.parquet")
                    df = pd.DataFrame()
            else:
                # Fall back to full cohort parquet for multi-user runs
                carbs_suffix = "_with_carbs" if self.use_carbs_only else ""
                df = self._read_feature(
                    data_path, f'{feat_name}{carbs_suffix}.csv')
                if df is None:
                    print(f"  WARNING: Missing: {feat_name}")
                    df = pd.DataFrame()
                else:
                    print(f"  Loaded {key}: {len(df)} records")

            # Timestamp normalisation (same as before)
            if not df.empty:
                if key == 'cgm' and 'datestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['datestamp'])
                elif key == 'basal' and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # New pipeline uses basal_units (U/5min) — add basal_rate for compatibility
                    if 'basal_units' in df.columns and 'basal_rate' not in df.columns:
                        df['basal_rate'] = df['basal_units'] * 12.0  # U/5min -> U/hr
                elif key == 'basal' and 'basal_datestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['basal_datestamp'])
                elif 'datestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['datestamp'])

            datasets[key] = df

        # Filter to curated instances (still needed even with by_user,
        # since by_user contains ALL instances not just curated ones)
        for feature in datasets:
            if not datasets[feature].empty:
                before = len(datasets[feature])
                datasets[feature] = self.filter_df(datasets[feature])
                after = len(datasets[feature])
                if before != after:
                    print(f"    -> Filtered {feature}: {before} -> {after}")

        return datasets
    
    def _get_user_id_from_instance(self, datasets: Dict[str, pd.DataFrame], instance_id: str) -> Optional[str]:
        for dataset_name, dataset in datasets.items():
            if not dataset.empty and 'ex' in dataset.columns and 'user_id' in dataset.columns:
                instance_data = dataset[dataset['ex'] == instance_id]
                if not instance_data.empty:
                    return instance_data['user_id'].iloc[0]
        return None
    
    def prepare_instance_data_with_raw_events(self, cgm_data: pd.DataFrame, basal_data: pd.DataFrame, 
                                            bolus_data: pd.DataFrame, meals_data: pd.DataFrame):
        """Prepare data preserving raw timestamps for all insulin and meal events"""
        cgm_clean = cgm_data[['timestamp', 'cgm']].dropna(subset=['cgm']).sort_values('timestamp')
        cgm_clean['timestamp'] = pd.to_datetime(cgm_clean['timestamp'])

        basal_events = pd.DataFrame(columns=['timestamp', 'basal_rate', 'basal_units'])
        if not basal_data.empty:
            if 'timestamp' in basal_data.columns:
                ts_col = 'timestamp'
            elif 'basal_start' in basal_data.columns:
                ts_col = 'basal_start'
            else:
                ts_col = None

            if ts_col and ('basal_units' in basal_data.columns or 'basal_rate' in basal_data.columns):
                cols = [ts_col]
                if 'basal_units' in basal_data.columns:
                    cols.append('basal_units')
                if 'basal_rate' in basal_data.columns:
                    cols.append('basal_rate')
                basal_events = basal_data[cols].copy()
                basal_events.rename(columns={ts_col: 'timestamp'}, inplace=True)
                # Ensure both columns exist for compatibility
                if 'basal_units' in basal_events.columns and 'basal_rate' not in basal_events.columns:
                    basal_events['basal_rate'] = basal_events['basal_units'] * 12.0
                elif 'basal_rate' in basal_events.columns and 'basal_units' not in basal_events.columns:
                    basal_events['basal_units'] = basal_events['basal_rate'] / 12.0
                basal_events['timestamp'] = pd.to_datetime(basal_events['timestamp'])
                basal_events = basal_events.sort_values('timestamp')
        
        bolus_events = pd.DataFrame(columns=['timestamp', 'bolus'])
        if not bolus_data.empty:
            bolus_column = 'bolus' if 'bolus' in bolus_data.columns else 'bolus_dose'
            if bolus_column in bolus_data.columns:
                if 'timestamp' in bolus_data.columns:
                    bolus_events = bolus_data[['timestamp', bolus_column]].copy()
                elif 'datestamp' in bolus_data.columns:
                    bolus_events = bolus_data[['datestamp', bolus_column]].copy()
                    bolus_events.rename(columns={'datestamp': 'timestamp'}, inplace=True)
                
                if not bolus_events.empty:
                    bolus_events['timestamp'] = pd.to_datetime(bolus_events['timestamp'])
                    bolus_events = bolus_events[bolus_events[bolus_column] > 0]
                    bolus_events.rename(columns={bolus_column: 'bolus'}, inplace=True)
        
        meal_events = pd.DataFrame(columns=['timestamp', 'carbs'])
        if not meals_data.empty:
            meal_column = 'meal_dose' if 'meal_dose' in meals_data.columns else 'carbs'
            if meal_column in meals_data.columns:
                if 'timestamp' in meals_data.columns:
                    meal_events = meals_data[['timestamp', meal_column]].copy()
                elif 'datestamp' in meals_data.columns:
                    meal_events = meals_data[['datestamp', meal_column]].copy()
                    meal_events.rename(columns={'datestamp': 'timestamp'}, inplace=True)
                
                if not meal_events.empty:
                    meal_events['timestamp'] = pd.to_datetime(meal_events['timestamp'])
                    meal_events = meal_events[meal_events[meal_column] > 0]
                    meal_events.rename(columns={meal_column: 'carbs'}, inplace=True)

        if len(cgm_clean) < 48:
            # print(f"WARNING: Warning: CGM has only {len(cgm_clean)} points")
            if len(cgm_clean) > 0:
                cgm_split_time = cgm_clean['timestamp'].iloc[-1]
            else:
                return {}, {}
        else:
            cgm_split_time = cgm_clean['timestamp'].iloc[47]
        
        historical_cgm = cgm_clean[cgm_clean['timestamp'] <= cgm_split_time].copy()
        future_cgm = cgm_clean[cgm_clean['timestamp'] > cgm_split_time].copy()
        
        historical_basal = basal_events[basal_events['timestamp'] <= cgm_split_time].copy()
        future_basal = basal_events[basal_events['timestamp'] > cgm_split_time].copy()
        
        historical_bolus = bolus_events[bolus_events['timestamp'] <= cgm_split_time].copy()
        future_bolus = bolus_events[bolus_events['timestamp'] > cgm_split_time].copy()
        
        historical_meals = meal_events[meal_events['timestamp'] <= cgm_split_time].copy()
        future_meals = meal_events[meal_events['timestamp'] > cgm_split_time].copy()
                    
        historical_data = {
            'cgm': historical_cgm, 'basal_events': historical_basal,
            'bolus_events': historical_bolus, 'meal_events': historical_meals,
            'split_time': cgm_split_time
        }
        
        future_data = {
            'cgm': future_cgm, 'basal_events': future_basal,
            'bolus_events': future_bolus, 'meal_events': future_meals,
            'split_time': cgm_split_time
        }

        return historical_data, future_data

    def run_evaluation(self, datasets: Dict[str, pd.DataFrame], output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)

        print(f"DEBUG run_evaluation datasets:")
        print(f"  meals shape: {datasets['meals'].shape}")
        print(f"  meals ex unique count: {datasets['meals']['ex'].nunique()}")
        print(f"  meals sample ex values: {sorted(datasets['meals']['ex'].unique()[:5])}")
        # END DEBUG

        instance_ids = datasets['cgm']['ex'].unique()
        print(f"Found {len(instance_ids)} instances for user {self.target_user_id or 'N/A'}")
        
        # If we have 0 instances here, it implies filtering killed everything.
        if len(instance_ids) == 0:
            print("ERROR: No instances found after filtering! Aborting.")
            return

        historical_datas = []
        future_datas = []
        user_id = self.target_user_id

        for i, instance_id in enumerate(instance_ids, 1):
            if i % 50 == 0: print(f"Processing instance {i}/{len(instance_ids)}...", end='\r')

            cgm_data = datasets['cgm'][datasets['cgm']['ex'] == instance_id].copy()
            basal_data = datasets['basal'][datasets['basal']['ex'] == instance_id].copy()
            bolus_data = datasets['bolus'][datasets['bolus']['ex'] == instance_id].copy()
            meals_data = datasets['meals'][datasets['meals']['ex'] == instance_id].copy()
            
            hist_data, fut_data = self.prepare_instance_data_with_raw_events(
                cgm_data, basal_data, bolus_data, meals_data
            )
            
            if not hist_data: continue

            # Add dummy heart/step
            hist_data['heart_rate'] = pd.DataFrame()
            fut_data['heart_rate'] = pd.DataFrame()
            hist_data['step_count'] = pd.DataFrame()
            fut_data['step_count'] = pd.DataFrame()

            if user_id is None:
                user_id = self._get_user_id_from_instance(datasets, instance_id)

            historical_datas.append(hist_data)
            future_datas.append(fut_data)

        if historical_datas:
            print(f"\nRunning user-level inference for user {user_id} on {len(historical_datas)} instances...")
            
            prepared_historical, prepared_future = self._prepare_inference_data(
                historical_datas, future_datas
            )
            
            run_user_level_inference_unified(
                prepared_historical, 
                prepared_future, 
                user_id, 
                instance_ids,
                model_type=self.model_type,
                dataset_mode=self.dataset_mode,
                use_carbs_only=self.use_carbs_only,
                use_carb_uncertainty=self.use_carb_uncertainty,
                carb_uncertainty_type=self.carb_uncertainty_type,
                per_event_method=self.per_event_method,
                mc_carb_samples=self.mc_carb_samples,
                use_user_param_prior=self.use_user_param_prior,
                inference_method=self.inference_method,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                num_chains=self.num_chains,
                use_svi_map=(self.inference_method.lower() == 'svi'),
                svi_num_steps=self.num_svi_steps,
                svi_learning_rate=self.svi_lr,
                output_path=output_path,
                plot_subset_instances=self.plot_subset_instances,
                initial_params=self.initial_params,
                max_train_instances=self.max_train_instances,
                fixed_test_size=self.fixed_test_size,
                sampling_mode=self.sampling_mode,
                train_instance_ids=self.train_instance_ids,
                test_instance_ids=self.test_instance_ids,
            )
        else:
            print(f"No valid instances processed for user {user_id}")

    def _prepare_inference_data(self, historical_datas, future_datas):
        # (This remains standard)
        prepared_historical = []
        prepared_future = []
        for hist_data, fut_data in zip(historical_datas, future_datas):
            hist_prep = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k,v in hist_data.items()}
            fut_prep = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k,v in fut_data.items()}
            
            split_time = hist_prep.get('split_time')
            if split_time is not None:
                split_time = pd.to_datetime(split_time)
                for key in ['basal_events', 'bolus_events', 'meal_events']:
                    if key in hist_prep and not hist_prep[key].empty:
                        hist_prep[key] = self._convert_to_relative_minutes(hist_prep[key], split_time)
                    if key in fut_prep and not fut_prep[key].empty:
                        fut_prep[key] = self._convert_to_relative_minutes(fut_prep[key], split_time)
            
            prepared_historical.append(hist_prep)
            prepared_future.append(fut_prep)
        return prepared_historical, prepared_future

    def _convert_to_relative_minutes(self, event_df, prediction_start):
        df = event_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        prediction_start = pd.to_datetime(prediction_start)
        df['time'] = (df['timestamp'] - prediction_start).dt.total_seconds() / 60.0
        return df

def parse_arguments():
    # Read environment variable defaults
    default_num_warmup = int(os.environ.get('NUM_WARMUP', 500))
    default_num_samples = int(os.environ.get('NUM_SAMPLES', 1000))
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all-users', action='store_true', default=False)
    parser.add_argument('--carbs_only', action='store_true', default=False)
    parser.add_argument('--list-sources', action='store_true')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--isf-cr-path', type=str, default="data/derived_ISF_CR.csv")
    parser.add_argument('--output-path', type=str, default="output")
    parser.add_argument('--target-user-id', type=int, default=None)
    parser.add_argument('--model', type=str, default='ar1')
    parser.add_argument('--use-carb-uncertainty', action='store_true')
    parser.add_argument('--carb-uncertainty-type', type=str, default='global')
    parser.add_argument('--per-event-method', type=str, default='direct')
    parser.add_argument('--mc-carb-samples', type=int, default=30)
    parser.add_argument('--inference-method', type=str, default='svi')
    parser.add_argument('--num-svi-steps', type=int, default=5000)
    parser.add_argument('--svi-lr', type=float, default=0.001)
    parser.add_argument('--num-warmup', type=int, default=default_num_warmup)
    parser.add_argument('--num-samples', type=int, default=default_num_samples)
    parser.add_argument('--gp-num-epochs', type=int, default=50)
    parser.add_argument('--gp-learning-rate', type=float, default=0.1)
    parser.add_argument('--default-isf', type=float, default=50.0)
    parser.add_argument('--default-cr', type=float, default=10.0)
    parser.add_argument('--default-basal', type=float, default=1.0)
    parser.add_argument('--num-cpus', type=int, default=50)
    parser.add_argument('--plot-subset', type=int, default=None)
    parser.add_argument('--initial-params', type=str, default=None)
    parser.add_argument('--num-chains', type=int, default=4)
    parser.add_argument("--max-train-instances", type=int, default=None)
    parser.add_argument("--sampling-mode", type=str, default='chronological')
    parser.add_argument('--curation-dir', type=str, default=None)
    parser.add_argument('--no-ray', action='store_true',
                        help='Run in-process instead of starting a local Ray '
                             'cluster. Auto-enabled when --target-user-id is '
                             'set, since that means exactly one task and Ray\'s '
                             'control plane is then pure overhead.')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Ray is started only for multi-user invocations. run_cohort.py runs
    # this script once per user (--target-user-id), where Ray's local cluster
    # (GCS server, raylet, idle workers) would only add threads; parallelism
    # comes from running one subprocess per GPU index instead.
    ray_num_gpus = 1 if args.model not in _CPU_ONLY_MODELS else 0
    NO_RAY = args.no_ray or (args.target_user_id is not None)
    if NO_RAY:
        print(f"[no-ray] single-user invocation (target_user_id="
              f"{args.target_user_id}) -- running in-process, Ray not started")
    else:
        ray.init(num_cpus=args.num_cpus, num_gpus=ray_num_gpus,
                 include_dashboard=False)
    
    # 2. Check Paths
    if not args.data_path or not args.curation_dir:
        raise ValueError("Both --data-path and --curation-dir must be provided.")
    
    print(f"Loading CSVs from: {args.data_path}")
    print(f"Loading Curation from: {args.curation_dir}")

    # 3. Load Curation Pickles
    try:
        with open(os.path.join(args.curation_dir, "train_instances.pkl"), 'rb') as f:
            global_train_ids = set(pickle.load(f))
        with open(os.path.join(args.curation_dir, "test_instances.pkl"), 'rb') as f:
            global_test_ids = set(pickle.load(f))
        
        # Combine them to get the total allowed instances
        allowed_instances = global_train_ids.union(global_test_ids)
        print(f"Loaded Curation: {len(global_train_ids)} Train | {len(global_test_ids)} Test")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Curation pickles missing in {args.curation_dir}. Did you run curate_data.py?")

    # 4. Determine users to process via by_user directory (fast, no full cohort load)
    by_user_cgm_dir = os.path.join(args.data_path, 'by_user', 'cgm')
    if os.path.exists(by_user_cgm_dir):
        all_user_ids = sorted([
            f.replace('.parquet', '')
            for f in os.listdir(by_user_cgm_dir)
            if f.endswith('.parquet')
        ], key=lambda x: int(x))
        print(f"\nFound {len(all_user_ids)} users in by_user/cgm/")
    else:
        raise FileNotFoundError(f"by_user/cgm not found at {by_user_cgm_dir}. Run feature extraction first.")

    if args.target_user_id is not None:
        unique_users = [str(args.target_user_id)]
        print(f"Filtered to target user: {args.target_user_id}")
    else:
        unique_users = all_user_ids

    print(f"Processing {len(unique_users)} users\n")
    datasets = {}  # Not used — each user loads via by_user parquets


    # In main(), after loading datasets, add:


    # 5. Model Setup Logic (Keep your existing baseline mapping here)
    model_type_str = args.model
    
    if model_type_str == 'lstm_mc':
        model_type = ModelType.LSTM_MC

    if model_type_str == 'transformer_mc':
        model_type = ModelType.TRANSFORMER_MC

    if model_type_str == 'lstm_ensemble':
        model_type = ModelType.LSTM_ENSEMBLE

    if model_type_str == 'lstm_conformal':
        model_type = ModelType.LSTM_CONFORMAL

    if model_type_str == 'transformer_conformal':
        model_type = ModelType.TRANSFORMER_CONFORMAL

    if model_type_str == 'transformer_ensemble':
        model_type = ModelType.TRANSFORMER_ENSEMBLE

    if args.use_carb_uncertainty and args.model not in ['ar1_pure', 'iid_noise_fixed_loop']:
        model_map = {'ar1': 'ar1_with_carb_uncertainty', 'iid_noise': 'iid_noise_with_carb_uncertainty'}
        model_type_str = model_map.get(args.model, args.model)
    
    default_settings = {'isf': args.default_isf, 'carb_ratio': args.default_cr, 'basal_rate': args.default_basal}

    # ======================================================================
    # 6. EXECUTION LOOP (Parallel for SVI, Sequential for MCMC)
    # ======================================================================
    if args.inference_method.lower() == 'svi':
        print(f"\n{'='*60}\nPARALLEL EXECUTION (SVI MODE) for {len(unique_users)} users\n{'='*60}")
        
        # Fractional GPUs: Allows all 7 users to share the 1 GPU assigned to this tmux window.
        # CPU-only models (ray_num_gpus=0 in the ray.init() pool above) must request 0 GPU
        # here too, or Ray blocks forever trying to satisfy a 0.1-GPU request from an empty pool.
        _worker_num_gpus = 0.1 if args.model not in _CPU_ONLY_MODELS else 0
        if NO_RAY:
            # Eager shim: .remote() runs the function immediately and returns
            # its result, so the submit loop and collect call below are
            # identical for both paths.
            class _InProcess:
                @staticmethod
                def remote(**kw):
                    return run_evaluation_remote(**kw)
            run_eval_remote = _InProcess
        else:
            run_eval_remote = ray.remote(num_cpus=2, num_gpus=_worker_num_gpus)(run_evaluation_remote)
        
        futures = []
        for uid in unique_users:
            # Load per-user data via by_user parquets — fast, no full cohort load
            user_eval = SequentialLoopEvaluator(
                {}, dataset_mode='all_users', model_type=ModelType('ar1'),
                target_user_id=str(uid))
            user_eval._curation_dir = args.curation_dir
            user_datasets = user_eval.load_data(args.data_path)

            user_instances = set(user_datasets['cgm']['ex'].unique()) if not user_datasets['cgm'].empty else set()
            user_train_ids = set(user_instances.intersection(global_train_ids))
            user_test_ids  = set(user_instances.intersection(global_test_ids))
            user_train_count = len(user_train_ids)
            user_test_count  = len(user_test_ids)

            if user_train_count == 0:
                print(f"Skipping User {uid} — no train instances")
                continue

            print(f"Queueing User {uid} (Train: {user_train_count}, Test: {user_test_count})")

            futures.append(run_eval_remote.remote(
                user_datasets=user_datasets,
                default_settings=default_settings,
                target_user_id=uid,
                output_base_path=args.output_path,
                model_type_str=model_type_str,
                dataset_mode='all_users',
                use_carbs_only=args.carbs_only,
                use_carb_uncertainty=args.use_carb_uncertainty,
                carb_uncertainty_type=args.carb_uncertainty_type,
                per_event_method=args.per_event_method,
                mc_carb_samples=args.mc_carb_samples,
                use_user_param_prior=False,
                inference_method=args.inference_method,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                num_svi_steps=args.num_svi_steps,
                svi_lr=args.svi_lr,
                plot_subset_instances=args.plot_subset,
                initial_params=args.initial_params,
                max_train_instances=user_train_count,
                fixed_test_size=user_test_count,
                sampling_mode='chronological',
                train_instance_ids=user_train_ids,
                test_instance_ids=user_test_ids,
            ))
        if not NO_RAY:
            ray.get(futures)   # NO_RAY path already executed eagerly above

    else:
        print(f"\n{'='*60}\nSEQUENTIAL EXECUTION (MCMC MODE) for {len(unique_users)} users\n{'='*60}")
        # Process one-by-one to protect massive MCMC VRAM consumption
        for uid in unique_users:
            print(f"\n--- Processing Target User: {uid} ---")

            # Load per-user data via by_user parquets — fast
            user_eval = SequentialLoopEvaluator(
                {}, dataset_mode='all_users', model_type=ModelType('ar1'),
                target_user_id=str(uid))
            user_eval._curation_dir = args.curation_dir
            user_datasets = user_eval.load_data(args.data_path)

            user_instances = set(user_datasets['cgm']['ex'].unique()) if not user_datasets['cgm'].empty else set()
            user_train_ids = set(user_instances.intersection(global_train_ids))
            user_test_ids  = set(user_instances.intersection(global_test_ids))
            user_train_count = len(user_train_ids)
            user_test_count  = len(user_test_ids)

            
            if user_train_count == 0:
                continue
                
            print(f"User {uid} Split -> Train: {user_train_count} | Test: {user_test_count}")
            
            run_evaluation_remote(
                user_datasets=user_datasets,
                default_settings=default_settings,
                target_user_id=uid,
                output_base_path=args.output_path,
                model_type_str=model_type_str,
                dataset_mode='all_users',
                use_carbs_only=args.carbs_only,
                use_carb_uncertainty=args.use_carb_uncertainty,
                carb_uncertainty_type=args.carb_uncertainty_type,
                per_event_method=args.per_event_method,
                mc_carb_samples=args.mc_carb_samples,
                use_user_param_prior=False,
                inference_method=args.inference_method,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                num_svi_steps=args.num_svi_steps,
                svi_lr=args.svi_lr,
                plot_subset_instances=args.plot_subset,
                initial_params=args.initial_params,
                max_train_instances=user_train_count,
                fixed_test_size=user_test_count,
                sampling_mode='chronological',
                train_instance_ids=user_train_ids,
                test_instance_ids=user_test_ids,
            )

    print(f"\n{'='*70}\nALL TARGETED USERS COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()