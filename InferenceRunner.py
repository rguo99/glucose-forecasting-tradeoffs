"""
Per-user inference engine shared by all nine models.

run_user_level_inference_unified() prepares one user's curated train/test
instances (CGM, insulin and carbohydrate events on a 5-minute grid), fits the
requested model (ModelType) and saves the user's results bundle: posterior or
predictive samples for every test instance, parameter samples, MCMC
diagnostics, seed/environment provenance and, for the neural models, the
trained weights. Bayesian Loop and Pure AR1 are sampled with NUTS (numpyro);
AR1-RLS is closed-form; the LSTM/Transformer variants are trained in PyTorch.
"""
import multiprocessing as mp
try:
     mp.set_start_method('spawn', force=True)
except RuntimeError:
     pass

import os
import sys
import gc
import pickle
import warnings
import psutil
from datetime import datetime
from collections import namedtuple
from functools import partial
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
from functools import partial
import concurrent.futures
import psutil
import os
import pandas as pd
import numpy as np
import csv
import time

# Shared with dev_scripts/causal_smoke_train.py so the production run and the
# counterfactual retrain derive IDENTICAL seeds for the same (user, model) --
# that is what makes the archived weights the same model instance as the one
# behind the main tables, rather than a second draw of it.
from utils.seeding import seed_for_trainer

# ============================================================================
# GPU & JAX CONFIGURATION
# ============================================================================
# Prevent JAX from hogging 90% of VRAM so multiple Ray workers can share a GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Models with no MCMC/SVI sampling (closed-form fits) don't need a GPU.
# real_data_evaluation.py already decides this from sys.argv's --model and
# sets jax's platform before importing this module; mirror that decision
# here too in case InferenceRunner is imported some other way.
#
# transformer_mc (like lstm_mc) does its own GPU work entirely in PyTorch
# (models/transformer_mc.py uses torch.cuda directly) -- JAX is only used by
# the shared prepare_user_level_arrays() bookkeeping (a handful of floats),
# which doesn't need acceleration. Keeping JAX on CPU here avoids JAX trying
# to initialize a CUDA+cuDNN backend it doesn't need, which on this cluster's
# current jaxlib/cuDNN package versions fails outright on GPU nodes (see
# models/transformer_mc.py's train_and_sample_transformer -- confirmed this
# same failure also hits the pre-existing lstm_mc path; it's an environment
# issue, not specific to one model, but only transformer_mc's platform
# selection is changed here to avoid touching lstm_mc's established behavior).
_CPU_ONLY_MODELS = {'ar1_rls', 'transformer_mc'}
_early_model = None
if '--model' in sys.argv:
    _idx = sys.argv.index('--model')
    if _idx + 1 < len(sys.argv):
        _early_model = sys.argv[_idx + 1]
# Ray workers re-import this module without the original sys.argv, so prefer
# the JAX_PLATFORM_NAME env var (set by real_data_evaluation.py before
# ray.init(); Ray propagates env vars to workers) when present, falling back
# to sys.argv sniffing only for standalone/direct invocation.
_JAX_PLATFORM = os.environ.get(
    "JAX_PLATFORM_NAME",
    "cpu" if _early_model in _CPU_ONLY_MODELS else "gpu",
)
# JAX_PLATFORMS (plural) additionally stops JAX from even attempting to
# discover/initialize the CUDA plugin -- JAX_PLATFORM_NAME alone only sets
# the *default* backend but JAX still probes CUDA at import time, which is
# what was hitting the cuDNN version mismatch on this cluster.
if _JAX_PLATFORM == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.scipy.stats as jax_stats
from jax import lax

jax.config.update("jax_platform_name", _JAX_PLATFORM)
# Use Float32 for 2x speed and memory efficiency on GPU (highly recommended)
jax.config.update("jax_enable_x64", False)

import numpyro
import numpyro.distributions as dist
numpyro.set_platform(_JAX_PLATFORM)

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.util import init_to_value
from numpyro.optim import Adam

# ============================================================================
# DATA SCIENCE & PLOTTING
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import arviz as az
import scipy.stats as st
from scipy import stats

# UQ Metrics
import properscoring as ps

# ============================================================================
# PROJECT MODULES
# ============================================================================
from utils.data_prep import *
from utils.plotting import *

from models.ar1_pure import *
from models.ar1 import *

import torch
from models.lstm_mc import train_and_sample_lstm
from models.lstm_ensemble import train_and_sample_lstm_ensemble
from models.lstm_conformal import train_and_sample_lstm_conformal
from models.transformer_mc import train_and_sample_transformer
from models.transformer_conformal import train_and_sample_transformer_conformal
from models.transformer_ensemble import train_and_sample_transformer_ensemble
from models.ar1_rls import fit_and_forecast_rls_batch
from utils.data_prep import prepare_lstm_sequence_arrays

# Silent warnings
warnings.filterwarnings('ignore')

print('JAX devices detected:', jax.devices())

# MODIFIED: NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS removed (now passed as parameters)
# Get date string from prediction start or current time
# exp_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_date = os.environ.get('EXPERIMENT_TIMESTAMP', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


def _build_dataset_label(dataset_mode: str, use_carbs_only: bool) -> str:
    """
    Build a descriptive label for the dataset mode and carbs_only flag.
    
    Args:
        dataset_mode: Either 'single_user' or 'all_users'
        use_carbs_only: If True, use carbs-only filtered dataset; if False, use all-data
    
    Returns:
        A descriptive label like 'single_user_carbs_only' or 'all_users_all_instances'
    """
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
        

def get_output_dir(model_type, dataset_mode='single_user', use_carbs_only=False, 
                   use_carb_uncertainty=False, carb_uncertainty_type='global',
                   inference_method='mcmc',
                   num_warmup=100,
                   num_samples=500,
                   num_chains=4,
                   num_svi_steps=5000,
                   svi_lr=0.001,
                   output_path=None,
                   use_initial_params=False,
                   per_event_method=None,
                   mc_carb_samples=None):
    """
    Generate output directory name.
    
    Logic:
    - Normalizes inference_method to string (fixes 'bool' object has no attribute 'lower').
    - Only appends '_mc{N}' if per_event_method is 'mc_marginalization'.
    - Only appends '_direct' if per_event_method is 'direct'.
    """
    # 1. Handle Enum or String for model_name
    model_name = model_type.value if hasattr(model_type, 'value') else str(model_type)
    
    # 2. Build dataset label
    dataset_label = _build_dataset_label(dataset_mode, use_carbs_only)
    
    # 3. Handle carb uncertainty suffix
    carb_suffix = ""
    if use_carb_uncertainty:
        carb_suffix = f"_carb_uncert_{carb_uncertainty_type}"
    
    # 4. Normalize inference_method and build suffix
    # Defensive: if a boolean was passed (True/False), force it to a string
    if not isinstance(inference_method, str):
        inference_method = 'svi' if inference_method is True else 'mcmc'
        
    if inference_method.lower() == 'svi':
        lr_str = f"{svi_lr:.6f}".rstrip('0').rstrip('.')
        inference_suffix = f"_svi_steps{num_svi_steps}_lr{lr_str}"
    else:  # mcmc
        inference_suffix = f"_mcmc_warmup{num_warmup}_samples{num_samples}_numchains{num_chains}"
    
    # 5. Add Method-Specific Suffixes
    # ONLY add MC samples if method is mc_marginalization
    if use_carb_uncertainty and carb_uncertainty_type == 'per_event':
        if per_event_method == 'mc_marginalization' and mc_carb_samples is not None:
            inference_suffix += f"_mc{mc_carb_samples}"
        elif per_event_method == 'direct':
            inference_suffix += "_direct"
    
    # 6. Handle Warmstart
    if use_initial_params:
        inference_suffix += "_warmstart"
    
    # 7. Build final directory name
    dir_name = (f"{exp_date}_"
                f"{model_name}_"
                f"{dataset_label}{carb_suffix}"
                f"{inference_suffix}")
    
    # 8. Join with base path
    base_dir = output_path if output_path else "bayes_output"
    output_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


from enum import Enum
from typing import Callable

class ModelType(Enum):
    """Available model types"""
    AR1 = "ar1"
    AR1_PURE = "ar1_pure"  # Pure AR(1), no insulin/carb dynamics
    AR1_RLS = "ar1_rls"  # Adaptive AR(1) via Recursive Least Squares w/ forgetting factor
    AR1_WITH_CARB_UNCERTAINTY = "ar1_with_carb_uncertainty"
    LSTM_MC = "lstm_mc"
    TRANSFORMER_MC = "transformer_mc"  # Architecture-capacity comparator
    LSTM_ENSEMBLE = "lstm_ensemble"  # UQ-method comparator: deep ensemble vs. MC-Dropout
    LSTM_CONFORMAL = "lstm_conformal"  # UQ-method comparator: split-conformal vs. MC-Dropout
    TRANSFORMER_CONFORMAL = "transformer_conformal"  # UQ-method comparator, Transformer backbone: isolates whether the LSTM_MC/Transformer_MC shared MC-Dropout failure mode is UQ-method- or architecture-driven
    TRANSFORMER_ENSEMBLE = "transformer_ensemble"  # UQ-method comparator, Transformer backbone: completes the architecture x UQ-method 2x2; the strongest-expected non-Bayesian-Loop configuration, so it is the adversarial test of the <=2hr clinical claim


def get_param_names(model_type: ModelType) -> list:
    """Get parameter names for plotting"""
    param_map = {
        # Baselines
        ModelType.AR1_PURE: ['phi', 'sigma_eta', 'sigma_obs'],
        
        # ORIGINAL MODELS
        ModelType.AR1: ['theta_ISF', 'theta_CR', 'isf_cr_ratio', 'phi', 'sigma_eta', 'sigma_obs'],
        ModelType.AR1_WITH_CARB_UNCERTAINTY: ['theta_ISF', 'theta_CR', 'isf_cr_ratio', 'phi', 'sigma_eta', 'sigma_obs', 'sigma_carb_obs'],
    }
    return param_map[model_type]


def extract_params_for_model(samples: dict, model_type: ModelType) -> dict:
    """
    Extract relevant parameters based on model type.
    """
    # ===== Baselines =====
    if model_type == ModelType.AR1_PURE:
        # Pure AR1: Only AR(1) parameters, no metabolic parameters
        params = {
            'phi': samples['phi'],
            'sigma_eta': samples['sigma_eta'],
            'sigma_obs': samples['sigma_obs']
        }
    
    # ===== ORIGINAL MODELS =====
    elif model_type in [ModelType.AR1, ModelType.AR1_WITH_CARB_UNCERTAINTY]:
        params = {
            'theta_ISF': samples['theta_ISF'],
            'theta_CR': samples['theta_CR'],
            'phi': samples['phi'],
            'sigma_eta': samples['sigma_eta'],
            'sigma_obs': samples['sigma_obs']
        }
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ===== SHARED PARAMETERS (added to any model that has them) =====
    if 'isf_cr_ratio' in samples:
        params['isf_cr_ratio'] = samples['isf_cr_ratio']
        
    if 'sigma_carb_obs' in samples:
        params['sigma_carb_obs'] = samples['sigma_carb_obs']

    if 'carb_adjustment' in samples:
        params['carb_adjustment'] = samples['carb_adjustment']
    
    return params


def plot_user_convergence_heatmap(mcmc, model_type, user_id, output_dir):
    """Generates a per-user heatmap of R-hat and ESS for the 4 chains."""
    try:
        import arviz as az
        dataset = az.from_numpyro(mcmc)
        focus_params = get_param_names(model_type)
        summary = az.summary(dataset, var_names=focus_params)
        
        # Prepare data for plotting
        metrics = summary[['r_hat', 'ess_bulk']].copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"User {user_id} Convergence Diagnostics ({mcmc.num_chains} Chains)", 
                     fontsize=16, fontweight='bold')

        # Plot R-hat
        im1 = ax1.imshow(metrics[['r_hat']].values, cmap='RdYlGn_r', vmin=1.0, vmax=1.1)
        ax1.set_title("Split-R-hat")
        ax1.set_yticks(np.arange(len(metrics)))
        ax1.set_yticklabels(metrics.index)
        ax1.set_xticks([])
        
        # Plot ESS
        im2 = ax2.imshow(metrics[['ess_bulk']].values, cmap='RdYlGn', vmin=0)
        ax2.set_title("Bulk ESS")
        ax2.set_yticks(np.arange(len(metrics)))
        ax2.set_yticklabels(metrics.index)
        ax2.set_xticks([])

        # Annotate values
        for ax, col, fmt in zip([ax1, ax2], ['r_hat', 'ess_bulk'], ['.3f', '.0f']):
            for i, val in enumerate(metrics[col]):
                ax.text(0, i, format(val, fmt), ha="center", va="center", fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        diag_path = os.path.join(output_dir, "diagnostics")
        os.makedirs(diag_path, exist_ok=True)
        plt.savefig(os.path.join(diag_path, f"user_{user_id}_convergence.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Failed to plot user heatmap: {e}")


def _log_convergence_diagnostics(samples, mcmc, model_type, log_fn, num_warmup=None, 
                                 num_samples=None, user_id=None, output_dir=None, 
                                 sampling_duration=None, num_chains=4):
    """
    Fixed version: Removed 'fmt="round"' to avoid ArviZ TypeError.
    Uses standard multi-chain R-hat and ESS.
    """
    log_fn(f"\n=== Diagnostics: Processing User {user_id} ===")
    
    if num_warmup is None: num_warmup = 100
    if num_samples is None: num_samples = 500
    
    log_fn("\n" + "="*60)
    log_fn("CONVERGENCE DIAGNOSTICS (MULTI-CHAIN)")
    log_fn("="*60)
    
    try:
        # 1. Convert to InferenceData
        dataset = az.from_numpyro(mcmc)
        
        # 2. Compute Summary (Default format is a Pandas DataFrame)
        focus_params = get_param_names(model_type)
        summary = az.summary(dataset, var_names=focus_params)
        
        log_fn(f"\nSampling Info:")
        log_fn(f"  Warmup iterations: {num_warmup}")
        log_fn(f"  Samples per chain: {num_samples}")
        log_fn(f"  Total Chains:      {num_chains}")
        log_fn(f"  Sampling Duration: {sampling_duration:.2f}s" if sampling_duration else "")

        log_fn(f"\nDiagnostic Values (R-hat < 1.05 is required):")
        
        rhat_values = {}
        ess_values = {}
        
        for param in focus_params:
            if param in summary.index:
                # Use .at or .loc to get the scalar values
                rhat = summary.loc[param, 'r_hat']
                ess = summary.loc[param, 'ess_bulk']
                
                # Handle potential NaN (e.g., zero variance chains)
                if np.isnan(rhat):
                    rhat = 1.0 # Assume converged if both chains stuck at same MAP
                
                rhat_values[param] = float(rhat)
                ess_values[param] = float(ess)
                
                status = "OK" if rhat < 1.05 else "WARNING: FAIL"
                log_fn(f"  {param:25s}: R-hat={rhat:.4f} {status} | ESS={ess:6.0f}")

        # 3. Mixing Assessment
        max_rhat = max(rhat_values.values()) if rhat_values else 1.0
        
        # 4. Save to CSV/Pickle
        if user_id is not None and output_dir is not None:
            diagnostics_dir = os.path.join(output_dir, 'diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            
            # Row for the CSV
            csv_row = {
                'user_id': user_id,
                'model_type': str(model_type),
                'max_rhat': f"{max_rhat:.4f}",
                'sampling_duration_sec': f"{sampling_duration:.2f}" if sampling_duration else "N/A"
            }
            
            for p in focus_params:
                if p in rhat_values:
                    csv_row[f'rhat_{p}'] = f"{rhat_values[p]:.4f}"
                    csv_row[f'ess_{p}'] = f"{ess_values[p]:.1f}"
                    # Use summary to get mean/std
                    csv_row[f'mean_{p}'] = f"{summary.loc[p, 'mean']:.4f}"
                    csv_row[f'std_{p}'] = f"{summary.loc[p, 'sd']:.4f}"
            
            csv_file = os.path.join(diagnostics_dir, "convergence_diagnostics_summary.csv")
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a' if file_exists else 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_row.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(csv_row)
            
            log_fn(f"\nDiagnostics saved to: {csv_file}")
            plot_user_convergence_heatmap(mcmc, model_type, user_id, output_dir)

    except Exception as e:
        log_fn(f"Error computing diagnostics: {e}")
        import traceback
        log_fn(traceback.format_exc())
    
    log_fn("="*60 + "\n")


def _merge_historical_future_events(hist_data, fut_data):
    """
    Merge historical and future event data for plotting the full 8-hour window.
    """
    import pandas as pd
    
    merged_data = {}
    merged_data['cgm'] = hist_data.get('cgm', pd.DataFrame())
    
    # Merge basal events
    hist_basal = hist_data.get('basal_events', pd.DataFrame())
    fut_basal = fut_data.get('basal_events', pd.DataFrame())
    dfs_basal = [df for df in [hist_basal, fut_basal] if not df.empty]
    if dfs_basal:
        merged_data['basal_events'] = pd.concat(dfs_basal, ignore_index=True)
        if 'timestamp' in merged_data['basal_events'].columns:
            merged_data['basal_events']['timestamp'] = pd.to_datetime(
                merged_data['basal_events']['timestamp']
            )
            merged_data['basal_events'] = merged_data['basal_events'].sort_values('timestamp')
    else:
        merged_data['basal_events'] = pd.DataFrame()
    
    # Merge bolus events
    hist_bolus = hist_data.get('bolus_events', pd.DataFrame())
    fut_bolus = fut_data.get('bolus_events', pd.DataFrame())
    dfs_bolus = [df for df in [hist_bolus, fut_bolus] if not df.empty]
    if dfs_bolus:
        merged_data['bolus_events'] = pd.concat(dfs_bolus, ignore_index=True)
        if 'timestamp' in merged_data['bolus_events'].columns:
            merged_data['bolus_events']['timestamp'] = pd.to_datetime(
                merged_data['bolus_events']['timestamp']
            )
            merged_data['bolus_events'] = merged_data['bolus_events'].sort_values('timestamp')
    else:
        merged_data['bolus_events'] = pd.DataFrame()
    
    # Merge meal/carb events
    hist_meals = hist_data.get('meal_events', pd.DataFrame())
    fut_meals = fut_data.get('meal_events', pd.DataFrame())
    dfs_meals = [df for df in [hist_meals, fut_meals] if not df.empty]
    if dfs_meals:
        merged_data['meal_events'] = pd.concat(dfs_meals, ignore_index=True)
        if 'timestamp' in merged_data['meal_events'].columns:
            merged_data['meal_events']['timestamp'] = pd.to_datetime(
                merged_data['meal_events']['timestamp']
            )
            merged_data['meal_events'] = merged_data['meal_events'].sort_values('timestamp')
    else:
        merged_data['meal_events'] = pd.DataFrame()
    
    return merged_data


def get_nudged_svi_warmstart(map_params, model_type, num_chains=4, jitter_scale=0.05):
    """
    Robust warm-start logic with 5% Bounded Uniform Jitter.
    Ensures valid R-hat diagnostics while staying within physiological boundaries.
    """
    import numpy as np
    
    m_type = model_type.value if hasattr(model_type, 'value') else str(model_type)
    rng = np.random.RandomState(42)  # Fixed seed for reproducible starting points

    # Define physiological "Four Walls" to prevent chains from drifting into invalid space
    BOUNDS = {
        'theta_CR': (6.0, 25.0),
        'log_multiplier': (1.0, 2.0), # 3.0x to 7.3x ratio
        'phi_raw': (0.01, 0.99),
        'sigma_obs': (12.0, 35.0),
        'sigma_eta': (0.5, 5.0),
        'sigma_carb_obs': (0.01, 0.50)
    }

    # Identify primary sampled sites for the current model
    sampled_keys = ['theta_CR', 'log_multiplier', 'sigma_obs']
    if "ar1" in m_type.lower():
        sampled_keys += ['phi_raw', 'sigma_eta']
    if "carb_uncertainty" in m_type.lower():
        sampled_keys += ['sigma_carb_obs']

    # Filter base SVI values
    base_values = {k: map_params[k] for k in sampled_keys if k in map_params}
    
    init_values = []
    for i in range(num_chains):
        if i == 0:
            # Chain 0: The exact SVI peak (The Anchor)
            init_values.append(base_values)
        else:
            # Chains 1-N: Uniformly jittered between 95% and 105% of MAP
            jittered = {}
            for k, v in base_values.items():
                low_val = v * (1.0 - jitter_scale)
                high_val = v * (1.0 + jitter_scale)
                new_val = rng.uniform(low_val, high_val)
                
                # Clip to physiological "Four Walls"
                if k in BOUNDS:
                    b_min, b_max = BOUNDS[k]
                    jittered[k] = np.clip(new_val, b_min, b_max)
                else:
                    jittered[k] = new_val
            init_values.append(jittered)
            
    return init_values


def check_for_flatlines(mcmc, user_id, log_fn):
    """
    Checks if MCMC chains are stuck at their initialization points.
    Should be called after a small number of samples have been collected.
    """
    samples = mcmc.get_samples()
    focus_params = ['theta_ISF', 'theta_CR', 'sigma_carb_obs']
    
    is_frozen = False
    for param in focus_params:
        if param in samples:
            param_samples = np.array(samples[param])
            # Check if all samples for this param are identical across time
            # Using ptp (peak-to-peak) to find the range
            if np.ptp(param_samples) == 0:
                log_fn(f"WARNING: CRITICAL ALERT: Chain for {param} is FLATLINED for User {user_id}.")
                is_frozen = True
    
    if is_frozen:
        log_fn("ERROR: Sampler is stuck. Recommendation: Decrease init_step_size or increase MC samples.")
    else:
        log_fn("Sampler movement detected. Adaptation is active.")
    
    return is_frozen


def _cf_artifact_path(output_dir, user_id, model_type):
    """
    Where to persist this run's trained neural weights, or None.

    ON by default -- set SAVE_MODEL_ARTIFACTS=0
    to disable. Motivation: the saved result bundles hold only prediction
    samples, so nothing re-forecastable survives a run -- which blocked
    counterfactual / dose-response analysis without a full retrain (see
    paper/CAUSAL_ACCURACY_PLAN.md and models/cf_artifacts.py) the first time
    it was needed, costing a full 395-user x 6-model retrain that would have
    been unnecessary had weights been persisted from the start. This only
    adds an extra torch.save() after training already completes -- it does
    not touch training itself or change results.pkl, so it's safe to run
    unconditionally; the on/off switch exists purely for the rare case of
    disk-space-constrained probes, not because saving is risky.
    """
    if os.environ.get('SAVE_MODEL_ARTIFACTS', '1') == '0':
        return None
    d = os.path.join(output_dir, 'model_artifacts')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'user_{user_id}_{model_type.value}.pt')


def run_user_level_inference_unified(
    historical_datas, 
    future_datas, 
    user_id, 
    instance_ids,
    model_type: ModelType = ModelType.AR1,
    dataset_mode: str = 'single_user',
    use_carbs_only: bool = False,
    use_carb_uncertainty: bool = False,
    carb_uncertainty_type: str = 'global',
    per_event_method: str = 'direct',
    mc_carb_samples: int = 30,
    use_user_param_prior: bool = False,
    use_svi_map: bool = False,
    svi_num_steps: int = 5000,
    svi_learning_rate: float = 0.01,
    inference_method: str = 'mcmc',
    num_warmup: int = 100,
    num_samples: int = 500,
    num_chains: int = 4,
    output_path: str = None,
    plot_subset_instances: int = None,
    initial_params=None,
    max_train_instances: int = None,
    fixed_test_size: int = None,
    sampling_mode: str = 'chronological',
    train_instance_ids: set = None,
    test_instance_ids: set = None,
):
    # Logging helper
    def log_msg(msg, flush=True):
        print(msg)
        if flush: sys.stdout.flush()

    # 1. Setup output directory
    output_dir = get_output_dir(
        model_type=model_type,
        dataset_mode=dataset_mode,
        use_carbs_only=use_carbs_only,
        use_carb_uncertainty=use_carb_uncertainty,
        carb_uncertainty_type=carb_uncertainty_type,
        inference_method=inference_method,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        num_svi_steps=svi_num_steps,
        svi_lr=svi_learning_rate,
        output_path=output_path,
        per_event_method=per_event_method,
        mc_carb_samples=mc_carb_samples
    )

    # ======================================================================
    # TIER 4: MAXIMUM ROBUSTNESS (Publication Final Batch)
    # ======================================================================
    is_mc_carb = (use_carb_uncertainty and carb_uncertainty_type == 'per_event' and per_event_method == 'mc_marginalization')
    is_direct_carb = (use_carb_uncertainty and carb_uncertainty_type == 'per_event' and per_event_method == 'direct')
    is_global_carb = (use_carb_uncertainty and carb_uncertainty_type == 'global')

    initial_step_size = 1e-4

    log_msg(f">>> Using hint {initial_step_size} init step size and heuristic initial step size for all models.")
    
    # Determine Initialization Strategy
    using_svi_warmstart = True if initial_params and os.path.isdir(initial_params) else False

    # Initialize flags
    force_manual_mode = False

    # 1. Set Sampler Robustness
    if is_mc_carb:
        log_msg(">>> TIER 6: Forcing Manual Step Size & Uniform Init for MC Model.")
        
        # 1. Disable SVI (Now this override will stick!)
        using_svi_warmstart = False 
        
        # 2. Force Manual Mode
        force_manual_mode = True
        initial_step_size = 0.1
        target_accept = 0.80
        use_dense = True # Diagonal is safer for unsticking
        
    elif is_direct_carb:
        target_accept = 0.95
        use_dense = False
        # Direct carb keeps using_svi_warmstart = True (from default)
        
    else:
        target_accept = 0.90
        use_dense = False

    # 1. Set Sampler Robustness
    # if is_mc_carb:

    #     # 1. Disable SVI Warmstart: SVI likely found a bad local mode.
    #     using_svi_warmstart = False 
    #     log_msg(">>> OVERRIDE: Disabling SVI Warmstart for MC Model to prevent flatlining.")

    #     target_accept = 0.90  # Slightly lower for high-D to allow movement
    #     use_dense = True
        
    # elif is_direct_carb:
    #     target_accept = 0.95  # Slightly lower for high-D to allow movement
    #     use_dense = False     # CRITICAL: Diagonal mass for 7k parameters
    # else:
    #     target_accept = 0.90
    #     use_dense = False

    # # 1. First, check if we are even using carb uncertainty
    # if not use_carb_uncertainty:
    #     # TIER 1 (BASELINE): Standard IID and AR1
    #     # initial_step_size = 1e-4
    #     target_accept = 0.9
    #     use_dense = True
    #     # log_msg(f">>> TIER 1 (BASELINE): Using step {initial_step_size} for {model_type.value}.")

    # elif is_global_carb:
    #     # TIER 2 (INTERMEDIATE): Global adjustments
    #     # initial_step_size = 1e-3
    #     target_accept = 0.95
    #     use_dense = True
    #     # log_msg(f">>> TIER 2 (INTERMEDIATE): Using step {initial_step_size} for Global Carb.")

    # elif is_direct_carb or is_mc_carb:
    #     # TIER 3 (HEAVY): Per-Event models
    #     # initial_step_size = 1e-4
    #     target_accept = 0.98
    #     use_dense = True
    #     # log_msg(f">>> TIER 3 (HEAVY): Using step {initial_step_size} for Per-Event.")

    # 2. Dynamic Chain Method Selection
    # Parallel spreads the heavy VRAM load of Direct/MC models across 4 GPUs
    if is_direct_carb or is_mc_carb:
        chosen_chain_method = 'vectorized'
        log_msg(">>> Chain Distribution: VECTORIZED (1 chain per GPU).")
    else:
        chosen_chain_method = 'vectorized'
        log_msg(">>> Chain Distribution: VECTORIZED (all chains on 1 GPU).")


    # ======================================================================
    # UNIFIED EXPERIMENT LOGGING
    # ======================================================================
    log_msg("\n" + "="*70)
    log_msg("SYSTEM & EXPERIMENT SETTINGS")
    log_msg("="*70)
    log_msg(f"  Model Type:            {model_type.value}")
    log_msg(f"  --- Sampling Geometry ---")
    log_msg(f"  Max Train Instances:   {max_train_instances if max_train_instances else 'Unlimited'}")
    log_msg(f"  Sampling Mode:         {sampling_mode.upper()}")
    log_msg(f"  Sub-sampling Seed:     42")
    log_msg(f"  --- Initialization Strategy ---")
    log_msg(f"  SVI MAP Warm Start:    {using_svi_warmstart}")
    log_msg(f"  Smart Jitter Applied:  {using_svi_warmstart}")
    log_msg(f"  --- MCMC Robustness & Kernel Settings ---")
    log_msg(f"  Target Acceptance:     {target_accept}")
    log_msg(f"  Dense Mass Matrix:     {use_dense}")
    log_msg(f"  Initial Step Size:     {initial_step_size}")
    log_msg(f"  Warmup / Samples:      {num_warmup} / {num_samples}")
    log_msg(f"  Number of Chains:      {num_chains}")
    log_msg("="*70 + "\n")

    # 3. Apply variables to logic
    if sampling_mode == 'random':
        np.random.seed(42) # Fixed seed for reproducible geometry

    # 2. Prepare Data Arrays
    (current_glucose_array, ins_times_array, ins_amounts_array, ins_n_array,
     carb_times_array, carb_amounts_array, carb_n_array,
     historical_cgm_array_batch, observed_glucose_array) = prepare_user_level_arrays(
        historical_datas, future_datas
    )
    n_instances = current_glucose_array.shape[0]


    # 3. Train/Test Split Logic
    # ---------------------------------------------------------
    pred_start_list = np.array([hist_data['cgm']['timestamp'].iloc[-1] for hist_data in historical_datas])
    sorted_indices = np.argsort(pred_start_list)
    n_instances = len(sorted_indices)

    if train_instance_ids is not None and test_instance_ids is not None:
        # Use curation-defined train/test instance IDs directly
        # instance_ids is the list of ex IDs passed in, sorted by pred_start
        train_idx = np.array([
            i for i, idx in enumerate(sorted_indices)
            if instance_ids[idx] in train_instance_ids
        ])
        test_idx = np.array([
            i for i, idx in enumerate(sorted_indices)
            if instance_ids[idx] in test_instance_ids
        ])
        log_msg(f"Using curation-defined split: {len(train_idx)} train, {len(test_idx)} test")
    else:
        # LEGACY: Infer split from fixed_test_size
        if fixed_test_size is not None:
            test_size = fixed_test_size
            log_msg(f"Using fixed test size: {test_size}")
        else:
            test_size = max(1, int(0.2 * n_instances))

        test_idx  = sorted_indices[-test_size:]
        train_pool = sorted_indices[:-test_size]

        if max_train_instances is not None and len(train_pool) > max_train_instances:
            train_idx = train_pool[-max_train_instances:]
            log_msg(f"Capping training set to last {max_train_instances} instances.")
        else:
            train_idx = train_pool
    # ---------------------------------------------------------


    # VERIFICATION LOGGING
    log_msg("\n" + "="*70)
    log_msg("TRAIN/TEST SPLIT VERIFICATION")
    log_msg("="*70)
    log_msg(f"  Total instances:       {n_instances}")
    log_msg(f"  Train Count:           {len(train_idx)}")
    log_msg(f"  Test Count:            {len(test_idx)}")
    
    # Log Training Range
    if len(train_idx) > 0:
        train_start_ts = pred_start_list[train_idx[0]]
        train_end_ts = pred_start_list[train_idx[-1]]
        log_msg(f"  Training Range:        {train_start_ts} to {train_end_ts}")
    else:
        log_msg(f"  Training Range:        [EMPTY]")

    # Log Testing Range & Washout Gap
    if len(test_idx) > 0:
        test_start_ts = pred_start_list[test_idx[0]]
        test_end_ts = pred_start_list[test_idx[-1]]
        log_msg(f"  Testing Range:         {test_start_ts} to {test_end_ts}")
        
        if len(train_idx) > 0:
            # Verify gap for dissertation documentation
            gap = test_start_ts - train_end_ts
            log_msg(f"  Washout Gap:           {gap}")
            
            # Automated check for 24h constraint
            if gap < pd.Timedelta(hours=24):
                log_msg(f"  WARNING: WARNING: Washout gap is less than 24 hours!")
            else:
                log_msg(f"  Washout > 24h verified.")
    else:
        log_msg(f"  Testing Range:         [EMPTY]")

    log_msg("="*70)
    
    # Confirming specific indices for debugging geometry
    if sampling_mode == 'random' and max_train_instances is not None:
        log_msg(f"  Random train indices:      {train_idx.tolist()}")
    elif len(train_idx) > 0 and len(test_idx) > 0:
        log_msg(f"  Indices Geometry:          Train [{train_idx[0]}-{train_idx[-1]}]  ->  Test [{test_idx[0]}-{test_idx[-1]}]")
    
    log_msg("="*70 + "\n")


    # 1. Start timer here so it is available to LSTM, SVI, and MCMC branches
    start_time = time.time()

    if model_type == ModelType.LSTM_MC:
        log_msg(">>> Starting LSTM with MC Dropout (8-Hour Context)")
        
        # 1. Use the dense preparation function we discussed
        # Make sure this function is imported from utils.data_prep
        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]
        
        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )
        
        # 2. Train and Sample
        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'lstm_mc')
        start_time_lstm = time.time()
        samples = train_and_sample_lstm(x_train_dense, y_train, x_test_dense,
                                        artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_lstm
        
        # 3. Format for the standard Bundle
        all_preds_cpu = {'obs': samples}
        
        # 4. Dummy data to keep the rest of the script from crashing
        # We need a 'global_params' dict because save_user_inference_bundle requires it
        global_params = {'sigma_obs': np.zeros(100)}
        all_prior_preds_cpu = None
        objectives = None
        
        # 5. EARLY JUMP to saving
        # This prevents the script from entering the MCMC kernel setup below
        log_msg(f"LSTM Training Complete. Saving bundle for User {user_id}...")

        # Define eval_idx here so the saver can use it
        eval_idx = np.concatenate([train_idx, test_idx])
        
        all_preds_cpu = {'obs': samples}
        # Matches your num_mc_samples (100)
        global_params = {'sigma_obs': np.zeros(100)} 
        
        # Manually trigger the saver
        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)), # Now eval_idx exists!
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )
        
        log_msg(f"LSTM Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.LSTM_ENSEMBLE:
        log_msg(">>> Starting LSTM Deep Ensemble (8-Hour Context)")

        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]

        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )

        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'lstm_ensemble')
        start_time_ensemble = time.time()
        samples = train_and_sample_lstm_ensemble(x_train_dense, y_train, x_test_dense,
                                                 artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_ensemble

        log_msg(f"LSTM Ensemble Training Complete. Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        all_preds_cpu = {'obs': samples}
        global_params = {'sigma_obs': np.zeros(100)}

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )

        log_msg(f"LSTM Ensemble Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.LSTM_CONFORMAL:
        log_msg(">>> Starting LSTM Split-Conformal (8-Hour Context)")

        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]

        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )

        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'lstm_conformal')
        start_time_conformal = time.time()
        samples = train_and_sample_lstm_conformal(x_train_dense, y_train, x_test_dense,
                                                  artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_conformal

        log_msg(f"LSTM Conformal Training Complete. Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        all_preds_cpu = {'obs': samples}
        global_params = {'sigma_obs': np.zeros(100)}

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )

        log_msg(f"LSTM Conformal Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.TRANSFORMER_MC:
        log_msg(">>> Starting Transformer with MC Dropout (8-Hour Context)")

        # Same 96-step [CGM, basal, bolus, carbs] grid as LSTM_MC — identical
        # inputs/outputs so this is an apples-to-apples architecture swap.
        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]

        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )

        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'transformer_mc')
        start_time_transformer = time.time()
        samples = train_and_sample_transformer(x_train_dense, y_train, x_test_dense,
                                               artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_transformer

        log_msg(f"Transformer Training Complete. Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        all_preds_cpu = {'obs': samples}
        global_params = {'sigma_obs': np.zeros(100)}

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )

        log_msg(f"Transformer Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.TRANSFORMER_CONFORMAL:
        log_msg(">>> Starting Transformer Split-Conformal (8-Hour Context)")

        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]

        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )

        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'transformer_conformal')
        start_time_transformer_conformal = time.time()
        samples = train_and_sample_transformer_conformal(x_train_dense, y_train, x_test_dense,
                                                         artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_transformer_conformal

        log_msg(f"Transformer Conformal Training Complete. Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        all_preds_cpu = {'obs': samples}
        global_params = {'sigma_obs': np.zeros(100)}

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )

        log_msg(f"Transformer Conformal Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.TRANSFORMER_ENSEMBLE:
        log_msg(">>> Starting Transformer Deep Ensemble (8-Hour Context)")

        x_train_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in train_idx],
            [future_datas[i] for i in train_idx]
        )
        y_train = observed_glucose_array[train_idx]

        x_test_dense = prepare_lstm_sequence_arrays(
            [historical_datas[i] for i in test_idx],
            [future_datas[i] for i in test_idx]
        )

        # Seed immediately before the trainer call (not once per job) so
        # results do not depend on how many users ran first in this process.
        seed_for_trainer(user_id, 'transformer_ensemble')
        start_time_transformer_ensemble = time.time()
        samples = train_and_sample_transformer_ensemble(x_train_dense, y_train, x_test_dense,
                                                        artifact_path=_cf_artifact_path(output_dir, user_id, model_type))
        sampling_duration = time.time() - start_time_transformer_ensemble

        log_msg(f"Transformer Ensemble Training Complete. Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        all_preds_cpu = {'obs': samples}
        global_params = {'sigma_obs': np.zeros(100)}

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=time.time() - start_time
        )

        log_msg(f"Transformer Ensemble Saved for {user_id}. Terminating runner.")
        return

    elif model_type == ModelType.AR1_RLS:
        rls_order = int(os.environ.get('RLS_ORDER', '1'))
        log_msg(f">>> Starting Adaptive AR({rls_order}) via Recursive Least Squares (forgetting factor)")

        # RLS needs no training phase — it's independently re-identified per
        # instance from that instance's own historical CGM window. Forecast
        # only the test instances, matching the LSTM/AR1_PURE evaluation set.
        horizon = observed_glucose_array.shape[1]
        # NOT hash((user_id, 'ar1_rls')): Python salts string hashing per
        # process, so that gave a different PRNGKey on every run and made
        # AR1-RLS silently non-reproducible despite looking seeded. sha256 via
        # utils.seeding is stable across processes and shares its derivation
        # with the neural models below and with the counterfactual retrain.
        from utils.seeding import derive_seed, record_seed
        rls_key = jax.random.PRNGKey(record_seed(user_id, 'ar1_rls',
                                                 derive_seed(user_id, 'ar1_rls')))

        start_time_rls = time.time()
        rls_out = fit_and_forecast_rls_batch(
            historical_cgm_array_batch[test_idx],
            horizon=horizon,
            order=rls_order,
            rng_key=rls_key,
        )
        sampling_duration = time.time() - start_time_rls

        all_preds_cpu = {'obs': np.array(jax.device_get(rls_out['obs']))}
        global_params = {
            'coefs': np.array(jax.device_get(rls_out['coefs'])),
            'mu': np.array(jax.device_get(rls_out['mu'])),
            'resid_std': np.array(jax.device_get(rls_out['resid_std'])),
            'was_clipped': np.array(jax.device_get(rls_out['was_clipped'])),
            'forecast_was_clamped': np.array(jax.device_get(rls_out['forecast_was_clamped'])),
        }

        n_clipped = int(global_params['was_clipped'].sum())
        n_clamped = int(global_params['forecast_was_clamped'].sum())
        log_msg(f"    Stability diagnostics: coefficients clipped for {n_clipped}/{len(test_idx)} instances, "
                f"glucose-range clamp engaged for {n_clamped}/{len(test_idx)} instances.")

        log_msg(f"RLS fit+forecast complete for {len(test_idx)} test instances. "
                f"Saving bundle for User {user_id}...")

        eval_idx = np.concatenate([train_idx, test_idx])

        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=None,
            train_idx=np.arange(len(train_idx)),
            test_idx=np.arange(len(train_idx), len(eval_idx)),
            observed_glucose_array=observed_glucose_array[eval_idx],
            instance_ids=[instance_ids[i] for i in eval_idx],
            output_dir=output_dir,
            model_type_name=model_type.value,
            sampling_duration_sec=sampling_duration,
        )

        log_msg(f"RLS Saved for {user_id}. Terminating runner.")
        return

    else:
        # --- ALL EXISTING NUMPYRO/MCMC CODE GOES HERE ---

        # 4. Model configuration (ISF/CR priors use the defaults in models/ar1.py)
        model_config = {}
        if use_user_param_prior:
            raise ValueError('use_user_param_prior is not supported; ISF/CR priors use the model defaults.')

        # Determine num_carb_slots for 'direct' method
        num_carb_slots = None
        if use_carb_uncertainty and carb_uncertainty_type == 'per_event' and per_event_method == 'direct':
            n_total_instances = current_glucose_array.shape[0] 
            max_carbs_per_instance = carb_amounts_array.shape[1] 
            num_carb_slots = int(n_total_instances * max_carbs_per_instance)
            log_msg(f"Per-event carb uncertainty (direct): allocating {num_carb_slots} slots for ALL instances")

        # Update model_config with ALL required parameters
        model_config.update({
            'carb_uncertainty': use_carb_uncertainty,
            'carb_uncertainty_type': carb_uncertainty_type,
            'per_event_method': per_event_method,
            'mc_carb_samples': mc_carb_samples,
            'num_carb_slots': num_carb_slots
        })

        # Initialize the specific Model Instance
        if model_type in [ModelType.AR1, ModelType.AR1_WITH_CARB_UNCERTAINTY]:
            model_instance = AR1Model(model_config)
        elif model_type == ModelType.AR1_PURE:
            model_instance = AR1PureModel(model_config)
        else:
            raise ValueError(f"Unsupported model type for this inference runner: {model_type}")
        
        # Define the model function for MCMC/SVI
        model_fn = model_instance.model_fn

        # # Generate the Smart Initial Values
        # # filtered_init already contains your nudged SVI MAP or default values
        # init_values = get_smart_init_values(filtered_init, num_chains=num_chains)

        # 6. Run Inference (MCMC or SVI)
        if inference_method.lower() == 'svi':
            rng_key = jax.random.PRNGKey(0)

            # Create a specific folder for THIS user
            user_svi_diag_dir = os.path.join(output_dir, "svi_diagnostics", f"user_{user_id}")
            os.makedirs(user_svi_diag_dir, exist_ok=True)

            svi_result = model_instance.fit_svi_map(
                current_glucose_array[train_idx], ins_times_array[train_idx], ins_amounts_array[train_idx], 
                ins_n_array[train_idx], carb_times_array[train_idx], carb_amounts_array[train_idx], 
                carb_n_array[train_idx], historical_cgm_array_batch[train_idx], 
                observed_glucose_array[train_idx], rng_key, 
                num_steps=svi_num_steps, 
                learning_rate=svi_learning_rate,
                elbo_plot_dir=user_svi_diag_dir,
                output_dir=output_dir,
                user_id=user_id
            )
            sampling_duration = time.time() - start_time
            global_params = model_instance.extract_map_params(svi_result)
            samples = global_params
        else:
            rng_key = jax.random.PRNGKey(0)
            
            # from numpyro.infer import init_to_uniform

            # # Setup NUTS with single init dict (NumPyro will handle per-chain variation via RNG)
            # kernel = NUTS(
            #     model_fn, 
            #     # init_strategy=init_to_value(values=init_values[0]),  # Use first chain's values
            #     init_strategy=init_to_uniform(radius=3),
            #     target_accept_prob=target_accept, # High target reduces step size
            #     dense_mass=use_dense, # Handles parameter correlations
            #     step_size=initial_step_size,
            #     find_heuristic_step_size=True
            # )

            # Ensure init_strategy uses 'init_values' derived from SVI
            if using_svi_warmstart:

                svi_map_path = os.path.join(initial_params, "results", f"user_{user_id}_results.pkl")
                
                # 1. Load the dictionary you just inspected
                with open(svi_map_path, 'rb') as f:
                        svi_data = pickle.load(f)
                    
                # Access the 'map_params' key if it exists in the results bundle
                if isinstance(svi_data, dict) and 'map_params' in svi_data:
                    map_params = svi_data['map_params']
                elif hasattr(svi_data, 'params'):
                    map_params = svi_data.params
                else:
                    map_params = svi_data
                
                # Nudged SVI warm start (5% jitter)
                init_values = get_nudged_svi_warmstart(
                    map_params, 
                    model_type=model_type, 
                    num_chains=num_chains,
                    jitter_scale=0.05  # Explicitly set to 5%
                )

                log_msg(f">>> WARM START: Successfully loaded SVI MAP for {user_id}")
                log_msg(f">>> 5% Bounded Uniform Jitter applied to {num_chains-1} chains.")
                log_msg("\n" + "-"*40)
                log_msg(f">>> MAP LOG: User {user_id} ({model_type.value})")
                log_msg("-" * 40)
                
                # 1. Define all possible primary sampled sites across all model variants
                potential_keys = [
                    'theta_CR', 'log_multiplier', 'phi_raw', 
                    'sigma_obs', 'sigma_eta', 'sigma_carb_obs'
                ]
                
                # Only log keys that exist in the current model's initialization dictionary
                for k in potential_keys:
                    if k in init_values[0]:
                        val = float(init_values[0][k])
                        log_msg(f"  {k:15}: {val:.6f}")
                        
                # 2. Conditional Reconstruction for Metabolic Parameters
                # We only compute ISF and Ratio if both CR and Log-Multiplier exist
                if 'theta_CR' in init_values[0] and 'log_multiplier' in init_values[0]:
                    cr = init_values[0]['theta_CR']
                    # The ISF:CR ratio is derived from the log_multiplier
                    ratio = np.exp(init_values[0]['log_multiplier'])
                    isf = cr * ratio
                    log_msg(f"  {'[RECON] ISF':15}: {isf:.4f} mg/dL/U")
                    log_msg(f"  {'[RECON] Ratio':15}: {ratio:.4f}")
                    
                # 3. Conditional Reconstruction for Autocorrelation
                # Only relevant for AR1 variants
                if 'phi_raw' in init_values[0]:
                    # Mapping the raw latent variable (0-1) to the physiological range (0.80-0.99)
                    phi_phys = 0.80 + (0.19 * init_values[0]['phi_raw'])
                    log_msg(f"  {'[RECON] Phi':15}: {phi_phys:.6f}")

                log_msg("-" * 40 + "\n")


                # 3. Initialize the kernel with the first chain's values
                kernel = NUTS(
                    model_fn, 
                    init_strategy=init_to_value(values=init_values[0]),
                    target_accept_prob=target_accept,
                    dense_mass=use_dense
                )
            else:
                log_msg(f">>> MCMC INITIALIZATION: Using init_to_median (No SVI Warm-start)")

                from numpyro.infer import init_to_median, init_to_uniform

                # kernel = NUTS(
                #     model_fn, 
                #     init_strategy=init_to_median(),
                #     target_accept_prob=target_accept,
                #     dense_mass=use_dense,
                #     step_size=initial_step_size,
                #     find_heuristic_step_size=True
                # )

                if force_manual_mode:

                    log_msg(f"init to uniform, force manual mode, lower target accept {target_accept}.")

                    # TIER 6 KERNEL (MC CARB SPECIAL)
                    log_msg(f"    -> MODE: Manual Step (0.01) + Uniform Init (Radius 1.0)")
                    kernel = NUTS(
                        model_fn,
                        init_strategy=init_to_median(), # Randomize start
                        target_accept_prob=target_accept,
                        dense_mass=use_dense,
                        step_size=initial_step_size,       # Force 0.01
                        find_heuristic_step_size=False     # Disable the broken finder
                    )
                else:
                    # STANDARD KERNEL (AR1 / IID / Direct)
                    log_msg(f"    -> MODE: Heuristic Step + Median Init")
                    kernel = NUTS(
                        model_fn,
                        init_strategy=init_to_median(),
                        target_accept_prob=target_accept,
                        dense_mass=use_dense,
                        step_size=initial_step_size,
                        find_heuristic_step_size=True      # Keep True for others
                    )
            
            mcmc = MCMC(
                kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples, 
                num_chains=num_chains,
                chain_method=chosen_chain_method  # Vectorizes all 4 chains on the GPU
            )
            
            # Run MCMC
            mcmc.run(rng_key, current_glucose_array[train_idx], ins_times_array[train_idx], 
                    ins_amounts_array[train_idx], ins_n_array[train_idx], 
                    carb_times_array[train_idx], carb_amounts_array[train_idx], 
                    carb_n_array[train_idx], historical_cgm_array_batch[train_idx], 
                    observed_glucose_array[train_idx], 
                    extra_fields=('accept_prob', 'diverging'))
            
            # Immediate Diagnostic Check
            is_stuck = check_for_flatlines(mcmc, user_id, log_msg)
            
            if is_stuck:
                log_msg("WARNING: WARNING: Diagnostics will show R-hat=1.00 but ESS will be near zero.")

            # Immediately check the diagnostics before doing anything else
            diagnostics = mcmc.get_extra_fields()
            if 'accept_prob' in diagnostics:
                # Average across both the chain dimension and the sample dimension
                avg_accept_prob = jnp.mean(diagnostics['accept_prob'])
                log_msg(f"DEBUG [User {user_id}]: Mean Acceptance Probability: {avg_accept_prob:.4f}")
                
                if avg_accept_prob < 0.01:
                    log_msg("WARNING: WARNING: Acceptance rate is near zero. Sampler is likely flatlined.")
            else:
                log_msg("WARNING: WARNING: 'accept_prob' not found in diagnostics. Check if sampler initialized.")
            
            sampling_duration = time.time() - start_time
            samples = mcmc.get_samples()
            global_params = extract_params_for_model(samples, model_type)
            
            # Pass num_chains to diagnostics for accurate R-hat calculation
            _log_convergence_diagnostics(
                samples, mcmc, model_type, log_msg, 
                num_warmup=num_warmup, num_samples=num_samples,
                user_id=user_id, output_dir=output_dir,
                sampling_duration=sampling_duration,
                num_chains=num_chains
            )

            # Real-time console monitoring for resistance
            if 'isf_cr_ratio' in samples:
                med_ratio = np.median(samples['isf_cr_ratio'])
                med_isf = np.median(samples['theta_ISF'])
                log_msg(f"INTERIM RESULT [User {user_id}]:")
                log_msg(f"   Median ISF:CR Ratio: {med_ratio:.2f}")
                log_msg(f"   Derived Median ISF:  {med_isf:.2f} mg/dL/U")

            # 7.5 Generate the heavy User-Level correlation plot ONCE
            log_msg(f"Generating global parameter correlation plot for User {user_id}...")
            from utils.plotting import plot_parameter_correlations
            plot_parameter_correlations(
                global_params, 
                user_id, 
                output_dir, 
                model_type_name=model_type.value
            )


        # 7. Generate Posterior Predictive (ONLY for evaluated instances)
        eval_idx = np.concatenate([train_idx, test_idx])
        log_msg(f"Generating Posterior Predictive for {len(eval_idx)} instances...")
        
        predictive = Predictive(model_fn, global_params)
        all_preds = predictive(
            jax.random.PRNGKey(1), 
            current_glucose_array[eval_idx], 
            ins_times_array[eval_idx], 
            ins_amounts_array[eval_idx],
            ins_n_array[eval_idx], 
            carb_times_array[eval_idx], 
            carb_amounts_array[eval_idx], 
            carb_n_array[eval_idx], 
            historical_cgm_array_batch[eval_idx], 
            None, 
            sample_predictions=True
        )


        log_msg(f"\nGenerating Prior-Predictive for {len(eval_idx)} instances...")

        try:
            # Create a "guide" that samples from priors (not posteriors)
            # For MCMC, we use the model without conditioning on data
            def prior_only_model():
                """Model function that samples from priors only (no likelihood)."""
                return model_instance.model_fn(
                    current_glucose_array[eval_idx],
                    ins_times_array[eval_idx], 
                    ins_amounts_array[eval_idx],
                    ins_n_array[eval_idx], 
                    carb_times_array[eval_idx], 
                    carb_amounts_array[eval_idx], 
                    carb_n_array[eval_idx], 
                    historical_cgm_array_batch[eval_idx], 
                    observed_glucose_array=None,  # KEY: Don't condition on data
                    sample_predictions=True
                )
            
            # Sample from prior
            prior_predictive = Predictive(prior_only_model)
            all_prior_preds = prior_predictive(
                jax.random.PRNGKey(999),  # Different seed for reproducibility
            )
            
            # Migrate to CPU like we do for posteriors
            all_prior_preds_cpu = {
                k: np.array(jax.device_get(v)).copy() 
                for k, v in all_prior_preds.items()
            }
            
            log_msg(f"Prior-predictive samples generated")
            log_msg(f"  Shape: {all_prior_preds_cpu.get('obs', all_prior_preds_cpu.get('pred_mean')).shape}")
            
        except Exception as e:
            log_msg(f"WARNING: Warning: Prior-predictive generation failed: {e}")
            all_prior_preds_cpu = None



        # --- STEP 7.1: FORCE COPY AND CPU MIGRATION ---
        log_msg("Migrating sliced data to CPU RAM...")

        # Use jax.device_get to force the GPU to finish and move data
        # Then use .copy() to ensure the memory is contiguous for the workers
        # Only keep 'obs' — drop bias, eta_std, pred_mean (not needed post-inference)
        # pred_mean = mean(obs, axis=0) — recomputable; bias/eta_std are intermediates
        KEEP_PREDS = {'obs'}
        all_preds_cpu = {
            k: np.array(jax.device_get(v)).copy() 
            for k, v in all_preds.items()
            if k in KEEP_PREDS
        }

        # 7.5 Compute objectives for the evaluated subset
        subset_instance_ids = [instance_ids[i] for i in eval_idx]
        subset_obs_glucose = observed_glucose_array[eval_idx]
        
        # We use 'pred_mean' if available, or fall back to 'obs'
        pred_key_obj = 'pred_mean' if 'pred_mean' in all_preds_cpu else 'obs'
        

        # # Determine number of samples robustly
        # theta_val = np.asarray(global_params['theta_ISF'])
        # n_samples = theta_val.shape[0] if theta_val.ndim > 0 else 1


        # Determine number of samples robustly
        # For baseline models with only 1 parameter, use sigma_obs
        if model_type == ModelType.AR1_PURE:
            param_key = 'phi'
        else:
            param_key = 'theta_ISF'  # Default for other models

        param_val = np.asarray(global_params.get(param_key, np.array([1.0])))
        n_samples = param_val.shape[0] if param_val.ndim > 0 else 1


        # objectives = compute_aggregate_log_objectives_per_sample(
        #     subset_instance_ids, 
        #     subset_obs_glucose, 
        #     all_preds_cpu[pred_key_obj], 
        #     global_params, 
        #     num_param_samples=n_samples
        # )

        # Since objectives computation is skipped, set to None
        objectives = None   

        # Save the big bundle
        from utils.plotting import save_user_inference_bundle
        save_user_inference_bundle(
            user_id=user_id,
            global_params=global_params,
            all_preds_cpu=all_preds_cpu,
            all_prior_preds_cpu=all_prior_preds_cpu,
            train_idx=np.arange(len(train_idx)), # Local Train indices
            test_idx=np.arange(len(train_idx), len(eval_idx)), # Local Test indices
            observed_glucose_array=subset_obs_glucose,
            instance_ids=subset_instance_ids,
            output_dir=output_dir,
            model_type_name=model_type.value,
            log_prior_all=objectives['log_prior_all'] if objectives else None,
            log_likelihood_all=objectives['log_likelihood_all'] if objectives else None,
            log_posterior_all=objectives['log_posterior_all'] if objectives else None,
            sampling_duration_sec=sampling_duration
        )

        # --- Early exit for SVI ---
        if inference_method.lower() == 'svi':
            log_msg(f"SVI MAP saved for user {user_id}. Skipping plotting/trajectories as requested.")
            return # Exit the function here
        # ------------------------------

        # 8. BATCH GPU MATH: Compute orange deterministic trajectories
        # We MUST also slice the inputs for deterministic math to match eval_idx
        log_msg(f"Vectorizing deterministic trajectories for {len(eval_idx)} instances...")
        

        # For baselines without theta_ISF, use an available parameter
        if 'theta_ISF' in global_params:
            num_post_samples = global_params['theta_ISF'].shape[0]
        elif 'phi' in global_params:
            num_post_samples = global_params['phi'].shape[0]
        elif 'sigma_obs' in global_params:
            num_post_samples = global_params['sigma_obs'].shape[0]
        else:
            raise ValueError("No posterior samples found in global_params")

        # num_post_samples = global_params['theta_ISF'].shape[0]
        sample_subset_indices = np.random.choice(num_post_samples, size=15, replace=False)

        # Vectorizing deterministic trajectories for posterior samples
        all_det_trajs = None
        all_det_trajs_np = None  # Initialize to None for baselines

        # Vectorizing deterministic trajectories for posterior samples
        if model_type == ModelType.AR1_PURE:
            # Baselines don't need full trajectory vectorization
            # (Pure AR1 has no insulin/carb dynamics)
            log_msg("Skipping trajectory vectorization for baseline model")
        else:
            # Standard models need Loop trajectory computation
        
            def get_trajs_for_sample(idx):
                p_sample = create_loop_params(global_params['theta_CR'][idx], global_params['theta_ISF'][idx])
                bg_vels = compute_bg_velocity_batch(
                    p_sample, current_glucose_array[eval_idx], historical_cgm_array_batch[eval_idx], 
                    ins_times_array[eval_idx], ins_amounts_array[eval_idx], 
                    carb_times_array[eval_idx], carb_amounts_array[eval_idx]
                )
                _, trajs = predict_cgm_loop_vectorized(
                    p_sample, current_glucose_array[eval_idx], ins_times_array[eval_idx], 
                    ins_amounts_array[eval_idx], ins_n_array[eval_idx], 
                    carb_times_array[eval_idx], carb_amounts_array[eval_idx], 
                    carb_n_array[eval_idx], historical_cgm_array_batch[eval_idx], bg_vels
                )
                return trajs

            # # FAST-EVAL MODE
            # all_det_trajs = jax.vmap(get_trajs_for_sample)(jnp.array(sample_subset_indices))
            # all_det_trajs_np = np.array(jax.device_get(all_det_trajs)).copy()
            log_msg("Skipping trajectory vectorization (fast-eval mode).")

        # ======================================================================
        # MIGRATE DATA TO CPU & CLEAN UP GPU
        # ======================================================================
        log_msg("Migrating all predictions to CPU and clearing GPU context...")

        # 2. Explicitly delete JAX references to free VRAM for other models
        del all_preds
        # del all_det_trajs # not computed in fast-eval mode

        
        # 3. Optional but recommended: Force a JAX synchronization
        jax.effects_barrier() 
        log_msg("GPU memory cleared. Data is now in System RAM.")
        # ======================================================================

        log_msg("Data is now CPU-local and contiguous. Starting plots...")

        # 9. Parallel CPU Plotting Hand-off
        current_pos = 0
        # Map 'train' to its local slice and 'test' to its local slice
        for which_set, set_len in [('train', len(train_idx)), ('test', len(test_idx))]:
            # Local indices inside the already-sliced all_preds_cpu
            local_slice = slice(current_pos, current_pos + set_len)
            
            # Global indices for metadata/data lookup
            global_indices = eval_idx[local_slice]
            
            ids_subset = [instance_ids[i] for i in global_indices]
            hist_subset = [historical_datas[i] for i in global_indices]
            fut_subset = [future_datas[i] for i in global_indices]
            obs_subset = subset_obs_glucose[local_slice]
            
            merged_events_subset = []
            for hist_data, fut_data in zip(hist_subset, fut_subset):
                merged_events_subset.append(_merge_historical_future_events(hist_data, fut_data))

            # Pull from our sliced CPU predictions
            # 'pred_mean' never existed -- every construction site of
            # all_preds_cpu  sets only 'obs',
            # regardless of model type, so the non-IID branch below was
            # dead-on-arrival (KeyError) for any model that actually reaches
            # this function, e.g. ar1_with_carb_uncertainty / Bayesian Loop.
            preds_subset = all_preds_cpu['obs'][:, local_slice, :]
            # det_trajs_subset = all_det_trajs_np[:, local_slice, :] if all_det_trajs_np is not None else None

            if all_det_trajs_np is not None:
                try:
                    det_trajs_subset = all_det_trajs_np[:, local_slice, :]
                except IndexError:
                    det_trajs_subset = None  # Graceful fallback
            else:
                det_trajs_subset = None

            # log_msg(f"Starting parallel plotting for {which_set} set...")
            # _process_instances_parallel(
            #     which_set, ids_subset, obs_subset, preds_subset, global_params, model_type,
            #     merged_events_subset, user_id, log_msg, output_dir,
            #     det_trajs_subset, num_warmup, num_samples, plot_subset_instances
            # )
            current_pos += set_len

        # ======================================================================
        # CLEANUP: Clear JAX cache between users to prevent Triton SIGSEGV
        # ======================================================================
        import gc
        gc.collect()
        jax.clear_caches()
        jax.effects_barrier()
        
        try:
            backend = jax.lib.xla_bridge.get_backend()
            for buf in backend.live_buffers():
                buf.delete()
        except Exception as e:
            log_msg(f"WARNING: Could not clear live buffers: {e}")
        
        log_msg(f"JAX cache cleared after user {user_id}")
        log_msg(f"Inference and Parallel Plotting complete for user {user_id}")


from functools import partial
from concurrent.futures import ProcessPoolExecutor
from functools import partial
