"""
Saving of per-user results bundles (save_user_inference_bundle) and
per-user posterior diagnostic plots.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import arviz as az
import csv


def _seed_provenance(user_id, model_type_name):
    """
    Seed + numeric environment for this (user, model), for the result bundle.

    Fails soft on purpose: this is provenance metadata, so a missing torch (JAX
    -only runs) or an import problem must never abort a completed inference run
    that took GPU-hours to produce.
    """
    try:
        from utils.seeding import applied_seed, env_provenance, SEED_BASE
        return {'seed': applied_seed(user_id, model_type_name),
                'seed_base': SEED_BASE,
                'env_provenance': env_provenance()}
    except Exception as e:                                    # pragma: no cover
        return {'seed': None, 'seed_base': None,
                'env_provenance': {'error': repr(e)}}


def save_user_inference_bundle(
    user_id, global_params, all_preds_cpu, all_prior_preds_cpu,train_idx, test_idx, 
    observed_glucose_array, instance_ids, output_dir, model_type_name,
    log_prior_all=None, log_likelihood_all=None, log_posterior_all=None,
    sampling_duration_sec=None
):
    import os
    import pickle
    import numpy as np

    save_dir = os.path.join(output_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    # Find the MAP index based on the highest log-posterior
    map_idx = np.argmax(log_posterior_all) if log_posterior_all is not None else None

    # 1. First, compute the map_params dict robustly outside the save_dict
    if map_idx is not None:
        robust_map_params = {}
        for k, v in global_params.items():
            v_arr = np.asarray(v)
            robust_map_params[k] = v_arr.item() if v_arr.ndim == 0 else v_arr[map_idx]
    else:
        # For SVI: global_params IS the MAP — scalars already, store directly
        robust_map_params = {k: np.asarray(v).item() if np.asarray(v).ndim == 0 else np.asarray(v).tolist()
                            for k, v in global_params.items()}
    
    bundle = {
        'user_id': user_id,
        'model_type': model_type_name,
        'param_samples': {k: np.asarray(v) for k, v in global_params.items()},
        'all_preds': {k: np.asarray(v) for k, v in all_preds_cpu.items() if k in {'obs'}},
        'all_prior_preds': {k: np.asarray(v) for k, v in all_prior_preds_cpu.items()} if all_prior_preds_cpu else None,
        'obs_glucose': np.asarray(observed_glucose_array),
        'instance_ids': instance_ids,
        'train_indices': np.asarray(train_idx),
        'test_indices': np.asarray(test_idx),
        
        # Save the full distribution of objectives for UQ analysis
        'log_prior_all': np.asarray(log_prior_all) if log_prior_all is not None else None,
        'log_likelihood_all': np.asarray(log_likelihood_all) if log_likelihood_all is not None else None,
        'log_posterior_all': np.asarray(log_posterior_all) if log_posterior_all is not None else None,
        # Explicit MAP metrics
        # 'map_params': {k: np.asarray(v)[map_idx] for k, v in global_params.items()} if map_idx is not None else None,
        'map_params': robust_map_params,
        'map_log_prior': float(log_prior_all[map_idx]) if map_idx is not None else None,
        'map_log_likelihood': float(log_likelihood_all[map_idx]) if map_idx is not None else None,
        'map_log_posterior': float(log_posterior_all[map_idx]) if map_idx is not None else None,
        'sampling_duration_sec': sampling_duration_sec,
        # Reproducibility provenance
        # Recorded in the bundle, not just the job log: a seed printed to a log
        # is not reproducibility, because the log and the bundle get separated.
        # `seed` is None for Bayesian Loop / Pure AR1, which are deterministic
        # via fixed JAX PRNGKeys and never go through utils.seeding.
        **_seed_provenance(user_id, model_type_name),
    }

    file_path = f"{save_dir}/user_{user_id}_results.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Unified bundle saved with MAP metrics: {file_path}")


def plot_parameter_correlations(samples, user_id, output_dir, model_type_name="model"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import os
    import numpy as np
    import arviz as az

    # 1. Define the variables
    candidates = ['theta_ISF', 'theta_CR', 'isf_cr_ratio', 'phi', 'sigma_eta', 'sigma_obs', 'sigma_carb_obs']
    valid_samples = {}
    
    for name in candidates:
        if name in samples:
            arr = np.asarray(samples[name]).flatten()
            # Safety: Add tiny jitter (1e-6) if parameter is constant to prevent KDE crash
            if np.std(arr) < 1e-9:
                arr = arr + np.random.normal(0, 1e-6, size=arr.shape)
            valid_samples[name] = arr[np.newaxis, :] # Add chain dimension for ArviZ

    if len(valid_samples) < 2:
        return

    # 2. Create InferenceData
    data = az.from_dict(posterior=valid_samples)
    fig_size = len(valid_samples) * 3

    try:
        # 3. Generate KDE Plot
        axes = az.plot_pair(
            data,
            kind='kde',
            marginals=True,
            figsize=(fig_size, fig_size),
            kde_kwargs={"contourf_kwargs": {"alpha": 0.5}}
        )

        # 4. Remove Scientific Notation and Format
        fig = plt.gcf()
        for ax in fig.get_axes():
            # Force decimal notation
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            # Rotate labels to prevent overlap with decimals
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

        plt.suptitle(f"User {user_id} Parameter Correlations ({model_type_name})", 
                     fontweight='bold', fontsize=20, y=1.02)

        # 5. Save
        save_path = f"{output_dir}/plots/user_{user_id}/parameter_correlations.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"KDE Correlation Plot (Decimal) saved: {save_path}")

    except Exception as e:
        print(f"[CORR_PLOT] KDE failed despite safety: {e}")
    finally:
        plt.close('all')