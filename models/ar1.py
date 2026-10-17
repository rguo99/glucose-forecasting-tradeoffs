"""
Bayesian Loop: Loop-style insulin/carbohydrate dynamics (models/loop_model.py)
with an AR(1) residual process, per-user ISF and CR, and a global
carbohydrate-reporting error term (AR1Model). Fitted with NUTS (numpyro).
The PRIOR_SCALE environment variable sets the log-space scale of the CR and
ISF/CR-ratio priors (0.30 main analysis; 0.10 and 0.50 for the prior
sensitivity analysis).
"""
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
from .loop_model import (
    create_loop_params,
    predict_cgm_loop_vectorized,
    compute_bg_velocity_batch
)
import matplotlib.pyplot as plt
import numpy as np
import os
import optax # for gradient clipping
import pickle


import os as _os

# Read prior scale from environment variable (default 0.30 = main results)
# Set PRIOR_SCALE=0.10 for tighter sensitivity, 0.50 for looser sensitivity
_PRIOR_SCALE = float(_os.environ.get('PRIOR_SCALE', '0.30'))
sensitivity_analysis = _PRIOR_SCALE >= 0.50   # legacy flag for 0.70 scale
tighter_priors       = _PRIOR_SCALE <= 0.10   # legacy flag for 0.10 scale
_CUSTOM_SCALE        = not (sensitivity_analysis or tighter_priors)

print(f"[ar1.py] PRIOR_SCALE={_PRIOR_SCALE} "
      f"(sensitivity={sensitivity_analysis}, tight={tighter_priors}, custom={_CUSTOM_SCALE})")

# ============================================================================
# ELBO Convergence Plotting Helper
# ============================================================================

def _compute_convergence_step(elbo_history, threshold=0.0001):
    """
    Compute the step at which SVI converged.
    
    Convergence defined as: improvement per step < threshold for last 100 steps.
    
    Args:
        elbo_history: List of ELBO values
        threshold: Consider converged when recent improvement < this value
    
    Returns:
        (convergence_step, improvement_rate, is_converged): tuple
          - convergence_step: Step number where convergence detected (or -1 if not)
          - improvement_rate: Recent improvement rate at convergence
          - is_converged: Boolean indicating if converged
    """
    if len(elbo_history) < 100:
        # Not enough data to determine convergence
        return -1, np.mean(np.diff(elbo_history[-len(elbo_history):])), False
    
    # Check last 100 steps
    recent_diffs = np.abs(np.diff(elbo_history[-100:]))
    recent_mean_diff = np.mean(recent_diffs)
    
    if recent_mean_diff < threshold:
        # Likely converged in recent steps
        # Find when it started converging
        for i in range(len(elbo_history) - 1, 100, -1):
            window_start = max(0, i - 100)
            window_diffs = np.abs(np.diff(elbo_history[window_start:i]))
            
            if len(window_diffs) > 0 and np.mean(window_diffs) >= threshold:
                # Found transition point from optimizing to converged
                convergence_step = i + 1
                improvement_rate = np.mean(np.diff(elbo_history[i:]))
                return convergence_step, improvement_rate, True
        
        # Converged very early
        return 0, recent_mean_diff, True
    else:
        # Not converged at end
        return -1, recent_mean_diff, False
    

def _plot_elbo_convergence_enhanced(elbo_history, step_history, output_dir, 
                                    plot_interval=100):
    """
    Create dual-panel ELBO plots: linear and log scale.
    
    The log-scale plot reveals convergence behavior much more clearly:
    - Early optimization becomes compressed (less visual clutter)
    - Tail behavior (plateau, divergence) becomes obvious
    - Discontinuities become glaring
    - You can see whether improvement is exponential (straight line on log) 
      or sublinear (curve on log)
    
    Args:
        elbo_history: List of ELBO values at each step
        step_history: List of step numbers
        output_dir: Directory to save plots
        plot_interval: Not used, kept for API compatibility
    """
    os.makedirs(output_dir, exist_ok=True)

    elbo_history = np.array(elbo_history)
    step_history = np.array(step_history)
    
    # ===== COMPUTE CONVERGENCE STEP =====
    convergence_step, convergence_rate, is_converged = _compute_convergence_step(elbo_history)
    
    elbo_history = np.array(elbo_history)
    step_history = np.array(step_history)
    
    # ===== DIAGNOSTIC: Detect discontinuities =====
    diffs = np.diff(elbo_history)
    abs_diffs = np.abs(diffs)
    max_diff_idx = np.argmax(abs_diffs)
    max_diff_val = abs_diffs[max_diff_idx]
    max_diff_step = step_history[max_diff_idx]
    
    has_discontinuity = max_diff_val > np.std(abs_diffs) * 3
    
    # ===== PLOT 1: Linear scale (original) =====
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(step_history, elbo_history, 'b-', linewidth=2.5, label='ELBO')
    
    # Add reference lines
    mean_elbo = np.mean(elbo_history)
    ax1.axhline(y=mean_elbo, color='r', linestyle='--', alpha=0.5, linewidth=2,
               label=f'Mean ELBO: {mean_elbo:.2f}')
    
    # Moving average
    if len(elbo_history) > 100:
        moving_avg = np.convolve(elbo_history, np.ones(100)/100, mode='valid')
        moving_avg_steps = step_history[99:]
        ax1.plot(moving_avg_steps, moving_avg, 'g--', linewidth=2.5, 
                label='Moving avg (100 steps)', alpha=0.7)
    
    # Mark discontinuity if present
    if has_discontinuity:
        ax1.axvline(max_diff_step, color='red', linestyle=':', linewidth=2.5, alpha=0.7,
                   label=f'Discontinuity at step {max_diff_step}')
        ax1.scatter([max_diff_step], [elbo_history[max_diff_idx]], 
                   color='red', s=200, zorder=5, marker='X')
    
    # Compute improvement rate
    if len(elbo_history) > 1:
        recent_improvement = np.mean(np.diff(elbo_history[-100:]))
        title_suffix = f"\nRecent improvement: {recent_improvement:+.4f} per step"
    else:
        title_suffix = ""
    
    ax1.set_xlabel('SVI Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('ELBO', fontsize=13, fontweight='bold')
    ax1.set_title(f'Linear Scale: ELBO vs Step{title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    
    # Add status annotation
    if len(elbo_history) > 100:
        recent_avg = np.mean(elbo_history[-100:])
        earlier_avg = np.mean(elbo_history[:100])
        improvement_pct = 100 * (recent_avg - earlier_avg) / abs(earlier_avg) if earlier_avg != 0 else 0
        
        if improvement_pct < 0.1:
            status = "Likely converged"
            color = 'lightgreen'
        elif improvement_pct < 1.0:
            status = "≈ Approaching convergence"
            color = 'lightyellow'
        else:
            status = "WARNING: Still optimizing"
            color = 'lightcoral'
        
        ax1.text(0.02, 0.98, f"{status}\nImprovement: {improvement_pct:.3f}%", 
               transform=ax1.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    plot_path_linear = f"{output_dir}/elbo_convergence_linear.png"
    plt.savefig(plot_path_linear, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved linear-scale ELBO plot: {plot_path_linear}")
    
    
    # ===== PLOT 2: Log scale =====
    fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Standard log-scale plot
    # Since ELBO is negative, we plot -ELBO (the loss) on log scale
    elbo_positive = -elbo_history  # Flip sign so we have positive numbers
    
    ax2_left.semilogy(step_history, elbo_positive, 'b-', linewidth=2.5, label='-ELBO (loss)')
    
    # Add reference
    mean_loss = np.mean(elbo_positive)
    ax2_left.axhline(y=mean_loss, color='r', linestyle='--', alpha=0.5, linewidth=2,
                    label=f'Mean loss: {mean_loss:.2e}')
    
    # Moving average on log scale
    if len(elbo_positive) > 100:
        moving_avg_loss = np.convolve(elbo_positive, np.ones(100)/100, mode='valid')
        moving_avg_steps = step_history[99:]
        ax2_left.semilogy(moving_avg_steps, moving_avg_loss, 'g--', linewidth=2.5,
                         label='Moving avg (100 steps)', alpha=0.7)
    
    # Mark discontinuity
    if has_discontinuity:
        ax2_left.axvline(max_diff_step, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
        ax2_left.scatter([max_diff_step], [elbo_positive[max_diff_idx]], 
                        color='red', s=200, zorder=5, marker='X',
                        label=f'Discontinuity (Δ={max_diff_val:.2e})')
    
    ax2_left.set_xlabel('SVI Step', fontsize=13, fontweight='bold')
    ax2_left.set_ylabel('-ELBO (log scale)', fontsize=13, fontweight='bold')
    ax2_left.set_title('Log Scale: -ELBO vs Step', fontsize=14, fontweight='bold')
    ax2_left.grid(True, alpha=0.3, which='both')
    ax2_left.legend(fontsize=11, loc='best')
    
    
    # Right: Convergence velocity on log scale
    # Shows improvement rate (slope) which reveals convergence regime
    if len(step_history) > 10:
        # Compute local improvement rate over 100-step windows
        window = 100
        velocities = []
        velocity_steps = []
        
        for i in range(len(elbo_history) - window):
            velocity = (elbo_history[i+window] - elbo_history[i]) / window
            velocities.append(velocity)
            velocity_steps.append(step_history[i + window//2])
        
        velocities = np.array(velocities)
        velocity_steps = np.array(velocity_steps)
        
        # Plot improvement (positive = getting better ELBO, which is negative)
        ax2_right.plot(velocity_steps, velocities, 'b-', linewidth=2.5, label='Improvement rate')
        
        # Add zero line
        ax2_right.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        
        # Shade positive (improving) region
        ax2_right.fill_between(velocity_steps, 0, velocities, where=(velocities>=0), 
                              alpha=0.2, color='green', label='Improving ELBO')
        ax2_right.fill_between(velocity_steps, 0, velocities, where=(velocities<0),
                              alpha=0.2, color='red', label='Degrading ELBO')
        
        ax2_right.set_xlabel('SVI Step', fontsize=13, fontweight='bold')
        ax2_right.set_ylabel('ELBO improvement per 100 steps', fontsize=13, fontweight='bold')
        ax2_right.set_title('Convergence Velocity', fontsize=14, fontweight='bold')
        ax2_right.grid(True, alpha=0.3)
        ax2_right.legend(fontsize=11, loc='best')
        
        # Add convergence diagnosis annotation
        recent_velocity = np.mean(velocities[-100:]) if len(velocities) >= 100 else np.mean(velocities)
        if abs(recent_velocity) < 0.001:
            vel_status = "Converged"
            vel_color = 'lightgreen'
        elif recent_velocity < 0.01:
            vel_status = "≈ Slowing"
            vel_color = 'lightyellow'
        else:
            vel_status = "WARNING: Still improving"
            vel_color = 'lightcoral'
        
        ax2_right.text(0.02, 0.98, f"{vel_status}\nRecent velocity: {recent_velocity:+.4f}",
                      transform=ax2_right.transAxes, fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=vel_color, alpha=0.5))
    
    plt.tight_layout()
    plot_path_log = f"{output_dir}/elbo_convergence_logscale.png"
    plt.savefig(plot_path_log, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved log-scale ELBO plot: {plot_path_log}")
    
    
    # ===== PLOT 3: Detailed diagnostics summary =====
    fig3 = plt.figure(figsize=(14, 10))
    gs = fig3.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel 1: Full ELBO history
    ax3_1 = fig3.add_subplot(gs[0, :])
    ax3_1.plot(step_history, elbo_history, 'b-', linewidth=2)
    if has_discontinuity:
        ax3_1.axvline(max_diff_step, color='red', linestyle='--', alpha=0.7)
    ax3_1.set_ylabel('ELBO', fontweight='bold')
    ax3_1.set_title('Full ELBO History', fontweight='bold')
    ax3_1.grid(True, alpha=0.3)
    
    # Panel 2: ELBO differences (gradient)
    ax3_2 = fig3.add_subplot(gs[1, 0])
    diffs_abs = np.abs(diffs)
    ax3_2.semilogy(step_history[:-1], diffs_abs, 'b-', linewidth=1.5, alpha=0.7)
    ax3_2.axhline(y=np.median(diffs_abs), color='r', linestyle='--', 
                 label=f'Median: {np.median(diffs_abs):.2e}', linewidth=2)
    if has_discontinuity:
        ax3_2.scatter([max_diff_step], [max_diff_val], color='red', s=150, zorder=5, marker='X')
    ax3_2.set_ylabel('|ELBO difference|', fontweight='bold')
    ax3_2.set_title('Step-to-step changes (log scale)', fontweight='bold')
    ax3_2.grid(True, alpha=0.3, which='both')
    ax3_2.legend()
    
    # Panel 3: Cumulative improvement
    ax3_3 = fig3.add_subplot(gs[1, 1])
    cumulative_improvement = np.cumsum(diffs)
    ax3_3.plot(step_history[:-1], cumulative_improvement, 'g-', linewidth=2)
    ax3_3.set_ylabel('Cumulative improvement', fontweight='bold')
    ax3_3.set_title('Cumulative ELBO change', fontweight='bold')
    ax3_3.grid(True, alpha=0.3)
    
    # Panel 4: Diagnostics text
    ax3_4 = fig3.add_subplot(gs[2, :])
    ax3_4.axis('off')
    
    # Build convergence status string
    if is_converged:
        if convergence_step == 0:
            convergence_status = "Converged from start"
        else:
            convergence_status = f"Converged at step {convergence_step} / {len(step_history)}"
    else:
        convergence_status = f"WARNING: Not converged (final improvement: {convergence_rate:.4e} per step)"
    
    diagnostics_text = f"""
        ELBO CONVERGENCE DIAGNOSTICS
        {'='*60}

        Data Summary:
        Initial ELBO:     {elbo_history[0]:15.4f}
        Final ELBO:       {elbo_history[-1]:15.4f}
        Total improvement: {elbo_history[-1] - elbo_history[0]:15.4f}
        Mean ELBO:        {np.mean(elbo_history):15.4f}
        Std ELBO:         {np.std(elbo_history):15.4f}

        Convergence Metrics:
        Steps completed:  {len(step_history):15d}
        Convergence:      {convergence_status}
        Recent improvement (last 100): {np.mean(np.diff(elbo_history[-100:])):15.4f} per step
        Median step change: {np.median(diffs_abs):15.4e}
        Max step change:  {np.max(diffs_abs):15.4e}

        Discontinuity Detection:
        Detected: {'YES WARNING' if has_discontinuity else 'NO ok'}
        {"  Location: step " + str(max_diff_step) + f" (Δ={max_diff_val:.4e})" if has_discontinuity else ""}

        Interpretation:
        - If log-scale plot shows straight line: exponential convergence
        - If log-scale plot shows curve: sublinear convergence
        - If velocity stays positive: still improving
        - If velocity near zero: converged
        - If discontinuity present: possible gradient explosion or numerical issue
        """
    
    ax3_4.text(0.05, 0.95, diagnostics_text, transform=ax3_4.transAxes,
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ELBO Convergence Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    plot_path_diagnostics = f"{output_dir}/elbo_convergence_diagnostics.png"
    plt.savefig(plot_path_diagnostics, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved diagnostic plots: {plot_path_diagnostics}")
    
    # ===== PRINT TEXT DIAGNOSTICS =====
    print("="*70)
    print("ELBO CONVERGENCE DIAGNOSTICS")
    print("="*70)
    print(f"\nInitial ELBO:     {elbo_history[0]:15.4f}")
    print(f"Final ELBO:       {elbo_history[-1]:15.4f}")
    print(f"Total improvement: {elbo_history[-1] - elbo_history[0]:15.4f}")
    print(f"Mean ELBO:        {np.mean(elbo_history):15.4f}")
    print(f"Std ELBO:         {np.std(elbo_history):15.4f}")
    
    # Convergence step info
    if is_converged:
        if convergence_step == 0:
            print(f"\nConverged from start")
        else:
            convergence_pct = 100 * convergence_step / len(step_history)
            print(f"\nConverged at step {convergence_step} ({convergence_pct:.1f}% of total)")
            print(f"  Remaining {len(step_history) - convergence_step} steps for refinement")
    else:
        print(f"\nWARNING: Not converged after {len(step_history)} steps")
        print(f"  Recent improvement: {convergence_rate:.4e} per step")
        print(f"  Consider increasing num_steps or adjusting learning rate")
    
    print(f"\nRecent improvement (last 100 steps): {np.mean(np.diff(elbo_history[-100:])):+.4f} per step")
    print(f"Median step change: {np.median(diffs_abs):.4e}")
    print(f"Max step change:    {np.max(diffs_abs):.4e}")
    
    if has_discontinuity:
        print(f"\nWARNING: DISCONTINUITY DETECTED at step {max_diff_step}")
        print(f"  Jump magnitude: {max_diff_val:.4e}")
        print(f"  ELBO before: {elbo_history[max_diff_idx]:.4f}")
        print(f"  ELBO after:  {elbo_history[max_diff_idx+1]:.4f}")
        print(f"\n  Possible causes:")
        print(f"  1. Gradient explosion without clipping")
        print(f"  2. Learning rate too high -> overshooting optimum")
        print(f"  3. Numerical instability in likelihood computation")
        print(f"  4. Random seed dependent behavior")
    else:
        print(f"\nNo major discontinuities detected")
    
    print("="*70 + "\n")


# ============================================================================
# AR(1) Log-Likelihood (Kalman Filter)
# ============================================================================

# models/ar1.py
# AR(1) bias model with personalized ISF/CR priors + optional carb uncertainty
# Supports both MCMC (full Bayes) and SVI AutoDelta (MAP estimation)
# ENHANCED: Includes Phi-bounding to prevent ISF collapse.

# ============================================================================
# Likelihood Helpers
# ============================================================================

@staticmethod
@jax.jit
def ar1_log_likelihood(y, mu, phi, sigma_eta, sigma_obs):
    """Compute log-likelihood for AR(1) using Kalman filter."""
    T = y.shape[0]
    def kalman_step(carry, yt_mut):
        m, P, log_lik = carry
        y_t, mu_t = yt_mut
        m_pred = phi * m
        P_pred = phi**2 * P + sigma_eta**2
        y_pred_mean = mu_t + m_pred
        y_pred_var = P_pred + sigma_obs**2
        residual = y_t - y_pred_mean
        log_lik_t = jax.scipy.stats.norm.logpdf(residual, 0, jnp.sqrt(y_pred_var + 1e-8))
        K = P_pred / (y_pred_var + 1e-8)
        m_new = m_pred + K * residual
        P_new = (1 - K) * P_pred
        return (m_new, P_new, log_lik + log_lik_t), None
    m0, P0 = 0.0, 1e-8
    (_, _, total_log_lik), _ = jax.lax.scan(kalman_step, (m0, P0, 0.0), (y, mu))
    return total_log_lik

# ============================================================================
# MC Marginalization for Per-Event Carb Uncertainty
# ============================================================================

def mc_marginalize_carb_uncertainty_vectorized(
    params, cg, it, ia, inn, ct, ca, cn, hist, vel, obs,
    phi, sigma_eta, sigma_obs, sigma_carb_obs, key, num_mc_samples=30
):
    subkeys = jax.random.split(key, num_mc_samples)
    def single_mc_sample_with_kf(subkey):
        eps = jax.random.normal(subkey, ca.shape) * sigma_carb_obs

        # Force the denominator to be at least 0.33 (prevents more than 3x inflation)
        denom = jnp.clip(1.0 + eps, a_min=0.33, a_max=3.0)
        ca_true = jnp.maximum(ca / denom, 0.0)

        # ca_true = jnp.maximum(ca / (1.0 + eps), 0.0)
        _, mu_mc = predict_cgm_loop_vectorized(params, cg[None], it[None], ia[None], inn[None], 
                                              ct[None], ca_true[None], cn[None], hist[None], vel[None])
        mu_mc = mu_mc[0]
        log_lik_mc = ar1_log_likelihood(obs, mu_mc, phi, sigma_eta, sigma_obs)
        return log_lik_mc, mu_mc
    log_liks, mu_samples = jax.vmap(single_mc_sample_with_kf)(subkeys)
    log_lik_marginalized = jax.scipy.special.logsumexp(log_liks) - jnp.log(num_mc_samples)
    return log_lik_marginalized, jnp.mean(mu_samples, axis=0), jnp.std(mu_samples, axis=0)

# ============================================================================
# AR(1) Model Class
# ============================================================================

class AR1Model():
    def __init__(self, config=None):
        config = config or {}
        self.include_carb_uncertainty = config.get('carb_uncertainty', False)
        self.carb_uncertainty_type = config.get('carb_uncertainty_type', 'global')
        self.per_event_method = config.get('per_event_method', 'mc_marginalization')
        self.num_carb_slots = config.get('num_carb_slots', None)
        self.mc_carb_samples = config.get('mc_carb_samples', 30)
        self.param_names = ['theta_ISF', 'theta_CR', 'phi', 'sigma_eta', 'sigma_obs']
        if self.include_carb_uncertainty:
            self.param_names.append('sigma_carb_obs')

    def model_fn(self, current_glucose_array, ins_times_array, ins_amounts_array, ins_n_array,
                 carb_times_array, carb_amounts_array, carb_n_array,
                 historical_cgm_array_batch, observed_glucose_array=None, sample_predictions=False):

        if sensitivity_analysis:
            print("Using weakly informative priors for synthetic parameter recovery (0.70)")
            
            # 1. CR: Anchored near 10.0, but scale=0.7 allows it to easily range from 3 to 40
            theta_CR = numpyro.sample('theta_CR', dist.LogNormal(loc=2.30, scale=0.70))

            # 2. Ratio: Anchored near 4.5, but scale=0.7 allows ratios from 1.5 to 13.0
            log_multiplier = numpyro.sample('log_multiplier', dist.Normal(loc=1.50, scale=0.70))

            # 3. Deterministic Link
            isf_cr_ratio = numpyro.deterministic('isf_cr_ratio', jnp.exp(log_multiplier))

            # 4. Final Parameter
            theta_ISF = numpyro.deterministic('theta_ISF', theta_CR * isf_cr_ratio)

            # 5. Noise Floors: Wide scales so the model doesn't use ISF to absorb noise
            sigma_obs = numpyro.sample('sigma_obs', dist.LogNormal(loc=2.85, scale=0.50))
            sigma_eta = numpyro.sample('sigma_eta', dist.LogNormal(loc=1.10, scale=0.50))

            # AR1 Bias Persistence
            phi_raw = numpyro.sample("phi_raw", dist.Beta(10, 2))
            phi = numpyro.deterministic("phi", 0.80 + (phi_raw * 0.19))

        elif tighter_priors:

            print("Using tighter priors for publication (0.10 scale)")

            # CR: scale=0.10 allows range [7.5, 13.3] g/U
            # Prevents ISF collapse while still allowing patient differentiation
            theta_CR = numpyro.sample('theta_CR', dist.LogNormal(loc=2.30, scale=0.10))

            # ISF/CR ratio: scale=0.10 allows range [3.3, 6.1]
            # ISF prior range becomes [25, 81] mg/dL per U — physiologically defensible
            log_multiplier = numpyro.sample('log_multiplier', dist.Normal(loc=1.50, scale=0.10))

            isf_cr_ratio = numpyro.deterministic('isf_cr_ratio', jnp.exp(log_multiplier))
            theta_ISF    = numpyro.deterministic('theta_ISF', theta_CR * isf_cr_ratio)

            # sigma_obs: tighten slightly — scale=0.50 allows [7, 47] mg/dL
            # Wide enough for model mismatch, prevents collapse to 0.13
            sigma_obs = numpyro.sample('sigma_obs', dist.LogNormal(loc=2.85, scale=0.50))

            # sigma_eta: unchanged
            sigma_eta = numpyro.sample('sigma_eta', dist.LogNormal(loc=1.10, scale=0.50))

            phi_raw = numpyro.sample("phi_raw", dist.Beta(10, 2))
            phi     = numpyro.deterministic("phi", 0.80 + (phi_raw * 0.19))

        else:
            print(f"Using informative priors for publication (scale={_PRIOR_SCALE})")

            # CR: population center ~10 g/U
            # scale=0.30 (main): range [6, 16]; 0.10 (tight): [8, 12]; 0.50 (loose): [4, 24]
            theta_CR = numpyro.sample('theta_CR', dist.LogNormal(loc=2.30, scale=_PRIOR_SCALE))

            # Ratio ISF/CR: population center ~4.5
            # scale=0.30 (main): range [2.7, 7.4]; 0.10: [3.7, 5.5]; 0.50: [1.9, 10.8]
            log_multiplier = numpyro.sample('log_multiplier', dist.Normal(loc=1.50, scale=_PRIOR_SCALE))

            # Deterministic links — unchanged
            isf_cr_ratio = numpyro.deterministic('isf_cr_ratio', jnp.exp(log_multiplier))
            theta_ISF    = numpyro.deterministic('theta_ISF', theta_CR * isf_cr_ratio)

            # sigma_obs: FREE — real data has sensor noise + model mismatch (~15-30 mg/dL)
            # scale=1.0 lets each patient's posterior find its own level
            # was scale=0.03 which pinned all patients at 17.3 ± 0.9 (no patient variation)
            sigma_obs = numpyro.sample('sigma_obs', dist.LogNormal(loc=2.85, scale=1.00))

            # sigma_eta: moderate freedom — was 0.03 (frozen at 3.0 for all patients)
            # scale=0.50 allows range [1.1, 8.2], lets AR1 adapt per patient
            sigma_eta = numpyro.sample('sigma_eta', dist.LogNormal(loc=1.10, scale=0.50))

            # phi: unchanged
            phi_raw = numpyro.sample("phi_raw", dist.Beta(10, 2))
            phi     = numpyro.deterministic("phi", 0.80 + (phi_raw * 0.19))
            # ---

        
        num_instances = current_glucose_array.shape[0]
        params = create_loop_params(theta_CR, theta_ISF)

        # Base nominal velocity
        bg_velocity_batch = compute_bg_velocity_batch(params, current_glucose_array, historical_cgm_array_batch,
                                                     ins_times_array, ins_amounts_array, carb_times_array, carb_amounts_array)
        
        mu = None
        # 3. CARB UNCERTAINTY PATHS
        if self.include_carb_uncertainty:
            # sigma_carb_obs = numpyro.sample(
            #     'sigma_carb_obs', 
            #     dist.TruncatedNormal(loc=0.10, scale=0.05, low=0.0, high=1.0)
            # )
            
            # if self.carb_uncertainty_type == 'global':
            #     carb_adj = numpyro.sample('carb_adjustment', dist.Normal(0.0, sigma_carb_obs))
            #     carb_use = jnp.maximum(carb_amounts_array / (1.0 + carb_adj), 0.0)


            # --- MAX STRENGTH CARB UNCERTAINTY ANCHOR ---
            # Mode ~0.20 (20% relative error). 
            # Scale 0.03 is extremely informative; it keeps the error estimate near 20%
            # and prevents the 'global' adjustment from becoming a sink for model error.
            
            if sensitivity_analysis:

                # --- Sensitivity analysis - diffuse prior
                sigma_carb_obs = numpyro.sample('sigma_carb_obs', dist.LogNormal(loc=-1.60, scale=0.30))  # was 0.03
            
            else:

                # --- Publication prior
                # sigma_carb_obs = numpyro.sample('sigma_carb_obs', dist.LogNormal(loc=-1.60, scale=0.03))

                # sigma_carb_obs (if carb uncertainty enabled) — unchanged
                sigma_carb_obs = numpyro.sample('sigma_carb_obs', dist.LogNormal(loc=-1.60, scale=0.30))


            if self.carb_uncertainty_type == 'global':
                carb_adj = numpyro.sample('carb_adjustment', dist.Normal(0.0, sigma_carb_obs))
                
                # Allows for 3x inflation and 3x shrinkage—very safe for MCMC stability
                denom = jnp.clip(1.0 + carb_adj, a_min=0.33, a_max=3.0) 
                carb_use = carb_amounts_array / denom

                vel_adj = compute_bg_velocity_batch(params, current_glucose_array, historical_cgm_array_batch,
                                                   ins_times_array, ins_amounts_array, carb_times_array, carb_use)
                _, mu = predict_cgm_loop_vectorized(params, current_glucose_array, ins_times_array, ins_amounts_array,
                                                   ins_n_array, carb_times_array, carb_use, carb_n_array,
                                                   historical_cgm_array_batch, vel_adj)
                if observed_glucose_array is not None and not sample_predictions:
                    log_liks = jax.vmap(lambda y, m: ar1_log_likelihood(y, m, phi, sigma_eta, sigma_obs))(observed_glucose_array, mu)
                    numpyro.factor("obs", jnp.sum(log_liks))
                
            elif self.carb_uncertainty_type == 'per_event':
                if self.per_event_method == 'direct':
                    with numpyro.plate("carb_events", self.num_carb_slots):
                        adjs = numpyro.sample('carb_adjustment', dist.Normal(0.0, sigma_carb_obs))
                    flat_adjs = adjs[:num_instances * carb_amounts_array.shape[1]].reshape(carb_amounts_array.shape)

                    denom = jnp.clip(1.0 + flat_adjs, a_min=0.33, a_max=3.0) # Prevents division by zero
                    carb_use = jnp.maximum(carb_amounts_array / denom, 0.0)

                    # carb_use = jnp.maximum(carb_amounts_array / (1.0 + flat_adjs), 0.0)
                    vel_adj = compute_bg_velocity_batch(params, current_glucose_array, historical_cgm_array_batch,
                                                       ins_times_array, ins_amounts_array, carb_times_array, carb_use)
                    _, mu = predict_cgm_loop_vectorized(params, current_glucose_array, ins_times_array, ins_amounts_array,
                                                       ins_n_array, carb_times_array, carb_use, carb_n_array,
                                                       historical_cgm_array_batch, vel_adj)
                    if observed_glucose_array is not None and not sample_predictions:
                        log_liks = jax.vmap(lambda y, m: ar1_log_likelihood(y, m, phi, sigma_eta, sigma_obs))(observed_glucose_array, mu)
                        numpyro.factor("obs", jnp.sum(log_liks))
                
                elif self.per_event_method == 'mc_marginalization':
                    if observed_glucose_array is not None and not sample_predictions:
                        seed = jnp.asarray(numpyro.sample('_mc_seed', dist.Uniform(0, 2**31-1)), dtype=jnp.uint32)
                        base_key = jax.random.PRNGKey(seed)
                        log_liks, mu_means, _ = jax.vmap(lambda idx, cg, it, ia, inn, ct, ca, cn, hist, vel, obs: 
                            mc_marginalize_carb_uncertainty_vectorized(params, cg, it, ia, inn, ct, ca, cn, hist, vel, obs,
                            phi, sigma_eta, sigma_obs, sigma_carb_obs, jax.random.fold_in(base_key, idx), self.mc_carb_samples)
                        )(jnp.arange(num_instances), current_glucose_array, ins_times_array, ins_amounts_array, ins_n_array, 
                          carb_times_array, carb_amounts_array, carb_n_array, historical_cgm_array_batch, bg_velocity_batch, observed_glucose_array)
                        numpyro.factor("obs", jnp.sum(log_liks))
                        mu = mu_means
                    else:
                        _, mu = predict_cgm_loop_vectorized(params, current_glucose_array, ins_times_array, ins_amounts_array,
                                                           ins_n_array, carb_times_array, carb_amounts_array, carb_n_array,
                                                           historical_cgm_array_batch, bg_velocity_batch)
        else:
            # 4. NO CARB UNCERTAINTY
            _, mu = predict_cgm_loop_vectorized(params, current_glucose_array, ins_times_array, ins_amounts_array,
                                               ins_n_array, carb_times_array, carb_amounts_array, carb_n_array,
                                               historical_cgm_array_batch, bg_velocity_batch)
            if observed_glucose_array is not None and not sample_predictions:
                log_liks = jax.vmap(lambda y, m: ar1_log_likelihood(y, m, phi, sigma_eta, sigma_obs))(observed_glucose_array, mu)
                numpyro.factor("obs", jnp.sum(log_liks))

        # 5. PREDICTION MODE
        if sample_predictions:
            T = mu.shape[1]
            eta_std = numpyro.sample('eta_std', dist.Normal(jnp.zeros((num_instances, T)), 1.0))
            etas = sigma_eta[..., None] * eta_std
            def ar1_step(b_prev, eta_k):
                b_new = phi * b_prev + eta_k
                return b_new, b_new
            bs_full = jax.vmap(lambda b0, e: jax.lax.scan(ar1_step, b0, e)[1])(jnp.zeros(num_instances), etas)
            mean_gl = mu + bs_full
            numpyro.sample('obs', dist.Normal(mean_gl, sigma_obs).to_event(1))
            numpyro.deterministic('pred_mean', mean_gl)
            numpyro.deterministic('bias', bs_full)

    def extract_params(self, samples):
        return {k: samples[k] for k in self.param_names if k in samples}
    
    def get_param_names_for_plotting(self):
        return self.param_names
    

    # ========================================================================
    # SVI MAP ESTIMATION
    # ========================================================================

    def fit_svi_map(self, 
        current_glucose_array,
        ins_times_array, ins_amounts_array, ins_n_array,
        carb_times_array, carb_amounts_array, carb_n_array,
        historical_cgm_array_batch,
        observed_glucose_array,
        rng_key,
        num_steps=5000,
        learning_rate=0.001,
        verbose=True,
        elbo_plot_dir=None,
        plot_interval=100,
        max_grad_norm=0.5,
        enable_gradient_clipping=True,
        initial_params=None,
        output_dir=None,
        user_id=None):
        """
        SVI MAP estimation with warm-starting from MCMC.
        
        Args:
            current_glucose_array: [n_instances]
            ins_times_array: [n_instances, 140]
            ins_amounts_array: [n_instances, 140]
            ins_n_array: [n_instances]
            carb_times_array: [n_instances, 140]
            carb_amounts_array: [n_instances, 140]
            carb_n_array: [n_instances]
            historical_cgm_array_batch: [n_instances, history_len]
            observed_glucose_array: [n_instances, T]
            rng_key: JAX random key
            num_steps: Number of SVI iterations (default: 5000)
            learning_rate: Adam learning rate (default: 0.001)
            verbose: Print progress (default: True)
            elbo_plot_dir: Directory to save ELBO plots (default: None)
            plot_interval: Plot every N steps (default: 100)
            max_grad_norm: Gradient clipping threshold (default: 0.5)
            enable_gradient_clipping: Use gradient clipping (default: True)
            initial_params: Warm-start parameters from MCMC (default: None)
        
        Returns:
            SVIResult object with:
                - params: Optimized parameters
                - losses: Loss history
                - elbo_history: ELBO history
        """
        
        print(f"\nStarting SVI with enhanced optimizer...")
        print(f"  Num steps: {num_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gradient clipping: {'ON (max_norm={})'.format(max_grad_norm) if enable_gradient_clipping else 'OFF'}")
        if initial_params:
            print(f"  Warm-start: ON (will override random initialization)")
        else:
            print(f"  Warm-start: OFF (using random initialization)")
        if elbo_plot_dir:
            print(f"  ELBO tracking: ON (plot every {plot_interval} steps)")
        else:
            print(f"  ELBO tracking: OFF")
        
        # ===== STEP 1: WRAP MODEL IN CLOSURE =====
        def model_wrapped():
            return self.model_fn(
                current_glucose_array,
                ins_times_array, ins_amounts_array, ins_n_array,
                carb_times_array, carb_amounts_array, carb_n_array,
                historical_cgm_array_batch,
                observed_glucose_array=observed_glucose_array,
                sample_predictions=False
            )
        
        # ===== STEP 2: CREATE GUIDE =====
        guide = AutoDelta(model_wrapped)
        print(f"\n  Created AutoDelta guide")
        
        # ===== STEP 3: CREATE OPTIMIZER WITH GRADIENT CLIPPING =====
        if enable_gradient_clipping:
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate)
            )
            print(f"  Gradient clipping ENABLED")
            print(f"    Max gradient norm: {max_grad_norm}")
        else:
            optimizer = optax.adam(learning_rate)
            print(f"  Using standard Adam (no gradient clipping)")
        
        # ===== STEP 4: CREATE SVI =====
        svi = SVI(model_wrapped, guide, optimizer, Trace_ELBO())
        print(f"  Created SVI optimizer")
        
        # ===== STEP 5: INITIALIZE SVI STATE =====
        print(f"\n  Initializing SVI state with random parameters...")
        svi_state = svi.init(rng_key)
        print(f"  SVI state initialized")
        
        # Get initial parameters to see what AutoDelta generates
        initial_auto_params = svi.get_params(svi_state)
        print(f"\n  AutoDelta parameter names:")
        for auto_name in sorted(initial_auto_params.keys()):
            print(f"    {auto_name}")
        
        # ===== STEP 6: WARM-START IF PROVIDED =====
        if initial_params is not None:
            print(f"\n  Applying warm-start from MCMC MAP estimates...")
            print(f"    MCMC MAP contains: {list(initial_params.keys())}")
            
            # Get current params from SVI
            current_params = svi.get_params(svi_state)
            new_params = {}
            num_matched = 0
            
            for auto_name, auto_value in current_params.items():
                if auto_name.endswith('_auto_loc'):
                    clean_param_name = auto_name[:-9]
                else:
                    clean_param_name = auto_name
                
                if clean_param_name in initial_params:
                    warm_start_value = initial_params[clean_param_name]
                    new_params[auto_name] = jnp.asarray(warm_start_value)
                    num_matched += 1
                    if verbose:
                        print(f"    {clean_param_name:20s}: {float(warm_start_value):10.6f}")
                else:
                    new_params[auto_name] = auto_value
            
            print(f"\n  Warm-start summary:")
            print(f"    Parameters matched: {num_matched}")
            
            # Reinitialize SVI with warm-start params
            svi_state = svi.init(rng_key)
            
            # Update params
            try:
                for key, val in new_params.items():
                    svi_state.params[key] = val
                print(f"  Warm-start parameters applied")
            except Exception as e:
                print(f"    Warning: Could not fully update params ({e})")
                print(f"    Continuing with partially updated parameters...")
        
        # ===== STEP 7: TRACK METRICS =====
        elbo_history = []
        step_history = []
        losses_history = []
        
        print(f"\n  Running {num_steps} optimization steps...")
        print(f"  {'Step':>6} | {'ELBO':>12} | {'Improvement':>12} | Status")
        print(f"  {'-'*70}")
        
        # ===== STEP 8: MAIN OPTIMIZATION LOOP WITH EARLY STOPPING =====
        # Define early stopping parameters
        patience_window = 500   # was 1000 — start checking earlier
        check_frequency = 100   # keep same
        convergence_threshold = 0.05  # was 0.001
        
        for step_num in range(num_steps):
            svi_state, loss = svi.update(svi_state)
            elbo = -float(loss)
            elbo_history.append(elbo)
            step_history.append(step_num)

            # Save checkpoint every 100 steps (overwrites same file)
            if step_num % 100 == 0 and step_num > 0:
                try:
                    current_params = svi.get_params(svi_state)
                    checkpoint_data = {
                        'map_params': current_params,
                        'elbo_history': elbo_history,
                        'step': step_num,
                    }
                    os.makedirs(f"{output_dir}/results", exist_ok=True)
                    results_pkl_path = f"{output_dir}/results/user_{user_id}_results.pkl"
                    with open(results_pkl_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                except Exception as e:
                    print(f"WARNING: Checkpoint save failed at step {step_num}: {e}")

            # Early stopping
            if step_num > patience_window and step_num % check_frequency == 0:
                _, _, is_converged = _compute_convergence_step(
                    elbo_history, threshold=convergence_threshold
                )
                if is_converged:
                    print(f"Early stopping at step {step_num}")
                    break
            
            # Print progress every 500 steps
            if step_num % 500 == 0 or step_num == num_steps - 1:
                improvement = ""
                if len(elbo_history) > 1:
                    recent_imp = elbo_history[-1] - elbo_history[max(0, len(elbo_history)-101)]
                    improvement = f"{recent_imp:+12.4f}"
                else:
                    improvement = f"{'N/A':>12}"
                
                print(f"  {step_num:6d} | {elbo:12.4f} | {improvement}")
            
            # Update diagnostic plots
            if elbo_plot_dir and step_num > 0 and step_num % plot_interval == 0:
                try:
                    _plot_elbo_convergence_enhanced(elbo_history, step_history, elbo_plot_dir)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Could not plot ELBO: {e}")
        
        print(f"  {'='*70}")
        print(f"  SVI optimization complete")
        
        # Get final parameters
        params = svi.get_params(svi_state)
        
        # Print summary
        if verbose:
            print(f"\n  OPTIMIZATION SUMMARY:")
            print(f"  {'-'*70}")
            print(f"  Initial ELBO:        {elbo_history[0]:12.4f}")
            print(f"  Final ELBO:          {elbo_history[-1]:12.4f}")
            print(f"  Total improvement:   {elbo_history[-1] - elbo_history[0]:+12.4f}")
            print(f"  Steps completed:     {num_steps}")
            print(f"  Recent rate:         {(elbo_history[-1] - elbo_history[max(0, len(elbo_history)-101)]) / min(100, len(elbo_history)):+.4f} per step")
        
        # Return result object
        class SVIResult:
            def __init__(self, params, losses, elbo_history):
                self.params = params
                self.losses = losses
                self.elbo_history = elbo_history
        
        return SVIResult(params, losses_history, elbo_history)

    
    def extract_map_params(self, svi_result):
        import jax.numpy as jnp
        params = svi_result.params
        cleaned_params = {}
        
        for key, value in params.items():
            clean_key = key[:-9] if key.endswith('_auto_loc') else key
            if clean_key == 'carb_adjustment' and \
               self.carb_uncertainty_type == 'per_event' and \
               self.per_event_method == 'direct':
                continue
            cleaned_params[clean_key] = jnp.asarray(value)
        
        # --- RECONSTRUCT ALL DETERMINISTIC SITES ---
        
        # 1. Metabolic Parameters
        if 'theta_CR' in cleaned_params and 'log_multiplier' in cleaned_params:
            ratio = jnp.exp(cleaned_params['log_multiplier'])
            cleaned_params['isf_cr_ratio'] = ratio
            cleaned_params['theta_ISF'] = cleaned_params['theta_CR'] * ratio
            
        # 2. Autocorrelation (AR1 models only)
        if 'phi_raw' in cleaned_params:
            # Reconstruct phi using the same logic as model_fn
            cleaned_params['phi'] = 0.80 + (cleaned_params['phi_raw'] * 0.19)
            print(f"  Reconstructed deterministic phi: {float(cleaned_params['phi']):.4f}")

        return cleaned_params

    
    def sample_posterior_predictive_map(self,
        current_glucose_array,
        ins_times_array, ins_amounts_array, ins_n_array,
        carb_times_array, carb_amounts_array, carb_n_array,
        historical_cgm_array_batch,
        map_params,
        rng_key,
        num_samples=500):
        """
        Sample posterior predictive trajectories using MAP parameters.
        
        Uses fixed MAP parameters but still samples:
        - Carb uncertainty (if enabled): sampled carb adjustments propagate through Loop model
        - AR(1) bias realizations
        - Observation noise
        
        Args:
            current_glucose_array: [n_instances]
            ins_times_array: [n_instances, 140]
            ins_amounts_array: [n_instances, 140]
            ins_n_array: [n_instances]
            carb_times_array: [n_instances, 140]
            carb_amounts_array: [n_instances, 140]
            carb_n_array: [n_instances]
            historical_cgm_array_batch: [n_instances, history_len]
            map_params: dict of MAP parameter estimates
            rng_key: JAX random key for sampling
            num_samples: number of posterior predictive samples (default: 500)
        
        Returns:
            samples: dict with keys:
                - 'pred_mean': [n_instances, T, num_samples]
                - 'bias': [n_instances, T, num_samples]
                - 'obs': [n_instances, T, num_samples]
        """
        import numpy as np
        
        num_instances = current_glucose_array.shape[0]
        T = 48  # typical forecast horizon
        
        # Generate trajectories for each sample
        samples_pred_mean = []
        samples_bias = []
        samples_obs = []
        
        for s in range(num_samples):
            # Split key for this sample
            key = jax.random.fold_in(rng_key, s)
            key_carb, key_eta, key_obs = jax.random.split(key, 3)
            
            # ===== SAMPLE CARB AMOUNTS (if carb uncertainty enabled) =====
            carb_amounts_to_use = carb_amounts_array
            
            if self.include_carb_uncertainty and 'sigma_carb_obs' in map_params:
                sigma_carb_obs = float(map_params['sigma_carb_obs'])
                
                if self.carb_uncertainty_type == 'global':
                    carb_adjustment = jax.random.normal(key_carb, ()) * sigma_carb_obs
                    carb_amounts_to_use = carb_amounts_array / (1.0 + carb_adjustment)
                    
                elif self.carb_uncertainty_type == 'per_event':
                    # For per-event, sample adjustments regardless of method
                    # (method only matters during training/inference)
                    carb_adjustments = jax.random.normal(key_carb, carb_amounts_array.shape) * sigma_carb_obs
                    carb_amounts_to_use = carb_amounts_array / (1.0 + carb_adjustments)
                
                carb_amounts_to_use = jnp.maximum(carb_amounts_to_use, 0.0)
            
            # Create loop parameters from MAP estimates
            params = create_loop_params(
                float(map_params['theta_CR']), 
                float(map_params['theta_ISF'])
            )
            
            # Compute bg_velocity with (possibly adjusted) carbs
            bg_velocity_batch = compute_bg_velocity_batch(
                params,
                current_glucose_array,
                historical_cgm_array_batch,
                ins_times_array,
                ins_amounts_array,
                carb_times_array,
                carb_amounts_to_use
            )
            
            # Get Loop predictions with (possibly adjusted) carbs
            _, mu = predict_cgm_loop_vectorized(
                params,
                current_glucose_array,
                ins_times_array,
                ins_amounts_array,
                ins_n_array,
                carb_times_array,
                carb_amounts_to_use,
                carb_n_array,
                historical_cgm_array_batch,
                bg_velocity_batch
            )
            
            # Extract MAP noise parameters
            phi = float(map_params.get('phi', 0.85))
            sigma_eta = float(map_params.get('sigma_eta', 1.0))
            sigma_obs = float(map_params.get('sigma_obs', 1.0))
            
            # Sample AR(1) bias realizations
            eta_innovations = jax.random.normal(key_eta, (num_instances, T)) * sigma_eta
            
            # Build AR(1) trajectory
            b0 = jnp.zeros(num_instances)
            
            def ar1_step(b_prev, eta_k):
                b_new = phi * b_prev + eta_k
                return b_new, b_new
            
            def scan_per_instance(b0_i, etas_i):
                _, bs_i = jax.lax.scan(ar1_step, b0_i, etas_i)
                bs_full_i = jnp.concatenate([b0_i[None], bs_i[:-1]], axis=0)
                return bs_full_i
            
            bs_full = jax.vmap(scan_per_instance)(b0, eta_innovations)
            pred_mean = mu + bs_full
            
            # Sample observation noise
            obs_noise = jax.random.normal(key_obs, (num_instances, T)) * sigma_obs
            obs_samples = pred_mean + obs_noise
            
            samples_pred_mean.append(np.array(mu))
            samples_bias.append(np.array(bs_full))
            samples_obs.append(np.array(obs_samples))
        
        # Stack: [num_instances, T, num_samples]
        return {
            'pred_mean': np.stack(samples_pred_mean, axis=-1),
            'bias': np.stack(samples_bias, axis=-1),
            'obs': np.stack(samples_obs, axis=-1)
        }