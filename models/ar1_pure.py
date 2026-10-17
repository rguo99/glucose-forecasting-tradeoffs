"""
Pure AR1 baseline: glucose as an AR(1) process around the current value,
with no insulin or carbohydrate inputs (AR1PureModel, fitted with NUTS).
"""
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

class AR1PureModel():
    """
    Pure AR(1) model for blood glucose dynamics.
    Baseline model that models glucose as an AR(1) process decaying/persisting
    from the current glucose value (t=0), without Loop insulin/carb dynamics.
    
    Parameters:
    - phi: AR(1) autocorrelation (0.80 to 0.99)
    - sigma_eta: AR(1) process noise
    - sigma_obs: Observation noise
    """
    
    def __init__(self, config=None):
        config = config or {}
        self.param_names = ['phi', 'sigma_eta', 'sigma_obs']
    
    @staticmethod
    @jax.jit
    def ar1_log_likelihood(y, mu_scalar, phi, sigma_eta, sigma_obs):
        """
        Kalman filter log-likelihood for AR(1).
        mu_scalar: The starting glucose value (t=0) for this instance.
        """
        T = y.shape[0]
        
        def kalman_step(carry, yt):
            m, P, log_lik = carry
            m_pred = phi * m
            P_pred = phi**2 * P + sigma_eta**2
            
            # Prediction mean is baseline + AR deviation
            y_pred_mean = mu_scalar + m_pred
            y_pred_var = P_pred + sigma_obs**2
            
            residual = yt - y_pred_mean
            
            log_lik_t = jax.scipy.stats.norm.logpdf(
                residual, 0, jnp.sqrt(y_pred_var + 1e-8)
            )
            
            K = P_pred / (y_pred_var + 1e-8)
            m_new = m_pred + K * residual
            P_new = (1 - K) * P_pred
            return (m_new, P_new, log_lik + log_lik_t), None
        
        m0, P0 = 0.0, 1e-8
        (_, _, total_log_lik), _ = jax.lax.scan(kalman_step, (m0, P0, 0.0), y)
        return total_log_lik
    
    def model_fn(self, 
                 current_glucose_array,        # ARG 1: Shape (N,)
                 ins_times_array=None,         # ARG 2: Unused inputs
                 ins_amounts_array=None,       # ARG 3
                 ins_n_array=None,             # ARG 4
                 carb_times_array=None,        # ARG 5
                 carb_amounts_array=None,      # ARG 6
                 carb_n_array=None,            # ARG 7
                 historical_cgm_array_batch=None, # ARG 8
                 observed_glucose_array=None,  # ARG 9: Target (N, T)
                 sample_predictions=False):
        
        # current_glucose_array comes in as (N,), ensure it's handled as 1D
        num_instances = current_glucose_array.shape[0]
        
        # --- PRIORS ---
        sigma_obs = numpyro.sample('sigma_obs', dist.LogNormal(loc=2.85, scale=0.03))
        sigma_eta = numpyro.sample('sigma_eta', dist.LogNormal(loc=1.10, scale=0.03))
        
        phi_raw = numpyro.sample("phi_raw", dist.Beta(10, 2))
        phi = numpyro.deterministic("phi", 0.80 + (phi_raw * 0.19))
        
        # --- BASELINE MEAN ---
        # Use the CURRENT glucose (t=0) as the persistence baseline.
        # This allows valid forecasting when observed_glucose_array is None.
        mu_all = current_glucose_array  # Shape (N,)
        
        # --- LIKELIHOOD (Training) ---
        if observed_glucose_array is not None and not sample_predictions:
            # Vectorize over instances (N)
            log_liks = jax.vmap(lambda y, m: self.ar1_log_likelihood(y, m, phi, sigma_eta, sigma_obs))(
                observed_glucose_array, mu_all
            )
            numpyro.factor("obs_likelihood", jnp.sum(log_liks))
        
        # --- PREDICTION (Forecasting) ---
        if sample_predictions:
            # We need T to define the output shape. 
            # If observed_glucose_array is None, assume 48 steps (standard horizon).
            if observed_glucose_array is not None:
                T = observed_glucose_array.shape[1]
            else:
                T = 48
                
            # Sample latent AR(1) bias
            eta_std = numpyro.sample('eta_std', dist.Normal(jnp.zeros((num_instances, T)), 1.0))
            etas = sigma_eta * eta_std
            
            def ar1_step(b_prev, eta_k):
                b_new = phi * b_prev + eta_k
                return b_new, b_new
            
            # Generate bias trajectories
            _, bs_full = jax.vmap(lambda b0, e: jax.lax.scan(ar1_step, b0, e))(
                jnp.zeros(num_instances), etas
            )
            
            # Reconstruct glucose: Baseline + Bias
            mu_broadcast = mu_all[:, jnp.newaxis] # (N, 1)
            mean_gl = mu_broadcast + bs_full      # (N, T)
            
            # Sample observations with .to_event(2) to preserve (N, T) structure
            # This ensures Predictive output is (Samples, N, T)
            numpyro.sample('obs', dist.Normal(mean_gl, sigma_obs).to_event(2))
            
            numpyro.deterministic('pred_mean', mean_gl)
            numpyro.deterministic('bias', bs_full)
            
    def extract_params(self, samples):
        return {k: samples[k] for k in self.param_names if k in samples}
    
    def get_param_names_for_plotting(self):
        return self.param_names