# models/ar1_rls.py
# Adaptive AR(p) baseline via Recursive Least Squares (RLS) with an
# exponential forgetting factor (Adaptive AR1-RLS in the paper; order 1).
#
# Unlike Pure AR1 (models/ar1_pure.py), which fits one set of AR parameters
# per user, this model re-identifies the AR coefficients online for every
# forecast instance from that instance's own historical CGM window, following
# the recursively identified AR models of Sparacino et al. (2007,
# 10.1109/TBME.2006.889774). The order is configurable (RLS_ORDER); order 1
# reduces to a 1x1 companion matrix, so there is a single implementation.
# Like Pure AR1, it uses only the glucose history (no insulin/carb inputs).
#
# Why deviations from the window mean rather than a free intercept: fitting
# [phi, intercept] jointly on the raw glucose level (y_t on [y_{t-1}, 1]) is
# only identifiable when y_{t-1} has
# enough variance within the window; on a flat/plateau 4-hour window (common
# for real CGM data), y_{t-1} and the constant regressor are nearly
# collinear, so phi and intercept trade off along a ridge -- many wildly
# different (phi, intercept) pairs fit the training window equally well, and
# 47 recursive RLS steps can land on a point along that ridge that
# extrapolates to nonsense (glucose diverging toward -750 mg/dL over 4
# hours) despite fitting the window itself fine. Fitting AR coefficients
# alone on deviations from the window's own mean removes the intercept
# entirely, so there's nothing to trade off against.

import jax
import jax.numpy as jnp

DEFAULT_ORDER = 1                  # AR(p); Sparacino et al. used ~30. See module docstring.
DEFAULT_FORGETTING_FACTOR = 0.98   # lambda in [0.90, 0.99] per Sparacino-style adaptive AR
DEFAULT_DELTA = 1e4                # initial parameter variance (diffuse prior)
MIN_RESIDUAL_STD = 1.0             # mg/dL floor so degenerate windows don't collapse to zero noise
SPECTRAL_RADIUS_CLIP = 0.999       # keeps the point-estimate rollout from diverging

# Pure numerical-divergence backstop, deliberately far wider than the CGM
# device range so RLS stays free to predict negative/implausible values the
# same way Bayesian Loop's mechanistic model can -- not a physiological clamp.
# For order=1 the mean forecast is geometrically bounded near
# [mean_hist, y_origin] and this should never engage; for order>1 with only
# a spectral-radius-clipped (not re-derived) sampled coefficient draw, it's
# a real safety net -- see rls_forecast_samples. Report engagement via
# forecast_was_clamped and treat any nonzero rate on the deterministic path
# as a red flag; a low nonzero rate on the *sampled* path at higher orders
# is expected and not itself a bug.
GLUCOSE_MIN = -1000.0
GLUCOSE_MAX = 2000.0


def _companion_matrix(a):
    """AR(p) companion matrix for coefficients a (p,): top row = a, sub-diagonal = 1."""
    p = a.shape[0]
    C = jnp.zeros((p, p)).at[0, :].set(a)
    if p > 1:
        C = C.at[1:, :-1].set(jnp.eye(p - 1))
    return C


def _spectral_radius(a):
    C = _companion_matrix(a)
    eigs = jnp.linalg.eigvals(C)
    return jnp.max(jnp.abs(eigs))


def _nearest_psd(P, min_eig=1e-6):
    """
    Project P onto the nearest valid covariance matrix (symmetric, all
    eigenvalues >= min_eig). The RLS covariance recursion accumulates
    floating-point error over many iterations -- more so at higher AR
    order, where lagged regressors are more likely to be correlated -- and
    P can end up with negative eigenvalues despite being a "covariance"
    algebraically. A tiny uniform diagonal jitter is not sufficient to fix
    this (observed negative eigenvalues as large in magnitude as -1e4); an
    indefinite P otherwise makes jax.random.multivariate_normal's internal
    Cholesky-like step silently return NaN samples.

    A fixed absolute floor is not enough either: this runs in float32
    (jax_enable_x64=False), which has ~7 significant digits, and P's
    eigenvalues can span 4-5 orders of magnitude (observed range: -6.5e4 to
    +2.4e4). Reconstructing V @ diag(eigvals) @ V.T loses the small-
    eigenvalue directions to rounding error from the large one
    (float32_eps * largest_eigenvalue -- e.g. ~0.003 when the largest
    eigenvalue is ~2.4e4), so an absolute floor of 1e-6 leaves some
    reconstructed eigenvalues slightly negative again. Floor relative to
    the matrix's own scale instead.
    """
    P_sym = (P + P.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(P_sym)
    scale = jnp.max(jnp.abs(eigvals))
    floor = jnp.maximum(min_eig, scale * 1e-4)
    eigvals_clipped = jnp.maximum(eigvals, floor)
    P_recon = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    # Re-symmetrize (matmul can reintroduce tiny asymmetry) and add one more
    # scale-relative jitter as a final defense-in-depth guard.
    P_recon = (P_recon + P_recon.T) / 2
    return P_recon + jnp.eye(P.shape[0]) * floor


def _clip_to_stable(a, shrink=0.95, max_iters=100):
    """Uniformly shrink AR coefficients toward 0 until the companion matrix's
    spectral radius is at most SPECTRAL_RADIUS_CLIP.

    IMPORTANT: for order=1 the companion matrix is just [[a]], so scaling a
    by s scales the eigenvalue by exactly s -- a single closed-form clip
    works. For order>1, scaling the coefficient vector by s only scales the
    companion matrix's top row (the sub-diagonal identity block is
    untouched), so the spectral radius is NOT linear in s and a one-shot
    `a * (clip / r)` rescale does not reliably land under the clip (it can
    still leave the radius above 1). Instead, iterate: shrink by a fixed
    factor and recheck the actual spectral radius each time, so convergence
    doesn't depend on an incorrect linearity assumption.
    """
    def cond(state):
        a_i, i, r = state
        return jnp.logical_and(r > SPECTRAL_RADIUS_CLIP, i < max_iters)

    def body(state):
        a_i, i, _ = state
        a_next = a_i * shrink
        return a_next, i + 1, _spectral_radius(a_next)

    r0 = _spectral_radius(a)
    was_clipped = r0 > SPECTRAL_RADIUS_CLIP
    a_final, n_iters, r_final = jax.lax.while_loop(cond, body, (a, 0, r0))
    return a_final, was_clipped


def rls_step(carry, xy, forgetting_factor):
    """
    Single multivariate Recursive Least Squares update with forgetting factor.

    carry: (a, P) -- AR coefficient vector (p,) and its covariance (p, p).
    xy: (x_t, b_t) -- lagged-deviation regressor vector (p,) [most recent
        lag first] and target deviation b_t.
    """
    a, P = carry
    x_t, b_t = xy

    e_t = b_t - x_t @ a
    Px = P @ x_t
    denom = forgetting_factor + x_t @ Px
    K = Px / denom

    a_new = a + K * e_t
    P_new = (P - jnp.outer(K, Px)) / forgetting_factor

    return (a_new, P_new), e_t


def _lagged_design_matrix(b_hist, order):
    """
    x_seq[j] = [b_{j+order-1}, b_{j+order-2}, ..., b_{j}] (most recent lag
    first), b_seq[j] = b_{j+order}, for j = 0 .. len(b_hist)-order-1.
    """
    T = b_hist.shape[0]
    x_seq = jnp.stack([b_hist[order - 1 - i: T - 1 - i] for i in range(order)], axis=1)
    b_seq = b_hist[order:]
    return x_seq, b_seq


def rls_fit_window(y_hist, order=DEFAULT_ORDER, forgetting_factor=DEFAULT_FORGETTING_FACTOR,
                    delta=DEFAULT_DELTA):
    """
    Recursively fit an AR(p) model (deviations from the window mean) over a
    single historical CGM window, ending at the forecast origin.

    Args:
        y_hist: (T,) chronological historical glucose values; y_hist[-1] is
                the current glucose (the forecast origin). Must have T > order.
        order: AR order p (1 = pure persistence/decay, no momentum; p>1 can
                represent trend/velocity continuation -- see module docstring).
        forgetting_factor: lambda, exponential weight on past data (0.90-0.99
                typical; lower = faster adaptation / shorter effective memory).
        delta: initial variance P_0 = delta * I (diffuse prior, so the first
                few updates behave like ordinary least squares).

    Returns:
        a_final: (p,) adapted AR coefficients at the forecast origin
                 [lag1, lag2, ..., lagp], stability-clipped.
        P_final: (p, p) coefficient covariance at the forecast origin
        mu: scalar, the window mean used as the local baseline
        resid_std: scalar std of the one-step-ahead RLS innovations
        was_clipped: bool, True if the raw fitted spectral radius exceeded
                the stability bound before clipping -- report the rate in
                QC material rather than assuming it's rare.
    """
    mu = jnp.mean(y_hist)
    b_hist = y_hist - mu

    x_seq, b_seq = _lagged_design_matrix(b_hist, order)

    a0 = jnp.zeros(order)
    P0 = delta * jnp.eye(order)

    def step(carry, xy):
        return rls_step(carry, xy, forgetting_factor)

    (a_final, P_final), innovations = jax.lax.scan(step, (a0, P0), (x_seq, b_seq))

    # Short (47-sample) windows can still fit an unstable AR(p) -- one whose
    # companion-matrix spectral radius exceeds 1 -- that diverges over a
    # 48-step rollout. Clip at the source so every downstream consumer
    # inherits a stable coefficient vector.
    a_final, was_clipped = _clip_to_stable(a_final)

    resid_std = jnp.sqrt(jnp.mean(innovations ** 2) + 1e-6)
    resid_std = jnp.maximum(resid_std, MIN_RESIDUAL_STD)

    return a_final, P_final, mu, resid_std, was_clipped


def rls_forecast_mean(a, mu, y_hist, horizon):
    """
    Deterministic multi-step rollout using the final adapted coefficients.
    Operates in deviation-from-mean space, carrying a sliding buffer of the
    last `order` deviations (most recent first), seeded from the tail of
    the historical window.

    Returns (y_future, was_clamped): was_clamped is True if the
    [GLUCOSE_MIN, GLUCOSE_MAX] numerical-divergence guard changed any step
    of the rollout. For order=1 this is geometrically bounded and should
    never engage; for order>1 a stability-clipped `a` (spectral radius <=
    SPECTRAL_RADIUS_CLIP) keeps the recursion bounded too, so this remains
    a backstop, not an expected code path.
    """
    order = a.shape[0]
    mu_arr = mu  # scalar
    b_hist = y_hist - mu_arr
    buf0 = jnp.flip(b_hist[-order:])  # most-recent-first

    def step(buf, _):
        raw_next = a @ buf
        new_buf = jnp.concatenate([raw_next[None], buf[:-1]])
        return new_buf, raw_next

    _, b_future = jax.lax.scan(step, buf0, None, length=horizon)
    raw_y_future = mu_arr + b_future
    y_future = jnp.clip(raw_y_future, GLUCOSE_MIN, GLUCOSE_MAX)
    return y_future, jnp.any(y_future != raw_y_future)


def rls_forecast_samples(a, P, mu, resid_std, y_hist, horizon, rng_key, num_samples):
    """
    Forecast-distribution samples combining RLS parameter uncertainty
    (a, P at the forecast origin) with residual-based observation noise, in
    the same deviation-from-mean, sliding-buffer space as rls_forecast_mean.

    NOTE: sampled coefficient draws are NOT individually spectral-radius
    clipped (that would require an eigendecomposition per sample per
    instance -- 500 x N per user -- which is unnecessarily expensive for a
    per-instance closed-form baseline). Instead each rollout step is
    clamped to [GLUCOSE_MIN, GLUCOSE_MAX], which is the real reason that
    backstop exists at order>1: expect a low but nonzero
    forecast_was_clamped rate on this path as order grows, and report it.
    """
    order = a.shape[0]
    b_hist = y_hist - mu
    buf0 = jnp.flip(b_hist[-order:])

    # Project onto the nearest valid covariance before the Cholesky in
    # multivariate_normal -- see _nearest_psd's docstring for why a tiny
    # uniform diagonal jitter alone is not enough at order>1.
    P_safe = _nearest_psd(P)

    def one_sample(key):
        key_a, key_noise = jax.random.split(key)
        # The RLS posterior covariance of the coefficients is sigma^2 * P
        # (P alone is the unscaled inverse information matrix). Each draw is
        # clipped to the same stability bound as the point estimate (exact for
        # order 1, where the spectral radius is |a|).
        a_s = jax.random.multivariate_normal(key_a, a, (resid_std ** 2) * P_safe)
        if order == 1:
            a_s = jnp.clip(a_s, -SPECTRAL_RADIUS_CLIP, SPECTRAL_RADIUS_CLIP)
        noise = jax.random.normal(key_noise, (horizon,)) * resid_std

        def step(buf, eps):
            raw_next = a_s @ buf + eps
            new_buf = jnp.concatenate([raw_next[None], buf[:-1]])
            return new_buf, raw_next

        _, b_traj = jax.lax.scan(step, buf0, noise)
        raw_traj = mu + b_traj
        return jnp.clip(raw_traj, GLUCOSE_MIN, GLUCOSE_MAX)

    keys = jax.random.split(rng_key, num_samples)
    return jax.vmap(one_sample)(keys)  # (num_samples, horizon)


def fit_and_forecast_rls_batch(historical_cgm_array_batch, horizon, order=DEFAULT_ORDER,
                                forgetting_factor=DEFAULT_FORGETTING_FACTOR,
                                delta=DEFAULT_DELTA, num_samples=500, rng_key=None):
    """
    Batched RLS fit + forecast, one independent fit per instance.

    Args:
        historical_cgm_array_batch: (N, hist_len) — the same array
            InferenceRunner.prepare_user_level_arrays produces for every
            other model. hist_len is the 4-hour history (48 steps at the
            5-minute TIMESTEP); the last column is the forecast origin.
            Must have hist_len > order.
        horizon: number of forecast steps (48 for the 4-hour horizon).
        order: AR order p -- see module docstring.
        num_samples: number of forecast-distribution samples to draw per
            instance (for CRPS/coverage-style metrics).

    Returns:
        dict with:
            'pred_mean': (N, horizon) deterministic point forecast
            'obs':       (num_samples, N, horizon) forecast samples
            'coefs':     (N, order) final adapted AR coefficients per instance
            'mu':        (N,) local baseline (historical window mean) per instance
            'resid_std': (N,) one-step-ahead residual std per instance
            'was_clipped': (N,) bool, stability clip engaged during fit
            'forecast_was_clamped': (N,) bool, numerical-divergence guard
                engaged during the deterministic rollout.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def fit_one(y_hist):
        a, P, mu, resid_std, was_clipped = rls_fit_window(y_hist, order, forgetting_factor, delta)
        pred_mean, forecast_was_clamped = rls_forecast_mean(a, mu, y_hist, horizon)
        return a, P, mu, resid_std, was_clipped, pred_mean, forecast_was_clamped

    (coefs, Ps, mus, resid_stds, was_clipped,
     pred_means, forecast_was_clamped) = jax.vmap(fit_one)(historical_cgm_array_batch)

    n_instances = historical_cgm_array_batch.shape[0]
    keys = jax.random.split(rng_key, n_instances)

    def sample_one(a, P, mu, resid_std, y_hist, key):
        return rls_forecast_samples(a, P, mu, resid_std, y_hist, horizon, key, num_samples)

    samples = jax.vmap(sample_one)(coefs, Ps, mus, resid_stds, historical_cgm_array_batch, keys)  # (N, S, H)
    samples = jnp.transpose(samples, (1, 0, 2))                                                    # (S, N, H)

    return {
        'pred_mean': pred_means,
        'obs': samples,
        'coefs': coefs,
        'mu': mus,
        'resid_std': resid_stds,
        'was_clipped': was_clipped,
        'forecast_was_clamped': forecast_was_clamped,
    }

