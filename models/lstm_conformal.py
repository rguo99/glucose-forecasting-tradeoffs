"""
LSTM-Conformal: LSTM point forecaster with split-conformal prediction
intervals from a held-out calibration fold (train_and_sample_lstm_conformal).
"""
import copy
import os

import torch
import torch.nn as nn
import numpy as np

from models.lstm_mc import LSTM_MC


def train_and_sample_lstm_conformal(x_train, y_train, x_test, calib_frac=0.2,
                                     alpha=0.05, num_samples=100, max_epochs=2000,
                                     patience=100, use_val_selection=False,
                                     artifact_path=None, artifact_extra=None):
    """
    Split-conformal UQ comparator for LSTM_MC ("MC-Dropout
    is a known weak UQ method... I suggest authors add further comparators
    (deep ensembles or conformal calibration)"). Same architecture and input
    grid as LSTM_MC; only the UQ mechanism changes.

    Calibration set is carved out of the TRAINING period (last `calib_frac`
    of the chronologically-ordered training instances), never the test
    period -- using test-period data to calibrate would (a) leak future
    information into earlier test-instance predictions, since instances are
    chronological with only a washout gap enforced, and (b) give this
    comparator an information advantage the other UQ methods (MC-Dropout,
    deep ensemble, AR1-RLS, Bayesian Loop) don't have, since none of them
    see any test-period data before forecasting. Restricting calibration to
    pre-test data keeps the comparison apples-to-apples.

    Tradeoff to disclose in Methods: the proper-training subset used to fit
    the point predictor is smaller than LSTM_MC's (e.g. ~80% of the ~150
    per-user training instances), since some training data is spent on
    calibration instead of model fitting -- inherent to split-conformal, not
    a bug.

    Conformal quantile is computed per horizon step (nonconformity grows
    with lead time, same qualitative pattern as PI width in the other
    models). A symmetric Gaussian is fit around each test point-forecast
    whose (alpha/2, 1-alpha/2) quantiles equal the conformalized interval
    exactly, then `num_samples` are drawn from it, so this model's output
    plugs into the same samples-array interface (and downstream
    coverage/PI-width/interval-score code) as every other UQ comparator.
    """
    # Ablation-only overrides via env vars (see models/mnode.py for the same
    # pattern) -- defaults unchanged, production call site never sets these.
    patience = int(os.environ.get('LSTM_CONFORMAL_PATIENCE_OVERRIDE', patience))
    use_val_selection = os.environ.get(
        'LSTM_CONFORMAL_USE_VAL_SELECTION_OVERRIDE', '1' if use_val_selection else '0') == '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_np = np.array(x_train)  # already chronologically ordered
    x_test_np = np.array(x_test)
    y_train_np = np.array(y_train)

    n_train = x_train_np.shape[0]
    n_calib = max(1, int(round(n_train * calib_frac)))
    n_proper = n_train - n_calib
    assert n_proper >= 1, "Not enough training instances to split off a calibration fold."

    x_proper_np, x_calib_np = x_train_np[:n_proper], x_train_np[n_proper:]
    y_proper_np, y_calib_np = y_train_np[:n_proper], y_train_np[n_proper:]

    # ════════════════════════════════════════════════════════════════════════
    # 1. INPUT FEATURE SCALING -- stats from the proper-train subset only
    # ════════════════════════════════════════════════════════════════════════
    x_mean = x_proper_np.mean(axis=(0, 1), keepdims=True)
    x_std = x_proper_np.std(axis=(0, 1), keepdims=True) + 1e-6

    x_proper_scaled = (x_proper_np - x_mean) / x_std
    x_calib_scaled = (x_calib_np - x_mean) / x_std
    x_test_scaled = (x_test_np - x_mean) / x_std

    y_scale = 100.0

    x_proper_t = torch.tensor(x_proper_scaled, dtype=torch.float32).to(device)
    y_proper_t = torch.tensor(y_proper_np, dtype=torch.float32).to(device) / y_scale
    x_calib_t = torch.tensor(x_calib_scaled, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    last_cgm_proper = torch.tensor(x_proper_np[:, 47, 0:1], dtype=torch.float32).to(device)
    last_cgm_calib = torch.tensor(x_calib_np[:, 47, 0:1], dtype=torch.float32).to(device)
    last_cgm_test = torch.tensor(x_test_np[:, 47, 0:1], dtype=torch.float32).to(device)

    # ════════════════════════════════════════════════════════════════════════
    # 2. TRAIN ON PROPER-TRAIN SUBSET ONLY
    # ════════════════════════════════════════════════════════════════════════
    model = LSTM_MC(input_dim=x_proper_t.shape[-1], hidden_dim=32, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    best_val_state = None
    # Ablation-only: reuse the already-held-out calib fold for validation-
    # based checkpoint selection too. It's already excluded from gradient
    # updates, so tracking val loss on it during training leaks nothing new
    # beyond what's already used -- and it's used for its actual conformal-
    # quantile role afterward exactly as before, unaffected either way.
    y_calib_t = torch.tensor(y_calib_np, dtype=torch.float32).to(device) / y_scale

    # With validation selection, the checkpoint is chosen on the calibration fold.
    x_fit_t, y_fit_t, x_sel_t, y_sel_t = x_proper_t, y_proper_t, x_calib_t, y_calib_t
    print(f"Training LSTM-Conformal (proper-train n={n_proper}, calib n={n_calib}, "
          f"patience={patience}, use_val_selection={use_val_selection}, Max {max_epochs} epochs)...")
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        preds = model(x_fit_t)
        loss = criterion(preds, y_fit_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if use_val_selection:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_sel_t), y_sel_t).item()
            model.train()
            monitored_loss = val_loss
        else:
            monitored_loss = loss.item()

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            patience_counter = 0
            if use_val_selection:
                best_val_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if use_val_selection and best_val_state is not None:
        model.load_state_dict(best_val_state)

    # ════════════════════════════════════════════════════════════════════════
    # 3. DETERMINISTIC POINT FORECASTS (dropout OFF) + ANCHORING
    # ════════════════════════════════════════════════════════════════════════
    model.eval()
    with torch.no_grad():
        calib_raw = model(x_calib_t) * y_scale
        calib_bias_correction = last_cgm_calib - calib_raw[:, 0:1]
        calib_anchored = (calib_raw + calib_bias_correction).cpu().numpy()

        test_raw = model(x_test_t) * y_scale
        test_bias_correction = last_cgm_test - test_raw[:, 0:1]
        test_anchored = (test_raw + test_bias_correction).cpu().numpy()

    # ════════════════════════════════════════════════════════════════════════
    # 4. SPLIT-CONFORMAL QUANTILE PER HORIZON STEP (finite-sample corrected)
    # ════════════════════════════════════════════════════════════════════════
    nonconformity = np.abs(y_calib_np - calib_anchored)  # [n_calib, horizon]
    rank = int(np.ceil((n_calib + 1) * (1 - alpha)))
    rank = min(rank, n_calib)  # clip if calib fold is small
    q_hat = np.sort(nonconformity, axis=0)[rank - 1, :]  # [horizon]

    # ════════════════════════════════════════════════════════════════════════
    # 5. GAUSSIAN SAMPLES WHOSE (alpha/2, 1-alpha/2) QUANTILES MATCH q_hat
    # ════════════════════════════════════════════════════════════════════════
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    sigma = q_hat / z  # [horizon]

    rng = np.random.default_rng(0)
    noise = rng.normal(loc=0.0, scale=sigma[None, None, :],
                        size=(num_samples, *test_anchored.shape))
    samples = test_anchored[None, :, :] + noise

    # Optional weight persistence for counterfactual re-forecasting -- see
    # models/lstm_mc.py's identical block and models/cf_artifacts.py.
    if artifact_path:
        from models.cf_artifacts import save_artifact
        save_artifact(artifact_path, kind='lstm_conformal', backbone='lstm',
                      state_dicts=[{k: v.cpu() for k, v in model.state_dict().items()}],
                      input_dim=int(x_proper_t.shape[-1]), max_len=int(x_proper_t.shape[1]),
                      hidden_dim=32, dropout=0.2,
                      x_mean=x_mean, x_std=x_std, y_scale=y_scale,
                      q_hat=q_hat, sigma=sigma, alpha=alpha,
                      n_proper=n_proper, n_calib=n_calib,
                      x_test=x_test_np, **(artifact_extra or {}))

    return samples
