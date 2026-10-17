"""
Transformer-Ensemble: deep ensemble of independently initialized
Transformer_MC networks (train_and_sample_transformer_ensemble).
"""
import copy
import os

import torch
import torch.nn as nn
import numpy as np

from models.transformer_mc import Transformer_MC


def train_and_sample_transformer_ensemble(x_train, y_train, x_test, num_members=5,
                                          samples_per_member=20, max_epochs=2000,
                                          patience=100, use_val_selection=False,
                                          artifact_path=None, artifact_extra=None):
    """
    Deep-ensemble UQ comparator for Transformer_MC -- the diagonal cell of the
    architecture x UQ-method 2x2 that the rest of the pipeline already covers:

                    MC-Dropout      Deep Ensemble      Split-Conformal
        LSTM        lstm_mc         lstm_ensemble      lstm_conformal
        Transformer transformer_mc  THIS FILE          transformer_conformal

    Motivation: swapping UQ method on a FIXED LSTM backbone moves
    60-min hypoglycaemia AUROC by up to +0.068 (LSTM-MC 0.8377 -> LSTM-Ensemble
    0.9055), which is larger than Bayesian Loop's entire margin over the field.
    That makes "UQ method is a nuisance variable we controlled for" untenable
    without testing whether the same gain transfers to the higher-capacity
    Transformer backbone. LSTM-Ensemble is the strongest non-Bayesian-Loop
    model overall (best RMSE, best 4-hr Trajectory Correlation, 2nd on hypo
    AUROC at 30/60/90 min), so Transformer+Ensemble is the single most likely
    configuration to beat Bayesian Loop at the <=2-hr horizons where the
    paper's central clinical claim lives. This exists to test that claim
    adversarially rather than assume it.

    Construction is IDENTICAL to models/lstm_ensemble.py (same member count,
    same per-member deterministic forward pass with dropout off, same per-
    horizon-step residual-std aleatoric term, same total sample count) -- only
    the backbone changes, so any difference vs. lstm_ensemble is attributable
    to architecture alone, and any difference vs. transformer_mc is
    attributable to UQ method alone. Both comparisons stay clean.

    Each of `num_members` Transformer_MC instances is trained independently
    from a different random initialization (identical data/objective
    otherwise). At inference each member makes one deterministic forward pass
    (dropout off) -- member disagreement is the epistemic-uncertainty signal.
    Each member's point forecast is then perturbed by `samples_per_member`
    draws of Gaussian noise scaled by that member's own per-horizon-step
    training-residual std, yielding a predictive distribution directly
    comparable to Transformer_MC's num_mc_samples=100 output.
    Total samples returned = num_members * samples_per_member (default 100).
    """
    # Ablation-only overrides via env vars (same pattern as
    # models/lstm_ensemble.py) -- defaults unchanged, production call site
    # never sets these.
    patience = int(os.environ.get('TRANSFORMER_ENSEMBLE_PATIENCE_OVERRIDE', patience))
    use_val_selection = os.environ.get(
        'TRANSFORMER_ENSEMBLE_USE_VAL_SELECTION_OVERRIDE',
        '1' if use_val_selection else '0') == '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════════════════════════════════════════════════════════════
    # 1. INPUT FEATURE SCALING (identical to Transformer_MC / LSTM_MC)
    # ════════════════════════════════════════════════════════════════════════
    x_train_np = np.array(x_train)
    x_test_np = np.array(x_test)

    x_mean = x_train_np.mean(axis=(0, 1), keepdims=True)
    x_std = x_train_np.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train_scaled = (x_train_np - x_mean) / x_std
    x_test_scaled = (x_test_np - x_mean) / x_std

    y_scale = 100.0

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(device) / y_scale
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    last_cgm_train = torch.tensor(x_train_np[:, 47, 0:1], dtype=torch.float32).to(device)
    last_cgm_test = torch.tensor(x_test_np[:, 47, 0:1], dtype=torch.float32).to(device)

    # Ablation-only: see models/lstm_ensemble.py's identical validation-split
    # block. Applied identically per member below (each member still trains
    # from a different random init, only the fit/val split is shared).
    x_fit_t, y_fit_t, x_val_t, y_val_t = x_train_t, y_train_t, None, None
    if use_val_selection:
        n_train = x_train_t.shape[0]
        n_val = max(1, int(round(n_train * 0.2)))
        n_fit = n_train - n_val
        x_fit_t, x_val_t = x_train_t[:n_fit], x_train_t[n_fit:]
        y_fit_t, y_val_t = y_train_t[:n_fit], y_train_t[n_fit:]

    all_samples = []
    member_state_dicts = []
    member_resid_stds = []

    for member_idx in range(num_members):
        # ════════════════════════════════════════════════════════════════════
        # 2. TRAIN ONE INDEPENDENTLY-INITIALIZED MEMBER
        # ════════════════════════════════════════════════════════════════════
        torch.manual_seed(member_idx)
        model = Transformer_MC(
            input_dim=x_train_t.shape[-1],
            max_len=x_train_t.shape[1],
            dropout=0.2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        best_val_state = None

        print(f"Training Transformer-Ensemble member {member_idx + 1}/{num_members} "
              f"(patience={patience}, use_val_selection={use_val_selection}, "
              f"Max {max_epochs} epochs)...")
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
                    val_loss = criterion(model(x_val_t), y_val_t).item()
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

        # ════════════════════════════════════════════════════════════════════
        # 3. DETERMINISTIC POINT FORECAST + ANCHORING (dropout OFF)
        # ════════════════════════════════════════════════════════════════════
        model.eval()
        with torch.no_grad():
            train_raw = model(x_train_t) * y_scale
            train_bias_correction = last_cgm_train - train_raw[:, 0:1]
            train_anchored = (train_raw + train_bias_correction).cpu().numpy()

            test_raw = model(x_test_t) * y_scale
            test_bias_correction = last_cgm_test - test_raw[:, 0:1]
            test_anchored = (test_raw + test_bias_correction).cpu().numpy()

        # Member's own aleatoric noise scale: empirical residual std on its own
        # training set, PER HORIZON STEP (not pooled) -- identical to
        # lstm_ensemble.py, where pooling was a real bug that flattened the PI
        # so it was already wide at the anchor (step 0 is forced to match the
        # last known CGM exactly, so its residual is near zero) instead of
        # fanning out with lead time. Keep this per-step.
        resid_std = np.std(np.array(y_train) - train_anchored, axis=0)  # [horizon]

        rng = np.random.default_rng(member_idx)
        noise = rng.normal(loc=0.0, scale=resid_std[None, None, :],
                           size=(samples_per_member, *test_anchored.shape))
        member_samples = test_anchored[None, :, :] + noise
        all_samples.append(member_samples)

        if artifact_path:
            member_state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})
            member_resid_stds.append(resid_std)

    # Optional weight persistence for counterfactual re-forecasting -- see
    # models/lstm_mc.py's identical block and models/cf_artifacts.py.
    if artifact_path:
        from models.cf_artifacts import save_artifact
        save_artifact(artifact_path, kind='transformer_ensemble', backbone='transformer',
                      state_dicts=member_state_dicts,
                      input_dim=int(x_train_t.shape[-1]), max_len=int(x_train_t.shape[1]),
                      dropout=0.2,
                      x_mean=x_mean, x_std=x_std, y_scale=y_scale,
                      resid_std=np.stack(member_resid_stds, axis=0),
                      samples_per_member=samples_per_member,
                      x_test=x_test_np, **(artifact_extra or {}))

    return np.concatenate(all_samples, axis=0)
