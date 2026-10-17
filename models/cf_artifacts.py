"""
cf_artifacts.py -- persist trained neural-baseline weights and rebuild a
callable forecaster from them.

WHY THIS EXISTS
---------------
Every neural baseline in models/ is written as a single
`train_and_sample_*(x_train, y_train, x_test)` call that trains, samples, and
returns ONLY the sample array. Nothing capable of producing a *new* forecast
survives the run, so the saved bundles in experiments/*/results/*.pkl cannot
be used for any analysis that needs to re-forecast modified inputs --
counterfactual / dose-response ("causal accuracy") being the case that
motivated this file (see paper/CAUSAL_ACCURACY_PLAN.md).

IMPORTANT: a torch `state_dict` on its own is NOT sufficient to reproduce the
forecast. Each model z-scores its inputs with `x_mean`/`x_std` fitted on that
user's own training instances, divides targets by `y_scale`, and applies the
anchoring trick using channel 0 of the input at step 47. Save the weights
without those and the reloaded model silently produces garbage. Everything
needed is persisted here together.

Artifact contents (single torch .pt file per user x model):
    kind          str   one of the six neural baselines
    backbone      str   'lstm' | 'transformer'
    state_dicts   list  one entry per member (1 for MC / conformal, 5 for ensembles)
    input_dim, max_len, hidden_dim, dropout
    x_mean, x_std np    [1, 1, n_features] z-scoring stats (train-fit)
    y_scale       float
    num_mc_samples int  MC-dropout draws (MC kinds only)
    resid_std     np    [n_members, horizon] per-member aleatoric scale (ensembles)
    q_hat, sigma  np    [horizon] conformal quantile / gaussian scale (conformal)
    x_test        np    [n_test, 96, 4] the exact test-input grid used at run time
    test_instance_ids   list, aligned with x_test rows

The point predictor returned by `load_point_predictor` is DETERMINISTIC given
a fixed seed, which is what makes a counterfactual contrast meaningful: for MC
kinds the dropout masks are regenerated from the same seed on every call, so
baseline and perturbed forecasts share common random numbers and the sampling
noise cancels in the difference rather than swamping it.
"""

import os

import numpy as np
import torch


def save_artifact(path, **fields):
    """Write one artifact file. No-op if `path` is falsy."""
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(fields, path)
    print(f"  [cf_artifacts] saved {fields.get('kind')} artifact -> {path} "
          f"({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


def _build_module(art, device):
    if art['backbone'] == 'lstm':
        from models.lstm_mc import LSTM_MC
        return LSTM_MC(input_dim=art['input_dim'],
                       hidden_dim=art.get('hidden_dim', 32),
                       dropout=art.get('dropout', 0.2)).to(device)
    elif art['backbone'] == 'transformer':
        from models.transformer_mc import Transformer_MC
        return Transformer_MC(input_dim=art['input_dim'],
                              max_len=art['max_len'],
                              dropout=art.get('dropout', 0.2)).to(device)
    raise ValueError(f"unknown backbone {art['backbone']!r}")


def load_artifact(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=False)


def load_point_predictor(path_or_art, device=None, mc_seed=0, mc_samples=None):
    """
    Return `predict(x_np) -> [n_instances, horizon]`, the model's point
    forecast (posterior/ensemble mean) in mg/dL, anchored exactly as at
    training time.

    `x_np` is the raw (un-scaled) [n, 96, 4] input grid, so callers can edit
    the bolus/carb channels directly and re-forecast.

    Additive UQ terms (ensemble `resid_std`, conformal `sigma`) are
    input-independent zero-mean noise, so they are omitted here: they cancel
    identically in any counterfactual difference and only add variance.
    """
    art = load_artifact(path_or_art) if isinstance(path_or_art, str) else path_or_art
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modules = []
    for sd in art['state_dicts']:
        m = _build_module(art, device)
        m.load_state_dict(sd)
        modules.append(m)

    x_mean = np.asarray(art['x_mean'])
    x_std = np.asarray(art['x_std'])
    y_scale = float(art['y_scale'])
    is_mc = art['kind'] in ('lstm_mc', 'transformer_mc')
    n_mc = int(mc_samples or art.get('num_mc_samples') or 100)

    def predict(x_np):
        x_np = np.asarray(x_np, dtype=np.float64)
        x_t = torch.tensor((x_np - x_mean) / x_std, dtype=torch.float32).to(device)
        anchor = torch.tensor(x_np[:, 47, 0:1], dtype=torch.float32).to(device)

        outs = []
        with torch.no_grad():
            if is_mc:
                # dropout stays ON (production UQ mechanism); the seed reset
                # makes the mask sequence identical across calls so baseline
                # and counterfactual are a paired comparison.
                modules[0].train()
                torch.manual_seed(mc_seed)
                for _ in range(n_mc):
                    raw = modules[0](x_t) * y_scale
                    outs.append(raw + (anchor - raw[:, 0:1]))
            else:
                for m in modules:
                    m.eval()
                    raw = m(x_t) * y_scale
                    outs.append(raw + (anchor - raw[:, 0:1]))
        return torch.stack(outs, 0).mean(0).cpu().numpy()

    return predict
