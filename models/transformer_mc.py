"""
Transformer-MC: Transformer-encoder forecaster with Monte Carlo dropout
(train_and_sample_transformer). Its training loop is also used by the
Transformer-Ensemble and Transformer-Conformal variants.
"""
import copy
import math
import os

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=96):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Transformer_MC(nn.Module):
    """
    Decoder-free Transformer encoder regressor for the same 96-step (4h
    history + 4h known-future-feature) -> 48-step glucose forecasting task
    as LSTM_MC. Deliberately higher-capacity than LSTM_MC (d_model=64 vs.
    LSTM hidden_dim=32, plus multi-head attention) so it can serve as an
    architecture-capacity comparator.

    Same MC-Dropout UQ mechanism as LSTM_MC: dropout stays active at
    inference (model.train()) and predictions are resampled num_mc_samples
    times.
    """

    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, output_dim=48, dropout=0.2, max_len=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: [batch, 96, 4]
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        # Take the representation at the last time step of the history,
        # mirroring LSTM_MC's out[:, -1, :] readout.
        h = self.dropout(h[:, -1, :])
        return self.fc(h)


def train_and_sample_transformer(x_train, y_train, x_test, num_mc_samples=100, max_epochs=2000,
                                  patience=100, use_val_selection=False, artifact_path=None,
                                  artifact_extra=None):
    # Ablation-only overrides via env vars (see models/mnode.py for the same
    # pattern) -- defaults unchanged, production call site never sets these.
    patience = int(os.environ.get('TRANSFORMER_PATIENCE_OVERRIDE', patience))
    use_val_selection = os.environ.get(
        'TRANSFORMER_USE_VAL_SELECTION_OVERRIDE', '1' if use_val_selection else '0') == '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════════════════════════════════════════════════════════════
    # 1. INPUT FEATURE SCALING (Z-Score Normalization) — identical to LSTM_MC
    # ════════════════════════════════════════════════════════════════════════
    x_train_np = np.array(x_train)  # [batch, time, features]
    x_test_np = np.array(x_test)

    x_mean = x_train_np.mean(axis=(0, 1), keepdims=True)
    x_std = x_train_np.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train_scaled = (x_train_np - x_mean) / x_std
    x_test_scaled = (x_test_np - x_mean) / x_std

    y_scale = 100.0

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(device) / y_scale
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    # Ablation-only: see models/lstm_mc.py's identical validation-split block.
    x_fit_t, y_fit_t, x_val_t, y_val_t = x_train_t, y_train_t, None, None
    if use_val_selection:
        n_train = x_train_t.shape[0]
        n_val = max(1, int(round(n_train * 0.2)))
        n_fit = n_train - n_val
        x_fit_t, x_val_t = x_train_t[:n_fit], x_train_t[n_fit:]
        y_fit_t, y_val_t = y_train_t[:n_fit], y_train_t[n_fit:]

    # ════════════════════════════════════════════════════════════════════════
    # 2. MODEL SETUP & TRAINING
    # ════════════════════════════════════════════════════════════════════════
    model = Transformer_MC(
        input_dim=x_train_t.shape[-1],
        max_len=x_train_t.shape[1],
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_val_state = None

    print(f"Training Transformer-MC (patience={patience}, use_val_selection={use_val_selection}, "
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

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4} | Loss: {loss.item():.6f}"
                  + (f" | Val: {monitored_loss:.6f}" if use_val_selection else ""))

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if use_val_selection and best_val_state is not None:
        model.load_state_dict(best_val_state)
        print(f"  Restored best-validation checkpoint (val loss={best_loss:.6f})")

    # Optional weight persistence for counterfactual re-forecasting -- see
    # models/lstm_mc.py's identical block and models/cf_artifacts.py.
    if artifact_path:
        from models.cf_artifacts import save_artifact
        save_artifact(artifact_path, kind='transformer_mc', backbone='transformer',
                      state_dicts=[{k: v.cpu() for k, v in model.state_dict().items()}],
                      input_dim=int(x_train_t.shape[-1]), max_len=int(x_train_t.shape[1]),
                      dropout=0.2,
                      x_mean=x_mean, x_std=x_std, y_scale=y_scale,
                      num_mc_samples=num_mc_samples,
                      x_test=x_test_np, **(artifact_extra or {}))

    # ════════════════════════════════════════════════════════════════════════
    # 3. INFERENCE WITH ANCHORING TRICK — identical to LSTM_MC
    # ════════════════════════════════════════════════════════════════════════
    model.train()  # Keep dropout active for MC sampling
    samples = []

    last_cgm_test = torch.tensor(x_test_np[:, 47, 0:1], dtype=torch.float32).to(device)

    with torch.no_grad():
        for _ in range(num_mc_samples):
            raw_output = model(x_test_t) * y_scale
            bias_correction = last_cgm_test - raw_output[:, 0:1]
            pred_anchored = raw_output + bias_correction
            samples.append(pred_anchored.cpu().numpy())

    return np.array(samples)
