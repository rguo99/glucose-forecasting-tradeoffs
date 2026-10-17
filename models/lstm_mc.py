"""
LSTM-MC: two-layer LSTM forecaster with Monte Carlo dropout for predictive
uncertainty (train_and_sample_lstm).
"""
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTM_MC(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=48, dropout=0.2):
        super().__init__()
        # 2 layers with 32 units: stable capacity for small datasets
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch, 96, 4]
        out, _ = self.lstm(x)
        # Take the hidden state from the last time step of the history
        out = self.dropout(out[:, -1, :]) 
        return self.fc(out)

def train_and_sample_lstm(x_train, y_train, x_test, num_mc_samples=100, max_epochs=2000,
                           patience=100, use_val_selection=False, artifact_path=None,
                           artifact_extra=None):
    # Ablation-only overrides via env vars (see models/mnode.py for the same
    # pattern) -- defaults above are unchanged, and the main pipeline's call
    # site (InferenceRunner.py) never sets these, so production LSTM_MC runs
    # are unaffected regardless of when this file is (re)imported.
    patience = int(os.environ.get('LSTM_MC_PATIENCE_OVERRIDE', patience))
    use_val_selection = os.environ.get(
        'LSTM_MC_USE_VAL_SELECTION_OVERRIDE', '1' if use_val_selection else '0') == '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════════════════════════════════════════════════════════════
    # 1. INPUT FEATURE SCALING (Z-Score Normalization)
    # ════════════════════════════════════════════════════════════════════════
    x_train_np = np.array(x_train) # [batch, time, features]
    x_test_np = np.array(x_test)

    # Calculate statistics across batch and time dimensions (axis 0, 1)
    # This puts CGM (200) and Bolus (1.5) on the same 'volume' level
    x_mean = x_train_np.mean(axis=(0, 1), keepdims=True)
    x_std = x_train_np.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train_scaled = (x_train_np - x_mean) / x_std
    x_test_scaled = (x_test_np - x_mean) / x_std

    # Target scaling for stability
    y_scale = 100.0

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(device) / y_scale
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    # Ablation-only: carve the last 20% of the (chronologically-ordered)
    # training instances into a validation fold for checkpoint selection,
    # matching the authors' own protocol (track val loss per epoch, keep the
    # best checkpoint) instead of the patience-on-train-loss + final-epoch-
    # weights approach every baseline in this pipeline currently uses.
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
    model = LSTM_MC(input_dim=x_train_t.shape[-1], hidden_dim=32, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    best_val_state = None

    print(f"Training Anchored LSTM-MC (patience={patience}, use_val_selection={use_val_selection}, "
          f"Max {max_epochs} epochs)...")
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        preds = model(x_fit_t)
        loss = criterion(preds, y_fit_t)
        loss.backward()
        # Gradient clipping prevents the 'RMSE 120' instability
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

    # Optional: persist weights + preprocessing state so this forecast can be
    # re-run later on modified inputs (counterfactual / dose-response). Purely
    # additive -- off unless a caller passes artifact_path. See
    # models/cf_artifacts.py for why x_mean/x_std/y_scale must travel with the
    # state_dict.
    if artifact_path:
        from models.cf_artifacts import save_artifact
        save_artifact(artifact_path, kind='lstm_mc', backbone='lstm',
                      state_dicts=[{k: v.cpu() for k, v in model.state_dict().items()}],
                      input_dim=int(x_train_t.shape[-1]), max_len=int(x_train_t.shape[1]),
                      hidden_dim=32, dropout=0.2,
                      x_mean=x_mean, x_std=x_std, y_scale=y_scale,
                      num_mc_samples=num_mc_samples,
                      x_test=x_test_np, **(artifact_extra or {}))

    # ════════════════════════════════════════════════════════════════════════
    # 3. INFERENCE WITH ANCHORING TRICK
    # ════════════════════════════════════════════════════════════════════════
    model.train() # Keep dropout active for MC sampling
    samples = []
    
    # CRITICAL: Anchor to the REAL CGM value at t=0 (Index 47)
    # last_cgm_test shape: [batch, 1]
    last_cgm_test = torch.tensor(x_test_np[:, 47, 0:1], dtype=torch.float32).to(device)

    with torch.no_grad():
        for _ in range(num_mc_samples):
            # A. Get Absolute Prediction and scale back to mg/dL
            raw_output = model(x_test_t) * y_scale
            
            # B. THE ANCHORING TRICK
            # Find the bias between model t=0 and ground truth t=0
            bias_correction = last_cgm_test - raw_output[:, 0:1]
            
            # C. Shift the whole trajectory to align with history
            pred_anchored = raw_output + bias_correction
            
            samples.append(pred_anchored.cpu().numpy())
    
    return np.array(samples)