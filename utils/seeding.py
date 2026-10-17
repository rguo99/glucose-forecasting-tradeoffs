"""
Deterministic seeding shared by the production pipeline (InferenceRunner.py)
and the counterfactual retrain (dev_scripts/causal_smoke_train.py).

WHY THIS IS SHARED AND NOT DUPLICATED
-------------------------------------
Both paths train the same six neural baselines on the same curation-defined
splits. If they derive seeds the same way from the same base, then for a given
(user_id, model) they enter the trainer with identical RNG state and identical
data, and therefore produce identical weights. That is what makes the archived
counterfactual weights and a future production run the SAME model instance
rather than two draws of it -- which is the whole point, because otherwise the
causal-response claims and the calibration claims describe different models.

SCOPE OF THE REPRODUCIBILITY GUARANTEE -- read before promising anything
-----------------------------------------------------------------------
Seeding buys bitwise identity only within a fixed numeric environment. Verified
bitwise (max_abs_dev = 0.0, including a 5-member ensemble) on one Tesla V100
with CUBLAS_WORKSPACE_CONFIG=:4096:8. It will NOT hold across:
  * a different GPU model      -- different kernels, different reduction order
  * a different torch/cuDNN    -- different default algorithm selection
  * CUBLAS_WORKSPACE_CONFIG unset -- cuBLAS sizes workspaces dynamically, so
    reduction order varies run to run even with every RNG seeded. torch's own
    docs require this for reproducible RNN/LSTM training; torch.manual_seed
    alone is NOT sufficient.
So the honest claim to a third party is "seed S in environment E reproduces our
weights exactly; other hardware gives statistically equivalent models, not
identical ones." Call env_provenance() and persist it alongside any result that
claims reproducibility, so E is recorded rather than assumed.
"""
import hashlib
import os
import random

import numpy as np

# Base seed for the whole project. Changing this invalidates every archived
# (user, model) seed, so treat it as a constant, not a knob. Overridable by
# env var only so that a deliberate multi-seed sensitivity run does not
# require editing tracked source.
SEED_BASE = int(os.environ.get('BAYESIANUQ_SEED_BASE', '0'))


def derive_seed(uid, model, base=None):
    """
    Deterministic per-(user, model) seed.

    sha256 rather than hash(): Python salts string hashing per process, so
    hash() would return a different value in every job and silently defeat the
    point. Independent seeds per (user, model) rather than one global seed so a
    single user/model can be regenerated on its own, without replaying a whole
    shard in order.
    """
    base = SEED_BASE if base is None else base
    h = hashlib.sha256(f'{base}:{uid}:{model}'.encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def seed_everything(seed):
    """
    Seed every RNG the trainers touch, and pin cuDNN to deterministic
    algorithms.

    Deliberately does NOT call torch.use_deterministic_algorithms(True): that
    raises on several cuDNN RNN kernels rather than falling back to a
    deterministic path, which would break training outright instead of making
    it reproducible.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def env_provenance():
    """The numeric environment a seed is only meaningful relative to."""
    import torch
    return {
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'torch_version': torch.__version__,
        'cudnn_version': torch.backends.cudnn.version(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>'),
    }


# Seeds actually APPLIED in this process, keyed by (str(user_id), model).
# utils.plotting.save_user_inference_bundle reads this so every result bundle
# records the seed that produced it -- a seed printed only to a log is not
# reproducibility, because the log and the bundle get separated.
_APPLIED = {}


def applied_seed(uid, model):
    """Seed applied for this (user, model) in this process, or None if the
    model never went through seeding (Bayesian Loop / Pure AR1 use fixed JAX
    PRNGKeys instead and are deterministic without this machinery)."""
    return _APPLIED.get((str(uid), str(model)))


def record_seed(uid, model, seed):
    """Register a seed applied by a path that seeds itself (e.g. AR1-RLS builds
    a jax PRNGKey directly rather than touching the torch RNG)."""
    _APPLIED[(str(uid), str(model))] = int(seed)
    return seed


def seed_for_trainer(uid, model, base=None, verbose=True):
    """Convenience: derive, apply, and return the seed. Call IMMEDIATELY before
    the trainer call -- not once per job -- so a run does not depend on how many
    runs preceded it in the same process (i.e. on shard boundaries)."""
    seed = derive_seed(uid, model, base=base)
    seed_everything(seed)
    record_seed(uid, model, seed)
    if verbose:
        print(f"  [seeding] user={uid} model={model} seed={seed} "
              f"(base={SEED_BASE if base is None else base})", flush=True)
    return seed
