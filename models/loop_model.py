"""
Loop-style glucose forecast: insulin action curve (rise to a peak, then
exponential decay over the action duration), linear carbohydrate absorption,
initial momentum from recent CGM, and retrospective correction from the last
30 minutes. All functions are
JAX-jitted/vmapped and shared by Bayesian Loop and the data preparation.

Draws on LoopDocs (https://loopkit.github.io/loopdocs/operation/algorithm/prediction)
and PyLoopKit (https://github.com/tidepool-org/PyLoopKit), simplified for use as
an offline forecaster on logged data (see the paper's Methods). PyLoopKit's
license (BSD-2-Clause, Tidepool Project) is in THIRD_PARTY_NOTICES.
"""
import jax
import jax.numpy as jnp
from jax import lax
from collections import namedtuple


import os

# Read the environment variable. Default to False if it doesn't exist.
# We check if the string matches "true" (case-insensitive).
SYNTHETIC = os.environ.get("SYNTHETIC_DATA_MODE", "false").lower() == "true"

if SYNTHETIC:
    print(">>> loop_model.py: SYNTHETIC MODE ENABLED (Heuristics OFF)")
    USE_HEURISTICS = 0.0
else:
    USE_HEURISTICS = 1.0
    
# # ============================================================================
# # EXPERIMENT TOGGLE
# # ============================================================================
# # Set to 0.0 for Synthetic Parameter Recovery (Pure Physics)
# # Set to 1.0 for Real Patient Inference (Includes Loop App Heuristics)
# USE_HEURISTICS = 1.0 
# # ============================================================================

# Constants
MAX_EVENTS = 140  
PAD_VALUE = -1e6  
PREDICTION_MINUTES = 240
TIMESTEP = 5
RETROSPECTIVE_MINUTES = 30  

LoopParams = namedtuple("LoopParams", [
    "carb_ratio", "isf", "insulin_peak_time", "insulin_duration", "carb_absorption_time"
])

def create_loop_params(carb_ratio, isf, insulin_peak_time=75, insulin_duration=360, carb_absorption_time=120):
    return LoopParams(carb_ratio, isf, insulin_peak_time, insulin_duration, carb_absorption_time)

# --- Optimized Math Functions ---

# 1. DEFINED FIRST: The raw insulin curve
@jax.jit
def insulin_action_fraction(t, peak_time, duration):
    t_peak = jnp.clip(t, 0.0, peak_time)
    peak_val = 0.9 * (t_peak / peak_time) ** 2
    decay_progress = jnp.clip((t - peak_time) / (duration - peak_time), 0.0, 1.0)
    decay_val = 0.9 * jnp.exp(-2.5 * decay_progress)
    return jnp.where(t <= peak_time, peak_val, jnp.where(t <= duration, decay_val, 0.0))

# 2. DEFINED SECOND: The norm factor (which relies on the function above)
@jax.jit
def get_insulin_norm_factor(peak_time, duration):
    # Evaluate over a fixed max window (e.g., 600 mins / 10 hours) for JIT safety
    t_array = jnp.arange(0, 600, TIMESTEP)
    raw_rates = insulin_action_fraction(t_array, peak_time, duration)
    return jnp.sum(raw_rates)

@jax.jit
def carb_absorption_fraction(t, carb_absorption_time):
    return jnp.clip(t / (1.5 * carb_absorption_time), 0.0, 1.0)

@jax.jit
def calculate_initial_momentum(cgm_values):
    """Calculates slope using a fixed-size 4-point slice to avoid Tracer errors."""
    array_len = cgm_values.shape[0]
    start_idx = jnp.maximum(0, array_len - 4)
    y = lax.dynamic_slice(cgm_values, (start_idx,), (4,))
    slope = (y[-1] - y[0]) / 3.0 
    return jnp.clip(slope, -4.0, 4.0)

# --- The Master Loop Logic ---

@jax.jit
def predict_cgm_loop_single(
        params, current_glucose,
        ins_times, ins_amounts, ins_n,
        carb_times, carb_amounts, carb_n,
        historical_cgm_array, bg_velocity):
    
    t_steps = (PREDICTION_MINUTES // TIMESTEP)
    times = jnp.arange(TIMESTEP, PREDICTION_MINUTES + 1, TIMESTEP).reshape(t_steps, 1)
    times_prev = times - TIMESTEP

    # Momentum & RC
    initial_momentum = calculate_initial_momentum(historical_cgm_array)
    mom_w = jnp.clip((20.0 - times) / 15.0, 0.0, 1.0) * (times <= 20.0)
    rc_w = jnp.clip((60.0 - times) / 55.0, 0.0, 1.0) * (times <= 60.0)
    
    # Event Matrices
    i_t = ins_times.reshape(1, -1)
    i_a = ins_amounts.reshape(1, -1)
    c_t = carb_times.reshape(1, -1)
    c_a = carb_amounts.reshape(1, -1)

    # --- CORRECTED Insulin Deltas ---
    norm_factor = get_insulin_norm_factor(params.insulin_peak_time, params.insulin_duration)
    dt = times - i_t
    current_ins_rate = insulin_action_fraction(dt, params.insulin_peak_time, params.insulin_duration) / norm_factor
    ins_deltas = jnp.sum(-params.isf * i_a * current_ins_rate, axis=1)

    # --- Carb Deltas ---
    dt_c = times - c_t
    dt_cp = times_prev - c_t
    c_act = carb_absorption_fraction(dt_c, params.carb_absorption_time)
    c_act_p = carb_absorption_fraction(dt_cp, params.carb_absorption_time)
    carb_deltas = jnp.sum((params.isf / params.carb_ratio) * c_a * (c_act - c_act_p), axis=1)

    # --- Integration (Multiplies heuristics by 0.0 or 1.0) ---
    heuristic_deltas = (initial_momentum * mom_w).flatten() + (bg_velocity * rc_w).flatten()
    total_deltas = ins_deltas + carb_deltas + (USE_HEURISTICS * heuristic_deltas)
    
    return times.flatten(), current_glucose + jnp.cumsum(total_deltas)

@jax.jit
def predict_cgm_loop_single_no_rc(
        params, current_glucose,
        ins_times, ins_amounts,
        carb_times, carb_amounts,
        historical_cgm_array):
    
    t_steps = (RETROSPECTIVE_MINUTES // TIMESTEP)
    times = jnp.arange(TIMESTEP, RETROSPECTIVE_MINUTES + 1, TIMESTEP).reshape(t_steps, 1)
    times_prev = times - TIMESTEP
    
    initial_momentum = calculate_initial_momentum(historical_cgm_array)
    mom_w = jnp.clip((20.0 - times) / 15.0, 0.0, 1.0) * (times <= 20.0)

    i_t, i_a = ins_times.reshape(1, -1), ins_amounts.reshape(1, -1)
    c_t, c_a = carb_times.reshape(1, -1), carb_amounts.reshape(1, -1)

    # --- CORRECTED Insulin Deltas ---
    norm_factor = get_insulin_norm_factor(params.insulin_peak_time, params.insulin_duration)
    dt = times - i_t
    current_ins_rate = insulin_action_fraction(dt, params.insulin_peak_time, params.insulin_duration) / norm_factor
    ins_deltas = jnp.sum(-params.isf * i_a * current_ins_rate, axis=1)

    # --- Carb Deltas ---
    dt_c, dt_cp = times - c_t, times_prev - c_t
    carb_deltas = jnp.sum((params.isf / params.carb_ratio) * c_a * (
        carb_absorption_fraction(dt_c, params.carb_absorption_time) - 
        carb_absorption_fraction(dt_cp, params.carb_absorption_time)
    ), axis=1)

    # --- Integration (Multiplies heuristics by 0.0 or 1.0) ---
    heuristic_deltas = (initial_momentum * mom_w).flatten()
    total_deltas = ins_deltas + carb_deltas + (USE_HEURISTICS * heuristic_deltas)
    
    return current_glucose + jnp.sum(total_deltas)

# --- Wrappers ---

@jax.jit
def extract_30min_history_jit(historical_cgm_array):
    # The last element is the forecast origin (t=0), so the reading 30 min
    # earlier is at len-7, and the 4-point momentum window ends at that reading.
    array_len = historical_cgm_array.shape[0]
    idx = jnp.maximum(3, array_len - 1 - (RETROSPECTIVE_MINUTES // TIMESTEP))
    curr_30 = historical_cgm_array[idx]
    hist_30 = lax.dynamic_slice(historical_cgm_array, (idx - 3,), (4,))
    return curr_30, hist_30

@jax.jit
def compute_bg_velocity_from_full_history_jit(
        params, current_glucose_now, historical_cgm_array_full,
        ins_times, ins_amounts, carb_times, carb_amounts):
    cg_30, hist_30 = extract_30min_history_jit(historical_cgm_array_full)
    # Retrospective correction: forecast the last 30 min from t=-30 and
    # compare with the observed glucose now. Event times are relative to the
    # forecast origin (t=0), so they are shifted by +30 min onto the clock of
    # this retrospective forecast, which then applies the insulin and carb
    # effects of the past 30 min (-30..0). Padded events (PAD_VALUE=-1e6)
    # remain far out of range after the shift.
    rc_pred = predict_cgm_loop_single_no_rc(params, cg_30,
                                            ins_times + RETROSPECTIVE_MINUTES, ins_amounts,
                                            carb_times + RETROSPECTIVE_MINUTES, carb_amounts, hist_30)
    return (1.0 / 6.0) * (current_glucose_now - rc_pred)

predict_cgm_loop_vectorized = jax.vmap(predict_cgm_loop_single, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))
compute_bg_velocity_batch = jax.vmap(compute_bg_velocity_from_full_history_jit, in_axes=(None, 0, 0, 0, 0, 0, 0))