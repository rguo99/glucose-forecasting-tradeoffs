"""
Conversion of per-instance CGM/insulin/carbohydrate data into model inputs:
padded event arrays relative to the forecast origin (Bayesian Loop and the AR
models) and 5-minute aligned sequence tensors (neural models).
"""
import pandas as pd
import numpy as np
import jax.numpy as jnp
from models.loop_model import PREDICTION_MINUTES, TIMESTEP

import torch

import os

# Read the environment variable. Default to False if it doesn't exist.
SYNTHETIC = os.environ.get("SYNTHETIC_DATA_MODE", "false").lower() == "true"

if SYNTHETIC:
    print(">>> data_prep.py: SYNTHETIC MODE ENABLED (Basal Unrolling Bypassed)")




def pad_events(ev_list, max_len, pad_value=-1e6):
    times = np.full((len(ev_list), max_len), pad_value, dtype=np.float32)
    amounts = np.zeros((len(ev_list), max_len), dtype=np.float32)
    n_arr = np.zeros(len(ev_list), dtype=np.int32)
    for idx, evs in enumerate(ev_list):
        n = min(len(evs), max_len)
        n_arr[idx] = n
        for i in range(n):
            times[idx, i] = evs[i]["time"]
            amounts[idx, i] = evs[i]["amount"]
    return jnp.array(times), jnp.array(amounts), jnp.array(n_arr)


def prepare_user_level_arrays(historical_datas, future_datas, max_insulin_events=140, max_carb_events=None):
    """
    Prepare user-level arrays for inference.
    
    Prediction horizon is now dynamically determined from loop_model.PREDICTION_MINUTES
    (120 min for optimized accuracy, can be changed to 240 min if needed).
    
    Args:
        historical_datas: List of historical data dicts
        future_datas: List of future data dicts
        max_insulin_events: Max number of insulin events to pad to (basal + bolus)
                           Default 140 (~1 event per ~10 minutes over 24 hours)
        max_carb_events: Max number of carb events to pad to
                        Default None (auto-calculate from data: typically 6-8)
    
    Returns:
        Tuple of arrays:
        - current_glucose: [num_instances]
        - ins_t: [num_instances, max_insulin_events]
        - ins_a: [num_instances, max_insulin_events]
        - ins_n: [num_instances]
        - carb_t: [num_instances, max_carb_events]
        - carb_a: [num_instances, max_carb_events]
        - carb_n: [num_instances]
        - hist_cgm_array_batch: [num_instances, max_hist_len]
        - observed_glucose_array: [num_instances, PREDICTION_STEPS]
    """
    num_instances = len(historical_datas)

    current_glucose = jnp.array([hist['cgm']['cgm'].iloc[-1] for hist in historical_datas])
    current_glucose = jnp.atleast_1d(current_glucose)  # Ensure batch dim

    # Prepare insulin events list per instance (combine basal + bolus and convert to event dict list)
    insulin_events_list = []
    carb_events_list = []

    hist_cgm_arrays = []

    observed_glucose = []

    # Prediction parameters
    prediction_minutes = PREDICTION_MINUTES
    timestep = TIMESTEP
    prediction_steps = prediction_minutes // timestep

    for i in range(num_instances):
        hist = historical_datas[i]
        fut = future_datas[i]
        
        # 1. Anchor Time (Zero-point for relative minutes)
        prediction_start = hist['cgm']['timestamp'].iloc[-1]
        prediction_start_naive = pd.to_datetime(prediction_start).tz_localize(None)

        # 2. Setup Clean Dataframes
        all_basal_df = pd.concat([hist['basal_events'], fut['basal_events']]).sort_values('timestamp').copy()
        all_basal_df['timestamp'] = pd.to_datetime(all_basal_df['timestamp']).dt.tz_localize(None)
        
        all_bolus_df = pd.concat([hist['bolus_events'], fut['bolus_events']]).sort_values('timestamp').copy()
        all_bolus_df['timestamp'] = pd.to_datetime(all_bolus_df['timestamp']).dt.tz_localize(None)

        # 3. Create a fresh list for this specific instance
        this_instance_insulin = []

        # 4. PROCESS BOLUSES (Using convert_to_relative_minutes as requested)
        actual_bolus_df = all_bolus_df[all_bolus_df['bolus'] > 0].copy()
        if not actual_bolus_df.empty:
            # We use your function to handle the time conversion
            actual_bolus_df = convert_to_relative_minutes(actual_bolus_df, prediction_start)
            for _, row in actual_bolus_df.iterrows():
                this_instance_insulin.append({
                    'time': float(row['time']),
                    'amount': float(row['bolus']),
                    'type': 'BOLUS'
                })


        ### UNCOMMENT BASAL FOR REAL DATA EXPT ###
        ### COMMENT BASAL FOR SYNTHETIC EXPT ###
        # use basal for real experiments; don't use for synthetic
        if not SYNTHETIC:
            # 5. UNROLL BASAL (Zero-Order Hold Logic)
            # This provides the continuous insulin pressure the model needs
            for step in range(-48, 48):
                rel_minute = step * 5.0
                t_step = prediction_start_naive + pd.Timedelta(minutes=rel_minute)
                
                # Find the most recent basal rate set at or before this 5-min mark
                idx = all_basal_df['timestamp'].searchsorted(t_step, side='right') - 1
                idx = max(0, idx)
                
                if not all_basal_df.empty:
                    # rate = all_basal_df.iloc[idx]['basal_rate']
                    # # Convert U/hr to the dose delivered in a 5-minute interval
                    # dose = (rate / 60.0) * 5.0 
                    dose = all_basal_df.iloc[idx]['basal_units']
                    # Delivered basal (U per 5-min slot); see README section 5.
                    this_instance_insulin.append({
                        'time': float(rel_minute),
                        'amount': float(dose),
                        'type': 'BASAL'
                    })

        ### DO NOT UNCOMMENT ###

        # # 5. UNROLL BASAL (Zero-Order Hold Logic with Schedule Offset) - SYNTHETIC EXPT
        # for step in range(-48, 48):
        #     rel_minute = step * 5.0
        #     t_step = prediction_start_naive + pd.Timedelta(minutes=rel_minute)
            
        #     idx = all_basal_df['timestamp'].searchsorted(t_step, side='right') - 1
        #     idx = max(0, idx)
            
        #     # if not all_basal_df.empty:
        #     #     rate_delivered = all_basal_df.iloc[idx]['basal_rate']
        #     #     # THE KEY FIX: Subtract the scheduled baseline (1.0)
        #     #     # This prevents the "Liver Blindness" that was shrinking your ISF
        #     #     net_rate = rate_delivered - 1.0 
                
        #     #     dose = (net_rate / 60.0) * 5.0 
        #     #     this_instance_insulin.append({
        #     #         'time': float(rel_minute),
        #     #         'amount': float(dose),
        #     #         'type': 'BASAL'
        #     #     })

        #     # Inside the UNROLLED BASAL loop
        #     if not all_basal_df.empty:
        #         rate_delivered = all_basal_df.iloc[idx]['basal_rate']
        #         # For this test, we want JAX to see NO basal insulin
        #         # If the generator basal is 0, this should be 0.
        #         net_rate = 0.0 
                
        #         dose = (net_rate / 60.0) * 5.0 
        #         this_instance_insulin.append({
        #             'time': float(rel_minute),
        #             'amount': float(dose),
        #             'type': 'BASAL'
        #         })

        ### DO NOT UNCOMMENT ###

        # 6. THE ONLY APPEND (One entry per instance)
        insulin_events_list.append(this_instance_insulin)


        all_meal_df = pd.concat([hist['meal_events'], fut['meal_events']]).sort_values('timestamp').reset_index(drop=True)
        all_meal_df["timestamp"] = pd.to_datetime(all_meal_df["timestamp"])
        all_meal_df = convert_to_relative_minutes(all_meal_df, hist['cgm']['timestamp'].iloc[-1])

        carb_events_list.append(prepare_event_arrays(all_meal_df))

        hist_cgm_arrays.append(jnp.array(hist['cgm']['cgm'].values))

        # Generate prediction timestamps for the horizon defined in loop_model
        pred_start = hist['cgm']['timestamp'].iloc[-1]
        pred_times = pd.date_range(
            pred_start + pd.Timedelta(minutes=timestep), 
            periods=prediction_steps, 
            freq=f"{timestep}min"
        )
        obs_glucose_aligned = prepare_observed_glucose(fut['cgm'], pred_times)
        observed_glucose.append(np.array(obs_glucose_aligned))

    # Determine actual max carb events if not provided
    if max_carb_events is None:
        actual_max_carbs = max((len(evs) for evs in carb_events_list), default=1)
        max_carb_events = actual_max_carbs
        print(f"Auto-determined max_carb_events: {max_carb_events} (actual max: {actual_max_carbs})")
    
    # Pad insulin and carbs with their respective max events
    ins_t, ins_a, ins_n = pad_events(insulin_events_list, max_len=max_insulin_events)
    carb_t, carb_a, carb_n = pad_events(carb_events_list, max_len=max_carb_events)

    observed_glucose_array = jnp.array(observed_glucose)
    observed_glucose_array = jnp.atleast_1d(observed_glucose_array)  # Ensure batch dim
    
    # Validate observed_glucose_array shape
    assert observed_glucose_array.shape[1] == prediction_steps, \
        f"observed_glucose_array has {observed_glucose_array.shape[1]} timesteps, " \
        f"expected {prediction_steps} (from PREDICTION_MINUTES={prediction_minutes}, TIMESTEP={timestep})"

    # Pad and stack historical CGM arrays to equal length here
    max_hist_len = max(arr.shape[0] for arr in hist_cgm_arrays)
    hist_cgm_arrays_padded = []
    for arr in hist_cgm_arrays:
        pad_width = max_hist_len - arr.shape[0]
        padded = jnp.pad(arr, (0, pad_width), constant_values=0.0)
        hist_cgm_arrays_padded.append(padded)
    hist_cgm_array_batch = jnp.stack(hist_cgm_arrays_padded)
    hist_cgm_array_batch = jnp.atleast_1d(hist_cgm_array_batch)

    return (current_glucose, ins_t, ins_a, ins_n, carb_t, carb_a, carb_n, hist_cgm_array_batch, observed_glucose_array)


def convert_to_relative_minutes(event_df: pd.DataFrame, prediction_start: pd.Timestamp):
    """Convert event timestamps to minutes relative to prediction start time"""
    # Ensure proper datetime dtype to prevent: TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Timestamp'
    event_df = event_df.copy()
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    prediction_start = pd.to_datetime(prediction_start)
    event_df["time"] = (event_df["timestamp"] - prediction_start).dt.total_seconds() / 60.0
    return event_df

def prepare_event_arrays(event_df: pd.DataFrame):
    if event_df.empty:
        return []
    amount_col_candidates = ["amount", "bolus", "carbs", "meal_dose", "basal_rate", "dose", "bolus_dose"]
    amount_col = next((c for c in amount_col_candidates if c in event_df.columns), None)
    if amount_col is None:
        return []
    amounts = event_df[amount_col].fillna(0.0)
    return [{"time": float(t), "amount": float(a)} for t, a in zip(event_df["time"], amounts)]

def prepare_observed_glucose(future_cgm: pd.DataFrame, prediction_timestamps: pd.DatetimeIndex):
    aligned = future_cgm.set_index("timestamp").reindex(prediction_timestamps, method="nearest")["cgm"]
    return jnp.array(aligned.values)


def prepare_lstm_sequence_arrays(historical_datas, future_datas):
    """
    Creates a 5-minute aligned grid for LSTM with 4 channels and 8-hour context.
    X shape: [num_instances, 96, 4]
    """
    import torch
    num_instances = len(historical_datas)
    # 96 steps (4h history + 4h future)
    sequence_data = np.zeros((num_instances, 96, 4)) 

    for i in range(num_instances):
        hist = historical_datas[i]
        fut = future_datas[i]
        prediction_start = hist['cgm']['timestamp'].iloc[-1]
        prediction_start_naive = pd.to_datetime(prediction_start).tz_localize(None)
        
        # 1. CGM CHANNEL (Index 0) - History Only
        # We put the 48 historical points in the first half and leave the rest 0
        sequence_data[i, :48, 0] = hist['cgm']['cgm'].values[-48:]

        # 2. COMBINE DATA FOR 8-HOUR COVERAGE
        basal_df = pd.concat([hist['basal_events'], fut['basal_events']]).copy()
        basal_df['timestamp'] = pd.to_datetime(basal_df['timestamp']).dt.tz_localize(None)
        
        bolus_df = pd.concat([hist['bolus_events'], fut['bolus_events']]).copy()
        bolus_df['timestamp'] = pd.to_datetime(bolus_df['timestamp']).dt.tz_localize(None)
        bolus_col = 'bolus' if 'bolus' in bolus_df.columns else 'bolus_dose'
        
        carb_df = pd.concat([hist['meal_events'], fut['meal_events']]).copy()
        carb_df['timestamp'] = pd.to_datetime(carb_df['timestamp']).dt.tz_localize(None)
        carb_col = 'meal_dose' if 'meal_dose' in carb_df.columns else 'carbs'

        # 3. UNROLL 96 STEPS (-4h to +4h)
        for step_idx, step in enumerate(range(-47, 49)): 
            rel_minute = step * 5.0
            t_step = prediction_start_naive + pd.Timedelta(minutes=rel_minute)
            t_prev = t_step - pd.Timedelta(minutes=5)

            # --- Channel 1: Basal ---
            active_basal = basal_df[basal_df['timestamp'] <= t_step].sort_values('timestamp')
            if not active_basal.empty:
                row = active_basal.iloc[-1]
                if 'basal_units' in row.index:
                    sequence_data[i, step_idx, 1] = row['basal_units']
                elif 'basal_rate' in row.index:
                    sequence_data[i, step_idx, 1] = row['basal_rate'] / 12.0
            
            # --- Channel 2: Bolus ---
            if not bolus_df.empty:
                sequence_data[i, step_idx, 2] = bolus_df[(bolus_df['timestamp'] > t_prev) & 
                                                         (bolus_df['timestamp'] <= t_step)][bolus_col].sum()

            # --- Channel 3: Carbs ---
            if not carb_df.empty:
                sequence_data[i, step_idx, 3] = carb_df[(carb_df['timestamp'] > t_prev) & 
                                                        (carb_df['timestamp'] <= t_step)][carb_col].sum()

    return torch.tensor(sequence_data, dtype=torch.float32)