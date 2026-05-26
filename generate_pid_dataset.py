import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
import pandas as pd
import time
from Abhyudaya.optimise_pid import find_optimal_pid
from Abhyudaya.constants import vs
 
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS — saved in current working directory
# ─────────────────────────────────────────────────────────────────────────────
# What: save files relative to where you run the script from
# Why: avoids path errors regardless of where the script lives
PARTIAL_SAVE_PATH = 'pid_dataset_partial.csv'
FINAL_SAVE_PATH   = 'pid_dataset.csv'
 
# ─────────────────────────────────────────────────────────────────────────────
# DEFINE THE GRID OF FLIGHT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
# What: define ranges of H_INIT, V_INIT, TARGET_APOGEE to cover
# Why: the NN needs many different conditions to learn the mapping
#      (H_INIT, V_INIT, TARGET) → (optimal Kp, Ki, Kd)
#
# 5 × 5 × 5 = 125 total conditions
# Each runs 50 Optuna trials → 6250 PID simulations total
H_INIT_VALUES = np.linspace(500,   3000,  6)
V_INIT_VALUES = np.linspace(1.0*vs, 2*vs, 6)
TARGET_VALUES = np.linspace(3000,  10000, 6)
 
total_conditions = len(H_INIT_VALUES) * len(V_INIT_VALUES) * len(TARGET_VALUES)
 
print(f"Generating PID dataset...")
print(f"H_INIT values:  {H_INIT_VALUES.round(0)}")
print(f"V_INIT values:  {V_INIT_VALUES.round(1)}")
print(f"TARGET values:  {TARGET_VALUES.round(0)}")
print(f"Total conditions: {total_conditions}")
print(f"Optuna trials per condition: 50")
print(f"Saving to: {os.path.abspath(FINAL_SAVE_PATH)}")
print("-" * 60)
 
# ─────────────────────────────────────────────────────────────────────────────
# GENERATE THE DATASET
# ─────────────────────────────────────────────────────────────────────────────
records    = []
count      = 0
start_time = time.time()
 
for h_init in H_INIT_VALUES:
    for v_init in V_INIT_VALUES:
        for target in TARGET_VALUES:
            count += 1
 
            # What: skip physically impossible conditions
            # Why: airbrakes can only slow the rocket down, not speed it up
            #      if target is above max possible apogee, no PID can help
            max_possible = h_init + (v_init**2) / (2 * 9.81)
 
            if target > max_possible * 0.95:
                print(f"[{count:3d}/{total_conditions}] SKIP "
                      f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m "
                      f"(unreachable, max≈{max_possible:.0f}m)")
                continue
 
            if target <= h_init:
                print(f"[{count:3d}/{total_conditions}] SKIP "
                      f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m "
                      f"(target below burnout altitude)")
                continue
 
            # What: also skip conditions where overshoot is tiny
            # Why: if uncontrolled apogee barely exceeds target, airbrakes
            #      have almost no work to do — not interesting training data
            overshoot_ratio = max_possible / target
            if overshoot_ratio < 1.05:
                print(f"[{count:3d}/{total_conditions}] SKIP "
                      f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m "
                      f"(overshoot too small: {overshoot_ratio:.2f}x)")
                continue
 
            # What: run Optuna to find optimal PID gains for this condition
            # Why: generates one labelled training example
            try:
                best_params, best_error = find_optimal_pid(
                    h_init, v_init, target,
                    n_trials=50,
                    verbose=False
                )
 
                # What: only keep conditions where Optuna found a good solution
                # Why: if best error is huge the gains are useless training data
                #      threshold of 100m = 1% error on smallest target (3000m)
                if best_error > 50:
                    print(f"[{count:3d}/{total_conditions}] POOR "
                          f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m "
                          f"→ Error={best_error:.1f}m (skipping — too large)")
                    continue
 
                record = {
                    'H_INIT':  round(h_init,  2),
                    'V_INIT':  round(v_init,  2),
                    'TARGET':  round(target,  2),
                    'Kp':      best_params['Kp'],
                    'Ki':      best_params['Ki'],
                    'Kd':      best_params['Kd'],
                    'error':   best_error
                }
                records.append(record)
 
                elapsed   = time.time() - start_time
                rate      = count / elapsed
                remaining = (total_conditions - count) / rate / 60
 
                print(f"[{count:3d}/{total_conditions}] OK   "
                      f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m "
                      f"→ Kp={best_params['Kp']:.4f} "
                      f"Ki={best_params['Ki']:.6f} "
                      f"Kd={best_params['Kd']:.4f} "
                      f"Err={best_error:.4f}m "
                      f"[~{remaining:.0f}min left]")
 
            except Exception as e:
                print(f"[{count:3d}/{total_conditions}] FAIL "
                      f"H={h_init:.0f}m V={v_init:.0f}m/s T={target:.0f}m: {e}")
                continue
 
            # What: save progress every 10 valid records
            # Why: if script crashes you keep all work done so far
            if len(records) % 10 == 0 and records:
                pd.DataFrame(records).to_csv(PARTIAL_SAVE_PATH, index=False)
                print(f"  → Progress saved ({len(records)} valid records)")
 
# ─────────────────────────────────────────────────────────────────────────────
# SAVE FINAL DATASET
# ─────────────────────────────────────────────────────────────────────────────
if not records:
    print("\nNo valid records generated. Check flight conditions.")
    sys.exit(1)
 
df         = pd.DataFrame(records)
total_time = (time.time() - start_time) / 60
 
df.to_csv(FINAL_SAVE_PATH, index=False)
 
print(f"\n{'='*60}")
print(f"Dataset generation complete.")
print(f"Valid conditions:  {len(records)} / {total_conditions}")
print(f"Skipped:           {total_conditions - count} (unreachable)")
print(f"Poor solutions:    {count - len(records)} (error > 100m)")
print(f"Total time:        {total_time:.1f} minutes")
print(f"Saved to:          {os.path.abspath(FINAL_SAVE_PATH)}")
print(f"\nDataset summary:")
print(df.describe().round(6))
print(f"\nRun nn_pid_predictor.py next.")