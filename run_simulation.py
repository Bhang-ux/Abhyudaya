# --- 6. Main Simulation ---
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import Atmosphere
import acceleration
import get_control_action
from constants import GRAVITY, ROCKET_AREA, ROCKET_MASS, vs, TARGET_APOGEE, H_INIT, V_INIT


def run_simulation(is_controlled):
    h = H_INIT
    vr = V_INIT
    time = 0
    DT=0.1
    
    
    time_history = [0]
    h_history = [h]
    v_history = [vr]
    delta_history = [0]
    
    while vr > 0:
        if is_controlled:
            delta = get_control_action.get_control_action(h, vr, TARGET_APOGEE, 5)
        else:
            delta = 0
        
        accel = acceleration.acceleration(h, vr, delta)
        
        
        vr += accel * DT
        h += vr * DT
        time += DT
        
        # Store results
        time_history.append(time)
        h_history.append(h)
        v_history.append(vr)
        delta_history.append(delta)
        
    return np.array(time_history), np.array(h_history), np.array(delta_history)

# Run both cases
if __name__ == '__main__':
    time_uncontrolled, alt_uncontrolled, _ = run_simulation(is_controlled=False)
    time_controlled, alt_controlled, delta_controlled = run_simulation(is_controlled=True)

    uncontrolled_apogee = alt_uncontrolled.max()
    controlled_apogee   = alt_controlled.max()

    absolute_error      = abs(controlled_apogee - TARGET_APOGEE)
    percentage_error    = absolute_error / TARGET_APOGEE * 100
    overshoot           = abs(uncontrolled_apogee - TARGET_APOGEE)
    improvement_pct     = (overshoot - absolute_error) / overshoot * 100
    total_timesteps     = len(time_controlled)

    activation_time = None
    for i, d in enumerate(delta_controlled):
        if d > 0:
            activation_time = time_controlled[i]
            break

    print(f"\n── Simulation Metrics ───────────────────────────────")
    print(f"Uncontrolled apogee:       {uncontrolled_apogee:.2f} m")
    print(f"Controlled apogee:         {controlled_apogee:.2f} m")
    print(f"Target apogee:             {TARGET_APOGEE:.2f} m")
    print(f"Absolute error:            {absolute_error:.2f} m")
    print(f"Percentage error:          {percentage_error:.4f} %")
    print(f"Uncontrolled overshoot:    {overshoot:.2f} m")
    print(f"Improvement:               {improvement_pct:.2f} %")
    print(f"Total timesteps:           {total_timesteps}")
    print(f"Airbrake activation time:  {activation_time:.1f} s")