# --- 6. Main Simulation ---
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Abhyudaya import Atmosphere, acceleration, get_control_action
from Abhyudaya.constants import GRAVITY, ROCKET_AREA, ROCKET_MASS, vs


def run_simulation(is_controlled):
    h = 1000

    vr=0.75*vs
    time = 0
    DT=0.1
    
    
    time_history = [0]
    h_history = [h]
    v_history = [vr]
    delta_history = [0]
    
    while vr > 0:
        if is_controlled:
            delta = get_control_action.get_control_action(h, vr, 3000, 5)
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
time_uncontrolled, alt_uncontrolled, _ = run_simulation(is_controlled=False)
time_controlled, alt_controlled, delta_controlled = run_simulation(is_controlled=True)

print(f"Uncontrolled Apogee: {alt_uncontrolled.max():.2f} m")
print(f"Controlled Apogee: {alt_controlled.max():.2f} m")