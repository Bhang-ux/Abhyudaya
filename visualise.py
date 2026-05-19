import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from Abhyudaya.run_simulation import run_simulation
from Abhyudaya import predict_apogee
from Abhyudaya.constants import TARGET_APOGEE

# Run both simulations
time_uncontrolled, alt_uncontrolled, delta_uncontrolled = run_simulation(is_controlled=False)
time_controlled,   alt_controlled,   delta_controlled   = run_simulation(is_controlled=True)

# Compute controller error at each timestep
error_history = []
from Abhyudaya.run_simulation import run_simulation
from Abhyudaya import predict_apogee, get_control_action
from Abhyudaya.constants import TARGET_APOGEE, vs

# Rerun controlled to capture predicted apogee at each step
h = 1500
vr = 2 * vs
DT = 0.1
error_history = []
time_err = []
t = 0
while vr > 0:
    pred = predict_apogee.predict_apogee(h, vr, 0)
    error_history.append(pred - TARGET_APOGEE)
    time_err.append(t)
    from Abhyudaya.acceleration import acceleration
    delta = get_control_action.get_control_action(h, vr, TARGET_APOGEE, 5)
    from Abhyudaya.acceleration import acceleration
    accel = acceleration(h, vr, delta)
    vr += accel * DT
    h  += vr * DT
    t  += DT

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F9FA',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

os.makedirs('plots', exist_ok=True)

# ── Plot 1: Altitude vs Time ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time_uncontrolled, alt_uncontrolled, color='#E07B39', linewidth=2, label='Uncontrolled')
ax.plot(time_controlled,   alt_controlled,   color='#2E75B6', linewidth=2, label='Controlled')

# Mark apogees with stars
unc_idx = np.argmax(alt_uncontrolled)
con_idx = np.argmax(alt_controlled)
ax.plot(time_uncontrolled[unc_idx], alt_uncontrolled[unc_idx], '*', color='#E07B39', markersize=15, zorder=5)
ax.plot(time_controlled[con_idx],   alt_controlled[con_idx],   '*', color='#2E75B6', markersize=15, zorder=5)

# Annotate apogee values
ax.annotate(f"{alt_uncontrolled[unc_idx]:.0f} m", xy=(time_uncontrolled[unc_idx], alt_uncontrolled[unc_idx]),
            xytext=(8, 8), textcoords='offset points', color='#E07B39', fontweight='bold')
ax.annotate(f"{alt_controlled[con_idx]:.0f} m", xy=(time_controlled[con_idx], alt_controlled[con_idx]),
            xytext=(8, 8), textcoords='offset points', color='#2E75B6', fontweight='bold')

# Target line
ax.axhline(TARGET_APOGEE, color='green', linewidth=1.5, linestyle='--', label=f'Target: {TARGET_APOGEE} m')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Altitude (m)', fontsize=12)
ax.set_title('Altitude vs. Time — Controlled vs. Uncontrolled', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/plot1_altitude_vs_time.png', dpi=150)
plt.close()
print("Plot 1 saved.")

# ── Plot 2: Airbrake Deflection vs Time ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ax.fill_between(time_controlled, delta_controlled, alpha=0.3, color='#2E75B6')
ax.plot(time_controlled, delta_controlled, color='#2E75B6', linewidth=2, label='Airbrake Deflection')
ax.axhline(50, color='red', linewidth=1, linestyle='--', label='Max deflection (50°)')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Deflection Angle (°)', fontsize=12)
ax.set_title('Airbrake Deflection vs. Time', fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/plot2_airbrake_deflection.png', dpi=150)
plt.close()
print("Plot 2 saved.")

# ── Plot 3: Velocity vs Altitude (Phase Portrait) ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Need velocity history — rerun to capture it
def get_v_history(controlled):
    h, vr, DT = 1000, 0.75*vs, 0.1
    from Abhyudaya.acceleration import acceleration
    v_hist, h_hist = [vr], [h]
    while vr > 0:
        delta = get_control_action.get_control_action(h, vr, TARGET_APOGEE, 5) if controlled else 0
        accel = acceleration(h, vr, delta)
        vr += accel * DT
        h  += vr * DT
        v_hist.append(vr)
        h_hist.append(h)
    return np.array(h_hist), np.array(v_hist)

h_unc, v_unc = get_v_history(False)
h_con, v_con = get_v_history(True)

ax.plot(v_unc, h_unc, color='#E07B39', linewidth=2, label='Uncontrolled')
ax.plot(v_con, h_con, color='#2E75B6', linewidth=2, label='Controlled')
ax.axhline(TARGET_APOGEE, color='green', linewidth=1.5, linestyle='--', label='Target Apogee')
ax.axvline(0, color='gray', linewidth=1, linestyle=':')

ax.set_xlabel('Velocity (m/s)', fontsize=12)
ax.set_ylabel('Altitude (m)', fontsize=12)
ax.set_title('Phase Portrait — Velocity vs. Altitude', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/plot3_phase_portrait.png', dpi=150)
plt.close()
print("Plot 3 saved.")

# ── Plot 4: Controller Error over Time ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(time_err, error_history, color='#C00000', linewidth=2, label='Predicted Apogee Error')
ax.fill_between(time_err, error_history, alpha=0.15, color='#C00000')
ax.axhline(0, color='green', linewidth=1.5, linestyle='--', label='Zero Error (perfect)')
ax.axhspan(-50, 50, alpha=0.08, color='green', label='±50 m acceptable band')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Error (m)  [Predicted Apogee − Target]', fontsize=12)
ax.set_title('Controller Error over Time — Apogee Prediction Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/plot4_controller_error.png', dpi=150)
plt.close()
print("Plot 4 saved.")

print("\nAll 4 plots saved to the plots/ folder.")