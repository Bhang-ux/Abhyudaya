import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from Abhyudaya.constants import H_INIT, V_INIT, TARGET_APOGEE
from Abhyudaya.pid_controller import run_pid_simulation
from Abhyudaya.optimise_pid import find_optimal_pid
from Abhyudaya.run_simulation import run_simulation

print("=" * 60)
print("SMART SIMULATION — ML-Adaptive PID Control")
print("=" * 60)
print(f"H_INIT         = {H_INIT} m")
print(f"V_INIT         = {V_INIT:.1f} m/s")
print(f"TARGET_APOGEE  = {TARGET_APOGEE} m")
print("=" * 60)

# ── Step 1: Run uncontrolled baseline ─────────────────────────────────────────
print("\n[1/4] Running uncontrolled baseline...")
times_u, alt_u, _ = run_simulation(is_controlled=False)
apogee_u = alt_u.max()
print(f"      Uncontrolled apogee: {apogee_u:.2f} m")

# ── Step 2: Run hand-tuned P-controller ──────────────────────────────────────
print("\n[2/4] Running P-controller (Kp=5, hand-tuned)...")
times_p, alt_p, delta_p = run_simulation(is_controlled=True)
apogee_p = alt_p.max()
error_p  = abs(apogee_p - TARGET_APOGEE)
print(f"      P-controller apogee: {apogee_p:.2f} m  (error: {error_p:.2f} m)")

# ── Step 3: Find optimal PID gains using Optuna ───────────────────────────────
print("\n[3/4] Finding optimal PID gains with Bayesian optimisation...")
print("      Running 100 Optuna trials — please wait ~2 minutes...")
best_params, best_error = find_optimal_pid(
    H_INIT, V_INIT, TARGET_APOGEE,
    n_trials=100,
    verbose=False
)
Kp = best_params['Kp']
Ki = best_params['Ki']
Kd = best_params['Kd']
print(f"      Optimal gains: Kp={Kp:.6f}  Ki={Ki:.6f}  Kd={Kd:.6f}")
print(f"      Optuna error:  {best_error:.6f} m")

# ── Step 4: Run PID simulation with optimal gains ─────────────────────────────
print("\n[4/4] Running PID simulation with optimal gains...")
times_pid, alt_pid, delta_pid, apogee_pid, error_pid = run_pid_simulation(
    Kp, Ki, Kd, H_INIT, V_INIT, TARGET_APOGEE
)
print(f"      PID apogee: {apogee_pid:.4f} m  (error: {error_pid:.6f} m)")

# ── Results table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Controller':<30} {'Apogee':>12} {'Error':>12} {'Error %':>10}")
print("-" * 60)
print(f"{'Uncontrolled':<30} {apogee_u:>12.2f}m {'N/A':>12} {'N/A':>10}")
print(f"{'P-Controller (Kp=5)':<30} {apogee_p:>12.2f}m {error_p:>12.4f}m {error_p/TARGET_APOGEE*100:>10.4f}%")
print(f"{'PID (Optuna-tuned)':<30} {apogee_pid:>12.4f}m {error_pid:>12.6f}m {error_pid/TARGET_APOGEE*100:>10.6f}%")
print(f"\nImprovement over P-controller: {error_p/max(error_pid,1e-10):.0f}x better")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'Smart Simulation — H={H_INIT}m  V={V_INIT:.0f}m/s  Target={TARGET_APOGEE}m',
    fontsize=13, fontweight='bold'
)

# Altitude vs time
ax1 = axes[0]
ax1.plot(times_u,   alt_u,
         color='gray',    linewidth=1.5, linestyle='--', label='Uncontrolled')
ax1.plot(times_p,   alt_p,
         color='#E07B39', linewidth=2,   label=f'P-Controller (err={error_p:.2f}m)')
ax1.plot(times_pid, alt_pid,
         color='#2E75B6', linewidth=2,   label=f'PID Optuna (err={error_pid:.4f}m)')
ax1.axhline(TARGET_APOGEE, color='green', linewidth=1.5,
            linestyle='--', label=f'Target: {TARGET_APOGEE}m')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_title('Altitude vs Time')
ax1.legend()
ax1.grid(True, alpha=0.4)

# Airbrake deflection
ax2 = axes[1]
ax2.plot(times_p[:len(delta_p)],
         delta_p,   color='#E07B39', linewidth=2, label='P-Controller')
ax2.plot(times_pid[:len(delta_pid)],
         delta_pid, color='#2E75B6', linewidth=2, label='PID Optuna')
ax2.axhline(50, color='red', linewidth=1, linestyle='--', label='Max 50°')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Airbrake Deflection (°)')
ax2.set_title('Airbrake Deflection vs Time')
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot11_smart_simulation.png', dpi=150)
plt.close()
print(f"\nPlot saved to plots/plot11_smart_simulation.png")
print(f"\nSmart simulation complete.")
print(f"Change H_INIT, V_INIT, TARGET_APOGEE in constants.py and run again.")