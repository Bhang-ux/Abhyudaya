import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import numpy as np
from Abhyudaya.pid_controller import run_pid_simulation
from Abhyudaya.constants import H_INIT, V_INIT, TARGET_APOGEE

# What: suppress Optuna's per-trial verbose output
# Why: we print our own cleaner progress summary
optuna.logging.set_verbosity(optuna.logging.WARNING)


def find_optimal_pid(h_init, v_init, target, n_trials=50, verbose=True):
    """
    Uses Bayesian optimisation to find optimal PID gains for given conditions.

    What: runs n_trials different (Kp, Ki, Kd) combinations using Optuna
          and returns the combination that minimises apogee error
    Why: manual tuning is imprecise and does not generalise across conditions.
         Bayesian optimisation finds the true optimum efficiently.

    Args:
        h_init:   starting altitude (m)
        v_init:   starting velocity (m/s)
        target:   target apogee (m)
        n_trials: number of (Kp, Ki, Kd) combinations to try
        verbose:  print progress

    Returns:
        best_params: dict with optimal Kp, Ki, Kd
        best_error:  minimum apogee error achieved (m)
    """

    def objective(trial):
        # What: Optuna suggests values for each gain within defined ranges
        # Why: ranges are set to cover physically meaningful values
        #      log=True means Optuna searches log scale — better for
        #      parameters that span multiple orders of magnitude
        #      e.g. Ki might be optimal at 0.0001 or 0.1 — log scale
        #      explores both regions equally rather than favouring large values

        Kp = trial.suggest_float('Kp', 0.001, 1.0,   log=True)
        Ki = trial.suggest_float('Ki', 1e-6,  0.1,   log=True)
        Kd = trial.suggest_float('Kd', 1e-6,  1.0,   log=True)

        # What: run complete PID simulation with suggested gains
        # Why: this is the expensive part — Optuna calls this n_trials times
        try:
            _, _, _, _, error = run_pid_simulation(
                Kp, Ki, Kd, h_init, v_init, target
            )
            return error

        except Exception:
            # What: return large error if simulation fails
            # Why: some gain combinations cause numerical instability
            #      returning large error tells Optuna to avoid this region
            return 99999.0

    # What: create Optuna study
    # direction='minimize': we want to minimise apogee error
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_error  = study.best_value

    if verbose:
        print(f"  Optimal Kp={best_params['Kp']:.6f} "
              f"Ki={best_params['Ki']:.6f} "
              f"Kd={best_params['Kd']:.6f} "
              f"→ Error: {best_error:.4f}m")

    return best_params, best_error


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Abhyudaya.run_simulation import run_simulation

    print("Finding optimal PID gains for current constants.py conditions...")
    print(f"H_INIT={H_INIT}m | V_INIT={V_INIT:.1f}m/s | TARGET={TARGET_APOGEE}m")
    print(f"Running 100 Optuna trials...")
    print("-" * 60)

    # What: run optimisation with more trials for better accuracy
    # Why: 100 trials gives more thorough search than the 50 used in dataset generation
    best_params, best_error = find_optimal_pid(
        H_INIT, V_INIT, TARGET_APOGEE,
        n_trials=100,
        verbose=True
    )

    Kp = best_params['Kp']
    Ki = best_params['Ki']
    Kd = best_params['Kd']

    # What: run final simulation with optimal gains
    # Why: get the full trajectory for plotting and comparison
    times_pid, alt_pid, delta_pid, apogee_pid, error_pid = run_pid_simulation(
        Kp, Ki, Kd, H_INIT, V_INIT, TARGET_APOGEE
    )

    # What: run P-controller for comparison
    times_p, alt_p, delta_p = run_simulation(is_controlled=True)
    apogee_p = alt_p.max()
    error_p  = abs(apogee_p - TARGET_APOGEE)

    # What: run uncontrolled for baseline
    times_u, alt_u, _ = run_simulation(is_controlled=False)
    apogee_u = alt_u.max()

    print(f"\n── Results Comparison ───────────────────────────────")
    print(f"{'Controller':<25} {'Apogee':>10} {'Error':>10} {'Error %':>10}")
    print("-" * 60)
    print(f"{'Uncontrolled':<25} {apogee_u:>10.2f}m {'N/A':>10} {'N/A':>10}")
    print(f"{'P-Controller':<25} {apogee_p:>10.2f}m {error_p:>10.4f}m {error_p/TARGET_APOGEE*100:>10.4f}%")
    print(f"{'PID (Optuna-tuned)':<25} {apogee_pid:>10.2f}m {error_pid:>10.4f}m {error_pid/TARGET_APOGEE*100:>10.4f}%")

    print(f"\n── Optimal PID Gains ────────────────────────────────")
    print(f"Kp = {Kp:.6f}")
    print(f"Ki = {Ki:.6f}")
    print(f"Kd = {Kd:.6f}")

    # What: comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PID vs P-Controller — Optuna Tuned', fontsize=14, fontweight='bold')

    # Plot 1: Altitude vs time
    ax1 = axes[0]
    ax1.plot(times_u,   alt_u,   color='gray',    linewidth=1.5, linestyle='--', label='Uncontrolled')
    ax1.plot(times_p,   alt_p,   color='#E07B39', linewidth=2,   label=f'P-Controller (error={error_p:.2f}m)')
    ax1.plot(times_pid, alt_pid, color='#2E75B6', linewidth=2,   label=f'PID Optuna (error={error_pid:.4f}m)')
    ax1.axhline(TARGET_APOGEE, color='green', linewidth=1.5, linestyle='--', label=f'Target: {TARGET_APOGEE}m')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Altitude vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # Plot 2: Airbrake deflection comparison
    ax2 = axes[1]
    ax2.plot(times_p[:len(delta_p)],   delta_p,   color='#E07B39', linewidth=2, label='P-Controller')
    ax2.plot(times_pid[:len(delta_pid)], delta_pid, color='#2E75B6', linewidth=2, label='PID Optuna')
    ax2.axhline(50, color='red', linewidth=1, linestyle='--', label='Max 50°')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Airbrake Deflection (°)')
    ax2.set_title('Airbrake Deflection vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/plot9_pid_comparison.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to plots/plot9_pid_comparison.png")
    print(f"\nStep 2 complete. Run generate_pid_dataset.py next.")