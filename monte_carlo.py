import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from Abhyudaya.constants import (
    H_INIT, V_INIT, TARGET_APOGEE,
    ROCKET_MASS, GRAVITY, ROCKET_AREA, vs
)
from Abhyudaya.optimise_pid import find_optimal_pid

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DEFINE UNCERTAINTY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# What: realistic tolerances for each uncertain parameter
# Why: based on real aerospace engineering standards
#      mass: ±3% manufacturing tolerance
#      velocity: ±2% motor burn variability
#      Cx: ±5% CFD/wind tunnel measurement uncertainty

N_TRIALS     = 1000
MASS_MEAN    = ROCKET_MASS
MASS_STD     = ROCKET_MASS * 0.03      # 3% std dev

V_MEAN       = V_INIT
V_STD        = V_INIT * 0.02           # 2% std dev

CX_NOISE_STD = 0.05                    # 5% multiplicative noise on Cx

np.random.seed(42)                     # reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DEFINE PERTURBED PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def atmosphere(h):
    """ISA air density model."""
    T   = 15.04 - 0.00649 * h
    k   = 1.2050
    rho = k * ((T + 273) / 288.08) ** 4.256
    return max(rho, 0.001)

def perturbed_acceleration(h, v, delta, mass, cx_multiplier):
    """
    What: computes acceleration with perturbed mass and Cx
    Why: Monte Carlo needs to vary these parameters each trial
         Cannot use the original acceleration.py directly because
         it uses fixed constants — we need to inject noise

    Args:
        h:             altitude (m)
        v:             velocity (m/s)
        delta:         airbrake deflection (degrees)
        mass:          perturbed rocket mass (kg)
        cx_multiplier: noise factor on Cx (e.g. 1.05 = 5% higher drag)
    """
    from Abhyudaya.bilinear_interpolate import bilinear_interpolate

    rho  = atmosphere(h)
    mach = abs(v) / vs

    # What: get base Cx from lookup table then apply noise multiplier
    # Why: Cx uncertainty is multiplicative — scales with base value
    cx_base = bilinear_interpolate(mach, delta)
    cx      = abs(cx_base) * cx_multiplier

    f_drag = 0.5 * rho * v**2 * cx * ROCKET_AREA
    a      = -f_drag / mass - GRAVITY
    return a

def perturbed_predict_apogee(h, v, delta, mass, cx_multiplier, dt=0.1):
    """
    What: forward Euler integrator with perturbed parameters
    Why: needed for PID controller's predict_apogee call during flight
         The PID controller uses predicted apogee to compute error —
         in Monte Carlo, this prediction also uses the perturbed model
    """
    while v > 0:
        a  = perturbed_acceleration(h, v, delta, mass, cx_multiplier)
        v += a * dt
        h += v * dt
    return h

def run_perturbed_pid(Kp, Ki, Kd, h_init, v_init, mass, cx_multiplier, dt=0.1):
    """
    What: runs full PID flight simulation with perturbed parameters
    Why: this is the core of each Monte Carlo trial —
         same optimal PID gains but different physical conditions

    Returns:
        apogee: maximum altitude reached (m)
    """
    h          = h_init
    v          = v_init
    integral   = 0.0
    prev_error = 0.0
    max_h      = h

    while v > 0:
        # What: predict apogee using perturbed model
        # Why: controller uses its (imperfect) model to predict trajectory
        #      This is realistic — the controller does not know the true mass
        predicted = perturbed_predict_apogee(h, v, 0, mass, cx_multiplier)
        error     = predicted - TARGET_APOGEE

        # PID control law
        integral   += error * dt
        integral    = np.clip(integral, -10000, 10000)
        derivative  = (error - prev_error) / dt
        prev_error  = error

        delta = Kp * error + Ki * integral + Kd * derivative
        delta = float(np.clip(delta, 0, 50))

        # Physics step with perturbed parameters
        a  = perturbed_acceleration(h, v, delta, mass, cx_multiplier)
        v += a * dt
        h += v * dt
        max_h = max(max_h, h)

    return max_h

def run_perturbed_uncontrolled(h_init, v_init, mass, cx_multiplier, dt=0.1):
    """
    What: uncontrolled flight with perturbed parameters
    Why: baseline comparison — shows natural spread without control
    """
    h, v = h_init, v_init
    max_h = h
    while v > 0:
        a  = perturbed_acceleration(h, v, 0, mass, cx_multiplier)
        v += a * dt
        h += v * dt
        max_h = max(max_h, h)
    return max_h

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FIND OPTIMAL PID GAINS FOR NOMINAL CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
# What: get the best PID gains for nominal (perfect) conditions
# Why: in real deployment you tune the controller once using your best
#      knowledge of the system. Then fly in real (uncertain) conditions.
#      Monte Carlo tests how well those nominal-tuned gains hold up.

print("="*60)
print("MONTE CARLO ROBUSTNESS ANALYSIS")
print("="*60)
print(f"H_INIT={H_INIT}m | V_INIT={V_INIT:.1f}m/s | TARGET={TARGET_APOGEE}m")
print(f"Trials: {N_TRIALS}")
print(f"Mass uncertainty:     ±{MASS_STD/MASS_MEAN*100:.0f}% (σ={MASS_STD:.1f}kg)")
print(f"Velocity uncertainty: ±{V_STD/V_MEAN*100:.0f}% (σ={V_STD:.1f}m/s)")
print(f"Cx uncertainty:       ±{CX_NOISE_STD*100:.0f}% (multiplicative)")
print("-"*60)

print("\nFinding optimal PID gains for nominal conditions...")
best_params, nominal_error = find_optimal_pid(
    H_INIT, V_INIT, TARGET_APOGEE,
    n_trials=100,
    verbose=False
)
Kp = best_params['Kp']
Ki = best_params['Ki']
Kd = best_params['Kd']
print(f"Nominal gains: Kp={Kp:.6f} Ki={Ki:.6f} Kd={Kd:.6f}")
print(f"Nominal error: {nominal_error:.6f}m")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RUN MONTE CARLO TRIALS
# ─────────────────────────────────────────────────────────────────────────────
# What: run N_TRIALS simulations each with randomly sampled parameters
# Why: builds up a distribution of possible apogee outcomes
#      capturing real-world uncertainty in the system

print(f"\nRunning {N_TRIALS} Monte Carlo trials...")

pid_apogees          = []
uncontrolled_apogees = []

for trial in range(N_TRIALS):

    # What: sample uncertain parameters from their distributions
    # Why: each trial represents one possible real flight
    #      np.random.normal: samples from Gaussian distribution
    mass          = np.random.normal(MASS_MEAN, MASS_STD)
    v_init_pert   = np.random.normal(V_MEAN,    V_STD)
    cx_multiplier = np.random.normal(1.0,       CX_NOISE_STD)

    # What: clip to physically valid ranges
    # Why: prevent negative mass or wildly unrealistic values
    mass          = np.clip(mass,          MASS_MEAN*0.85, MASS_MEAN*1.15)
    v_init_pert   = np.clip(v_init_pert,   V_MEAN*0.85,    V_MEAN*1.15)
    cx_multiplier = np.clip(cx_multiplier, 0.7,            1.3)

    # Run PID controlled flight
    pid_apogee = run_perturbed_pid(
        Kp, Ki, Kd,
        H_INIT, v_init_pert,
        mass, cx_multiplier
    )
    pid_apogees.append(pid_apogee)

    # Run uncontrolled flight
    unc_apogee = run_perturbed_uncontrolled(
        H_INIT, v_init_pert,
        mass, cx_multiplier
    )
    uncontrolled_apogees.append(unc_apogee)

    if (trial + 1) % 100 == 0:
        print(f"  Trial {trial+1:4d}/{N_TRIALS} complete")

pid_apogees          = np.array(pid_apogees)
uncontrolled_apogees = np.array(uncontrolled_apogees)
pid_errors           = np.abs(pid_apogees - TARGET_APOGEE)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: COMPUTE RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
# What: compute statistical measures of performance distribution
# Why: single-number metrics compress the full distribution into
#      actionable insights about controller robustness

print(f"\n{'='*60}")
print(f"MONTE CARLO RESULTS — PID CONTROLLER ROBUSTNESS")
print(f"{'='*60}")

# Basic statistics
mean_error   = np.mean(pid_errors)
median_error = np.median(pid_errors)
std_error    = np.std(pid_errors)

# Percentile metrics
p5_apogee    = np.percentile(pid_apogees, 5)
p95_apogee   = np.percentile(pid_apogees, 95)
p5_error     = np.percentile(pid_errors,  5)
p95_error    = np.percentile(pid_errors,  95)

# What: probability of missing target by more than threshold
# Why: this is the operational risk metric — probability of failure
pct_within_50   = np.mean(pid_errors < 50)   * 100
pct_within_100  = np.mean(pid_errors < 100)  * 100
pct_within_200  = np.mean(pid_errors < 200)  * 100
pct_over_500    = np.mean(pid_errors > 500)  * 100

# What: uncontrolled spread for comparison
unc_std  = np.std(uncontrolled_apogees)
unc_p5   = np.percentile(uncontrolled_apogees, 5)
unc_p95  = np.percentile(uncontrolled_apogees, 95)

print(f"\n── PID Controller Performance ───────────────────────────")
print(f"Mean apogee error:         {mean_error:.2f} m")
print(f"Median apogee error:       {median_error:.2f} m")
print(f"Std dev of error:          {std_error:.2f} m")
print(f"P5  apogee:                {p5_apogee:.2f} m")
print(f"P95 apogee:                {p95_apogee:.2f} m")
print(f"P5-P95 range:              {p95_apogee - p5_apogee:.2f} m")
print(f"\n── Success Rate ─────────────────────────────────────────")
print(f"Within 50m of target:      {pct_within_50:.1f}%")
print(f"Within 100m of target:     {pct_within_100:.1f}%")
print(f"Within 200m of target:     {pct_within_200:.1f}%")
print(f"Miss by > 500m:            {pct_over_500:.1f}%")
print(f"\n── Uncontrolled Baseline ────────────────────────────────")
print(f"Uncontrolled spread (std): {unc_std:.2f} m")
print(f"Uncontrolled P5-P95:       {unc_p95 - unc_p5:.2f} m")
print(f"\n── Finance-Parallel Risk Metrics ────────────────────────")
print(f"VaR equivalent (95%):      {p95_error:.2f} m")
print(f"  (95% of flights land within {p95_error:.2f}m of target)")
print(f"CVaR equivalent (worst 5%):{np.mean(pid_errors[pid_errors > np.percentile(pid_errors,95)]):.2f} m")
print(f"  (average error in worst 5% of flights)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Monte Carlo Robustness Analysis — {N_TRIALS} Trials\n'
    f'H={H_INIT}m | V={V_INIT:.0f}m/s | Target={TARGET_APOGEE}m',
    fontsize=13, fontweight='bold'
)

# Plot 1: Apogee distribution — PID vs Uncontrolled
ax1 = axes[0, 0]
ax1.hist(uncontrolled_apogees, bins=50, alpha=0.5,
         color='gray', label='Uncontrolled', density=True)
ax1.hist(pid_apogees, bins=50, alpha=0.7,
         color='#2E75B6', label='PID Controller', density=True)
ax1.axvline(TARGET_APOGEE, color='green', linewidth=2,
            linestyle='--', label=f'Target: {TARGET_APOGEE}m')
ax1.axvline(p5_apogee,  color='#2E75B6', linewidth=1.5,
            linestyle=':', label=f'P5: {p5_apogee:.0f}m')
ax1.axvline(p95_apogee, color='#2E75B6', linewidth=1.5,
            linestyle=':', label=f'P95: {p95_apogee:.0f}m')
ax1.set_xlabel('Final Apogee (m)')
ax1.set_ylabel('Density')
ax1.set_title('Apogee Distribution — PID vs Uncontrolled')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4)

# Plot 2: Error distribution with risk bands
ax2 = axes[0, 1]
ax2.hist(pid_errors, bins=50, color='#2E75B6', alpha=0.8, edgecolor='white')
ax2.axvline(mean_error,   color='red',    linewidth=2,
            linestyle='-',  label=f'Mean: {mean_error:.1f}m')
ax2.axvline(p95_error,    color='orange', linewidth=2,
            linestyle='--', label=f'P95 (VaR): {p95_error:.1f}m')
ax2.axvline(50,  color='green', linewidth=1.5,
            linestyle=':', alpha=0.7, label='50m threshold')
ax2.axvline(200, color='red',   linewidth=1.5,
            linestyle=':', alpha=0.7, label='200m threshold')
ax2.set_xlabel('Apogee Error (m)')
ax2.set_ylabel('Count')
ax2.set_title(f'Error Distribution ({pct_within_100:.1f}% within 100m)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4)

# Plot 3: Cumulative distribution of errors
# What: shows what % of flights land within X metres of target
# Why: directly answers "how reliable is this controller"
ax3 = axes[1, 0]
sorted_errors = np.sort(pid_errors)
cumulative    = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
ax3.plot(sorted_errors, cumulative, color='#2E75B6', linewidth=2)
ax3.axhline(90, color='orange', linewidth=1.5, linestyle='--',
            label='90th percentile')
ax3.axhline(95, color='red',    linewidth=1.5, linestyle='--',
            label='95th percentile')
ax3.axvline(50,  color='green', linewidth=1,   linestyle=':',
            label='50m')
ax3.axvline(100, color='gray',  linewidth=1,   linestyle=':',
            label='100m')
ax3.set_xlabel('Apogee Error (m)')
ax3.set_ylabel('Cumulative % of Flights')
ax3.set_title('Cumulative Error Distribution')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.4)
ax3.set_xlim(0, max(sorted_errors) * 1.05)

# Plot 4: Scatter — parameter sensitivity
# What: shows which uncertain parameter most affects apogee error
# Why: identifies which physical parameter is most critical to control
#      This is a sensitivity analysis — tells you where to focus
#      engineering effort to reduce real-world variability
ax4 = axes[1, 1]

# Regenerate samples with same seed for scatter plot
np.random.seed(42)
masses   = np.random.normal(MASS_MEAN, MASS_STD,    N_TRIALS)
v_inits  = np.random.normal(V_MEAN,    V_STD,       N_TRIALS)
cx_mults = np.random.normal(1.0,       CX_NOISE_STD, N_TRIALS)
masses   = np.clip(masses,   MASS_MEAN*0.85, MASS_MEAN*1.15)
v_inits  = np.clip(v_inits,  V_MEAN*0.85,    V_MEAN*1.15)
cx_mults = np.clip(cx_mults, 0.7,            1.3)

# Compute correlation of each parameter with error
v_corr  = np.corrcoef(v_inits,  pid_errors)[0,1]
m_corr  = np.corrcoef(masses,   pid_errors)[0,1]
cx_corr = np.corrcoef(cx_mults, pid_errors)[0,1]

sc = ax4.scatter(v_inits, pid_errors, c=masses,
                 cmap='RdYlGn_r', alpha=0.4, s=8)
plt.colorbar(sc, ax=ax4, label='Mass (kg)')
ax4.set_xlabel('Initial Velocity (m/s)')
ax4.set_ylabel('Apogee Error (m)')
ax4.set_title(f'Error Sensitivity\n'
              f'V corr={v_corr:.2f} | M corr={m_corr:.2f} | Cx corr={cx_corr:.2f}')
ax4.grid(True, alpha=0.4)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot12_monte_carlo.png', dpi=150)
plt.close()
print(f"\nPlot saved to plots/plot12_monte_carlo.png")
print(f"\nMonte Carlo analysis complete.")