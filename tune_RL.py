import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What: Optuna — Bayesian hyperparameter optimisation library
# Why: instead of manually trying alpha=0.1, then 0.2, then 0.3...
#      Optuna builds a probabilistic model of which hyperparameters
#      lead to good results and intelligently picks the next combination to try
# Real world: identical technique used in ML model tuning, drug discovery,
#      financial model calibration — anywhere you need to optimise
#      an expensive black-box function
import optuna
import numpy as np
from Abhyudaya.acceleration import acceleration
from Abhyudaya.constants import TARGET_APOGEE, H_INIT, V_INIT
from Abhyudaya.nn_apogee_predictor import nn_predict_apogee

# What: suppress Optuna's verbose per-trial output
# Why: we print our own cleaner summary
optuna.logging.set_verbosity(optuna.logging.WARNING)

DT      = 0.1
ACTIONS = [0, 10, 20, 30, 40, 50]
N_ACTIONS = len(ACTIONS)

def get_state(h, v, n_h_bins, n_v_bins, h_min, h_max, v_min, v_max):
    h_bins = np.linspace(h_min, h_max, n_h_bins + 1)
    v_bins = np.linspace(v_min, v_max, n_v_bins + 1)
    h_bin  = np.clip(np.digitize(h, h_bins) - 1, 0, n_h_bins - 1)
    v_bin  = np.clip(np.digitize(v, v_bins) - 1, 0, n_v_bins - 1)
    return (h_bin, v_bin)

def get_reward(h, v, apogee, done):
    if not done:
        predicted  = nn_predict_apogee(h, v, 0)
        pred_error = abs(predicted - TARGET_APOGEE)
        return -1 + (-pred_error / 5000)
    else:
        error = abs(apogee - TARGET_APOGEE)
        if error < 50:    return 1000
        elif error < 200: return 500
        else:             return -error / 10

def step(h, v, action_idx):
    delta = ACTIONS[action_idx]
    accel = acceleration(h, v, delta)
    v_new = v + accel * DT
    h_new = h + v_new * DT
    if v_new <= 0:
        return h_new, v_new, get_reward(h_new, v_new, h_new, True),  True,  h_new
    else:
        return h_new, v_new, get_reward(h_new, v_new, None,  False), False, None

def run_rl(alpha, gamma, epsilon_decay, episodes, n_h_bins, n_v_bins):
    """
    What: runs one complete RL training session with given hyperparameters
    Why: this is the objective function Optuna calls repeatedly to evaluate
         different hyperparameter combinations
    Returns: final apogee error — lower is better
    """
    H_MIN = H_INIT * 0.5
    H_MAX = H_INIT * 8
    V_MIN = 0
    V_MAX = V_INIT * 1.2

    Q       = np.zeros((n_h_bins, n_v_bins, N_ACTIONS))
    epsilon = 1.0

    for episode in range(episodes):
        h, v, done = H_INIT, V_INIT, False

        while not done:
            state = get_state(h, v, n_h_bins, n_v_bins, H_MIN, H_MAX, V_MIN, V_MAX)

            if np.random.random() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                action_idx = np.argmax(Q[state[0], state[1], :])

            h_new, v_new, reward, done, apogee = step(h, v, action_idx)
            new_state = get_state(h_new, v_new, n_h_bins, n_v_bins, H_MIN, H_MAX, V_MIN, V_MAX)

            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[new_state[0], new_state[1], :])

            Q[state[0], state[1], action_idx] += alpha * (
                target - Q[state[0], state[1], action_idx]
            )
            h, v = h_new, v_new

        epsilon = max(0.05, epsilon * epsilon_decay)

    # What: evaluate final policy with no exploration
    # Why: this is the true performance — pure exploitation of learned Q-table
    h, v, done = H_INIT, V_INIT, False
    final_apogee = H_INIT

    while not done:
        state      = get_state(h, v, n_h_bins, n_v_bins, H_MIN, H_MAX, V_MIN, V_MAX)
        action_idx = np.argmax(Q[state[0], state[1], :])
        h, v, _, done, apogee = step(h, v, action_idx)
        if apogee:
            final_apogee = apogee

    return abs(final_apogee - TARGET_APOGEE), Q, n_h_bins, n_v_bins, H_MIN, H_MAX, V_MIN, V_MAX

# ─────────────────────────────────────────────────────────────────────────────
# THE OBJECTIVE FUNCTION — what Optuna minimises
# ─────────────────────────────────────────────────────────────────────────────
# What: Optuna calls this function with different hyperparameter values
# Why: it returns the error — Optuna tries to minimise this number
#
# trial.suggest_float / suggest_int:
#      tells Optuna the search range for each hyperparameter
#      Optuna uses a probabilistic model (Tree Parzen Estimator) to
#      decide which values to try next — smarter than random search

def objective(trial):
    # What: define search space for each hyperparameter
    # Why: these ranges cover the reasonable spectrum for Q-learning
    alpha         = trial.suggest_float('alpha',         0.05, 0.5,  log=True)
    gamma         = trial.suggest_float('gamma',         0.90, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.990, 0.9999)
    episodes      = trial.suggest_int  ('episodes',      2000, 8000)
    n_h_bins      = trial.suggest_int  ('n_h_bins',      15,   40)
    n_v_bins      = trial.suggest_int  ('n_v_bins',      15,   40)

    error, _, _, _, _, _, _, _ = run_rl(
        alpha, gamma, epsilon_decay, episodes, n_h_bins, n_v_bins
    )
    return error

# ─────────────────────────────────────────────────────────────────────────────
# RUN THE OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
# What: creates an Optuna study and runs N_TRIALS evaluations
# Why: each trial tests one hyperparameter combination.
#      Optuna learns from each trial to pick better combinations next time.
#      This is Bayesian optimisation — smarter than grid search or random search.
#
# direction='minimize': we want to minimise apogee error
# n_trials=50: run 50 different hyperparameter combinations
#      First ~10 trials are random exploration
#      Remaining 40 trials exploit what was learned — focus on promising regions

N_TRIALS = 50

print(f"Running Bayesian hyperparameter optimisation...")
print(f"Trials: {N_TRIALS} | Target: minimise |apogee - {TARGET_APOGEE}m|")
print(f"Initial conditions: H_INIT={H_INIT}m, V_INIT={V_INIT:.1f}m/s")
print("-" * 60)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS,
               callbacks=[lambda study, trial: print(
                   f"Trial {trial.number+1:3d}/{N_TRIALS} | "
                   f"Error: {trial.value:8.2f}m | "
                   f"Best so far: {study.best_value:8.2f}m"
               )])

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
best = study.best_params
print(f"\n{'='*60}")
print(f"OPTIMAL HYPERPARAMETERS FOUND")
print(f"{'='*60}")
print(f"Alpha (learning rate):    {best['alpha']:.4f}")
print(f"Gamma (discount factor):  {best['gamma']:.4f}")
print(f"Epsilon decay:            {best['epsilon_decay']:.4f}")
print(f"Episodes:                 {best['episodes']}")
print(f"H bins:                   {best['n_h_bins']}")
print(f"V bins:                   {best['n_v_bins']}")
print(f"\nBest apogee error:        {study.best_value:.2f} m")
print(f"Target apogee:            {TARGET_APOGEE} m")

# What: run one final training with the best found hyperparameters
# Why: get the actual trained Q-table to evaluate and compare with P-controller
print(f"\nRunning final training with optimal hyperparameters...")
final_error, Q_best, n_h, n_v, H_MIN, H_MAX, V_MIN, V_MAX = run_rl(
    best['alpha'], best['gamma'], best['epsilon_decay'],
    best['episodes'], best['n_h_bins'], best['n_v_bins']
)

print(f"Final RL error with optimal params: {final_error:.2f} m")

# Compare with P-controller
from Abhyudaya.run_simulation import run_simulation
_, alt_p, _ = run_simulation(is_controlled=True)
p_error = abs(alt_p.max() - TARGET_APOGEE)

print(f"P-Controller error:                 {p_error:.2f} m")
print(f"\nUpdate your RL_controller.py with these values:")
print(f"  ALPHA         = {best['alpha']:.4f}")
print(f"  GAMMA         = {best['gamma']:.4f}")
print(f"  EPSILON_DECAY = {best['epsilon_decay']:.4f}")
print(f"  EPISODES      = {best['episodes']}")
print(f"  N_H_BINS      = {best['n_h_bins']}")
print(f"  N_V_BINS      = {best['n_v_bins']}")

# What: save optimisation history plot
# Why: shows how error improved as Optuna tried more combinations
import matplotlib.pyplot as plt

errors = [t.value for t in study.trials]
plt.figure(figsize=(10, 5))
plt.plot(errors, alpha=0.4, color='#2E75B6', linewidth=0.8, label='Trial error')
plt.plot(np.minimum.accumulate(errors), color='#C00000', linewidth=2, label='Best so far')
plt.axhline(p_error, color='green', linewidth=2, linestyle='--',
            label=f'P-Controller error ({p_error:.1f}m)')
plt.xlabel('Trial Number')
plt.ylabel('Apogee Error (m)')
plt.title('Bayesian Optimisation — RL Hyperparameter Search')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot9_optuna_search.png', dpi=150)
plt.close()
print(f"\nOptimisation plot saved to plots/plot9_optuna_search.png")
print(f"\nPhase 2D complete.")