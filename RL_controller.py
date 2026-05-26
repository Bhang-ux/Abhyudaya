from Abhyudaya.nn_apogee_predictor import nn_predict_apogee

def get_reward(h, v, delta, apogee, done):
    if not done:
        predicted  = nn_predict_apogee(h, v, 0)  # 200x faster
        pred_error = abs(predicted - TARGET_APOGEE)
        shaping    = -pred_error / 5000
        return -1 + shaping
    else:
        error = abs(apogee - TARGET_APOGEE)
        if error < 50:    return 1000
        elif error < 200: return 500
        else:             return -error / 10
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from Abhyudaya.acceleration import acceleration
from Abhyudaya.constants import TARGET_APOGEE, vs, H_INIT, V_INIT

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DEFINE THE ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
# What: defines all the parameters of the rocket flight environment
# Why: the agent needs to know the rules of the world it operates in —
#      what states exist, what actions are available, how time progresses

# Initial conditions — same as your main simulation
 
DT      = 0.1        # timestep (seconds) — 10 decisions per second

# What: discrete action space — 6 possible airbrake angles
# Why: discrete actions make Q-learning simple. Agent picks from a menu
#      rather than searching continuous space.
ACTIONS = [0, 10, 20, 30, 40, 50]   # degrees
N_ACTIONS = len(ACTIONS)             # 6

# What: discretise the state space into bins
# Why: Q-table needs discrete states. We bin continuous (h, v) into a grid.
#      20×20 = 400 possible states × 6 actions = 2400 Q-table entries
N_H_BINS = 20     # altitude bins
N_V_BINS = 20     # velocity bins

H_MIN, H_MAX = 1000, 12000     # altitude range (m)
V_MIN, V_MAX = 0, 500       # velocity range (m/s)

def get_state(h, v):
    """
    Convert continuous (h, v) to discrete (h_bin, v_bin).
    
    What: maps a real altitude and velocity to a row/column in the Q-table
    Why: Q-table needs discrete indices. np.digitize finds which bin
         a continuous value falls into.
    np.clip: ensures bin index stays within valid range even at boundaries
    """
    # What: np.linspace creates evenly spaced bin edges
    # np.digitize finds which bin the value falls into
    h_bins = np.linspace(H_MIN, H_MAX, N_H_BINS + 1)
    v_bins = np.linspace(V_MIN, V_MAX, N_V_BINS + 1)

    # What: clip to valid range — prevents index out of bounds
    h_bin = np.clip(np.digitize(h, h_bins) - 1, 0, N_H_BINS - 1)
    v_bin = np.clip(np.digitize(v, v_bins) - 1, 0, N_V_BINS - 1)

    return (h_bin, v_bin)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DEFINE THE REWARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# What: gives the agent a score after each action
# Why: this is how you communicate the goal to the agent.
#      The agent has no goal of its own — it just maximises total reward.
#      Every design decision here is intentional — see comments.

from Abhyudaya.predict_apogee import predict_apogee

def get_reward(h, v, delta, apogee, done):
    if not done:
        # What: reward based on how close PREDICTED apogee is to target
        # Why: gives agent feedback at every single timestep not just the end
        #      Without this agent only learns from terminal reward — very sparse
        #      With this agent gets signal throughout the entire flight
        predicted = predict_apogee(h, v, 0)
        pred_error = abs(predicted - TARGET_APOGEE)

        # What: normalise error to small number
        # Why: keeps shaping reward small relative to terminal reward
        #      shaping is a hint not the main goal
        shaping = -pred_error / 5000

        return -1 + shaping

    else:
        error = abs(apogee - TARGET_APOGEE)
        if error < 50:
            return 1000
        elif error < 200:
            return 500
        else:
            return -error / 10


def step(h, v, action_idx):
    delta = ACTIONS[action_idx]
    accel = acceleration(h, v, delta)
    v_new = v + accel * DT
    h_new = h + v_new * DT

    if v_new <= 0:
        done   = True
        apogee = h_new
        reward = get_reward(h_new, v_new, delta, apogee, done=True)
    else:
        done   = False
        apogee = None
        reward = get_reward(h_new, v_new, delta, apogee=None, done=False)

    return h_new, v_new, reward, done, apogee

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: INITIALISE THE Q-TABLE
# ─────────────────────────────────────────────────────────────────────────────
# What: creates the Q-table — the agent's memory
# Why: stores learned value of every (state, action) combination
#      Shape: (N_H_BINS, N_V_BINS, N_ACTIONS) = (20, 20, 6) = 2400 values
#
# Why initialise to 0?
#      Neutral starting point — agent has no prior knowledge.
#      Some methods initialise optimistically (large positive values) to
#      encourage exploration — called optimistic initialisation.
#      Zero is simpler and works fine here.
Q = np.zeros((N_H_BINS, N_V_BINS, N_ACTIONS))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# What: settings that control HOW the agent learns
# Why: these are not learned — you set them manually based on the problem

EPISODES    = 5000    # total number of flights to simulate
                      # Why 3000: enough for Q-table to converge on this
                      #           simple 400-state problem

ALPHA       = 0.2     # learning rate — how fast to update Q values
                      # Why 0.1: standard starting point. Too high (0.9)
                      #          causes unstable oscillating updates.
                      #          Too low (0.001) learns too slowly.

GAMMA       = 0.99    # discount factor — how much future rewards matter
                      # Why 0.99: we care a lot about the final apogee reward
                      #           even though it is 200+ steps away.
                      #           0.99^220 = 0.11 — terminal reward still
                      #           contributes 11% even 220 steps back.

EPSILON_START = 1.0   # starting exploration rate — 100% random
EPSILON_END   = 0.05  # minimum exploration rate — always 5% random
EPSILON_DECAY = 0.9995 # multiply epsilon by this each episode
                      # Why 0.999: decays from 1.0 to 0.05 over ~3000 episodes
                      # 1.0 × 0.999^3000 = 0.05 ✓

epsilon = EPSILON_START

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
# What: runs EPISODES complete rocket flights, updating Q-table each time
# Why: each flight gives the agent experience to learn from.
#      More episodes = more experience = better policy.

print("Training RL agent...")
print(f"Episodes: {EPISODES} | α={ALPHA} | γ={GAMMA} | ε: {EPSILON_START}→{EPSILON_END}")
print("-" * 60)

episode_rewards  = []    # track total reward per episode
episode_apogees  = []    # track final apogee per episode
episode_epsilons = []    # track epsilon decay

for episode in range(EPISODES):

    # What: reset environment — start fresh flight
    h = H_INIT
    v = V_INIT
    total_reward = 0
    done = False
    final_apogee = None

    while not done:

        # ── Choose action (epsilon-greedy) ────────────────────────────────
        # What: decide whether to explore or exploit
        # Why: balances learning new things vs using what we know

        state = get_state(h, v)

        if np.random.random() < epsilon:
            # What: EXPLORE — pick a completely random action
            # Why: discovers new (state, action) combinations the agent
            #      has not tried before. Prevents getting stuck.
            action_idx = np.random.randint(N_ACTIONS)
        else:
            # What: EXPLOIT — pick the action with highest Q value
            # Why: uses accumulated knowledge to make the best known decision
            # np.argmax: returns index of the largest value in the array
            action_idx = np.argmax(Q[state[0], state[1], :])

        # ── Take action, get result ───────────────────────────────────────
        h_new, v_new, reward, done, apogee = step(h, v, action_idx)
        new_state = get_state(h_new, v_new)
        total_reward += reward

        if done:
            final_apogee = apogee

        # ── Update Q-table (Bellman equation) ────────────────────────────
        # What: update Q value for the (state, action) pair we just tried
        # Why: moves Q value toward what actually happened
        #
        # Bellman equation:
        # Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]
        #
        # Breaking it down:
        # r + γ × max Q(s',a')  = target  (what actually happened)
        # Q(s,a)                = current estimate
        # target - Q(s,a)       = TD error (how wrong we were)
        # α × TD error          = how much to adjust
        #
        # If TD error > 0: reality was better than expected → increase Q
        # If TD error < 0: reality was worse than expected  → decrease Q
        # If TD error = 0: prediction was perfect → no change

        if done:
            # What: no future state — flight is over
            # Why: terminal state has no next Q value to bootstrap from
            #      target = just the immediate reward
            target = reward
        else:
            # What: best Q value available in next state
            # Why: this is the Bellman bootstrapping — use current estimate
            #      of future value to update current Q value
            target = reward + GAMMA * np.max(Q[new_state[0], new_state[1], :])

        # What: the actual Q-table update
        # Why: nudges Q(s,a) toward target by fraction α
        #      α=0.1 means take 10% step toward target each update
        Q[state[0], state[1], action_idx] += ALPHA * (
            target - Q[state[0], state[1], action_idx]
        )

        # What: move to next state
        h, v = h_new, v_new

    # ── End of episode ────────────────────────────────────────────────────
    episode_rewards.append(total_reward)
    episode_apogees.append(final_apogee if final_apogee else 0)
    episode_epsilons.append(epsilon)

    # What: decay epsilon after each episode
    # Why: as agent learns more, needs less random exploration
    #      Gradually shifts from exploring to exploiting
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # What: print progress every 300 episodes
    if (episode + 1) % 300 == 0:
        recent_apogees  = episode_apogees[-300:]
        recent_rewards  = episode_rewards[-300:]
        avg_apogee      = np.mean(recent_apogees)
        avg_reward      = np.mean(recent_rewards)
        apogee_error    = abs(avg_apogee - TARGET_APOGEE)
        print(f"Episode {episode+1:5d} | "
              f"Avg Apogee: {avg_apogee:7.1f}m | "
              f"Error: {apogee_error:6.1f}m | "
              f"Avg Reward: {avg_reward:8.1f} | "
              f"ε: {epsilon:.3f}")

print("-" * 60)
print("Training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EVALUATE THE LEARNED POLICY
# ─────────────────────────────────────────────────────────────────────────────
# What: run one clean flight using only the learned policy (no exploration)
# Why: shows what the agent actually learned — pure exploitation, ε=0

print("\nEvaluating learned policy (no exploration)...")

h, v  = H_INIT, V_INIT
done  = False
rl_altitudes  = [h]
rl_deflections = []
rl_times      = [0]
t = 0

while not done:
    state      = get_state(h, v)
    action_idx = np.argmax(Q[state[0], state[1], :])   # always best action
    delta      = ACTIONS[action_idx]

    h, v, reward, done, apogee = step(h, v, action_idx)
    rl_altitudes.append(h)
    rl_deflections.append(delta)
    rl_times.append(t)
    t += DT

final_apogee = max(rl_altitudes)
error        = abs(final_apogee - TARGET_APOGEE)
print(f"RL Agent Apogee:  {final_apogee:.2f} m")
print(f"Target Apogee:    {TARGET_APOGEE:.2f} m")
print(f"Absolute Error:   {error:.2f} m")
print(f"Percentage Error: {error/TARGET_APOGEE*100:.3f} %")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: COMPARE WITH P-CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
from Abhyudaya.run_simulation import run_simulation
import numpy as np

time_p, alt_p, delta_p = run_simulation(is_controlled=True)
p_apogee = alt_p.max()
p_error  = abs(p_apogee - TARGET_APOGEE)

print(f"\nP-Controller Apogee: {p_apogee:.2f} m")
print(f"P-Controller Error:  {p_error:.2f} m")
print(f"\nRL vs P-Controller:")
print(f"  RL Error:  {error:.2f} m")
print(f"  P Error:   {p_error:.2f} m")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reinforcement Learning Controller — Q-Learning', 
             fontsize=14, fontweight='bold')

# Plot 1: Learning curve — total reward per episode
# What: shows how total reward improved over training
# Why: upward trend = agent is learning. Flat = converged or stuck.
ax1 = axes[0, 0]
# Smooth with rolling average for readability
window = 100
smoothed_rewards = np.convolve(
    episode_rewards, 
    np.ones(window)/window, 
    mode='valid'
)
ax1.plot(episode_rewards, alpha=0.2, color='#2E75B6', linewidth=0.5)
ax1.plot(range(window-1, len(episode_rewards)), 
         smoothed_rewards, color='#2E75B6', linewidth=2, 
         label=f'{window}-episode moving average')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Learning Curve — Total Reward per Episode')
ax1.legend()
ax1.grid(True, alpha=0.4)

# Plot 2: Apogee per episode
# What: shows how close to target the agent gets over training
# Why: directly shows whether the agent is hitting the goal
ax2 = axes[0, 1]
smoothed_apogees = np.convolve(
    episode_apogees,
    np.ones(window)/window,
    mode='valid'
)
ax2.plot(episode_apogees, alpha=0.2, color='#E07B39', linewidth=0.5)
ax2.plot(range(window-1, len(episode_apogees)),
         smoothed_apogees, color='#E07B39', linewidth=2,
         label=f'{window}-episode moving average')
ax2.axhline(TARGET_APOGEE, color='green', linewidth=2, 
            linestyle='--', label=f'Target: {TARGET_APOGEE}m')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Final Apogee (m)')
ax2.set_title('Apogee Convergence over Training')
ax2.legend()
ax2.grid(True, alpha=0.4)

# Plot 3: Epsilon decay
# What: shows exploration rate decreasing over training
# Why: confirms agent shifts from exploring to exploiting as intended
ax3 = axes[1, 0]
ax3.plot(episode_epsilons, color='#C00000', linewidth=1.5)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Epsilon (ε)')
ax3.set_title('Exploration Rate Decay')
ax3.grid(True, alpha=0.4)

# Plot 4: RL vs P-controller altitude comparison
# What: side by side flight comparison of learned policy vs P-controller
# Why: shows whether RL agent performs better or worse than baseline
ax4 = axes[1, 1]
ax4.plot(rl_times, rl_altitudes, 
         color='#2E75B6', linewidth=2, label='RL Agent')
ax4.plot(time_p, alt_p, 
         color='#E07B39', linewidth=2, label='P-Controller')
ax4.axhline(TARGET_APOGEE, color='green', linewidth=1.5,
            linestyle='--', label=f'Target: {TARGET_APOGEE}m')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Altitude (m)')
ax4.set_title('RL Agent vs P-Controller — Final Flight')
ax4.legend()
ax4.grid(True, alpha=0.4)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot7_rl_training.png', dpi=150)
plt.close()
print("\nRL training plots saved to plots/plot7_rl_training.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: VISUALISE THE LEARNED Q-TABLE
# ─────────────────────────────────────────────────────────────────────────────
# What: shows which action the agent prefers at each (h, v) state
# Why: makes the learned policy interpretable — you can see the agent's
#      strategy across the entire flight envelope
# This is a heatmap of argmax_a Q(s,a) — the preferred action per state

fig, ax = plt.subplots(figsize=(10, 7))

# What: for each state find the best action index
best_actions = np.argmax(Q, axis=2)   # shape (20, 20)

# What: convert action indices to actual degrees for readability
best_deflections = np.array(ACTIONS)[best_actions]

im = ax.imshow(
    best_deflections.T,   # transpose: x=altitude, y=velocity
    origin='lower',
    aspect='auto',
    extent=[H_MIN, H_MAX, V_MIN, V_MAX],
    cmap='RdYlGn_r'       # red=high deflection, green=low deflection
)
plt.colorbar(im, ax=ax, label='Best Action — Airbrake Deflection (°)')
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Learned Policy — Best Airbrake Angle at Each Flight State')
plt.tight_layout()
plt.savefig('plots/plot8_rl_policy.png', dpi=150)
plt.close()
print("RL policy heatmap saved to plots/plot8_rl_policy.png")

print("\nPhase 2C complete.")