import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
 
from Abhyudaya.predict_apogee import predict_apogee
from Abhyudaya.constants import vs, H_INIT, V_INIT
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: GENERATE TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
# What: create a grid of (h, v, delta) combinations and compute apogee for each
# Why: the NN needs thousands of input-output examples to learn from.
#      We generate them by running the physics simulation many times.
#      This is synthetic data generation — we use one model to train another.
#
# Real world parallel: in finance, you might generate synthetic market
#      scenarios from a stochastic model to train a risk prediction NN.
 
print("Generating training data...")
print("(Running physics simulation thousands of times — takes ~30 seconds)")
 
# What: define the range of each input variable
# Why: these cover the realistic flight envelope of your rocket.
#      Ranges are centred around H_INIT and V_INIT from constants.py
#      so training data always matches your configured flight conditions.
h_values     = np.linspace(500,    H_INIT * 8, 20)   # 20 altitude values
v_values     = np.linspace(10,     V_INIT,     20)   # 20 velocity values
delta_values = np.linspace(0,      50,         15)   # 15 deflection values
 
# What: total samples = 20 × 20 × 15 = 6000 data points
# Why: enough for a small NN to learn the mapping well
total = len(h_values) * len(v_values) * len(delta_values)
print(f"Total samples to generate: {total}")
print(f"Using H_INIT={H_INIT}m, V_INIT={V_INIT:.1f}m/s from constants.py")
 
X = []
y = []
 
count = 0
for h in h_values:
    for v in v_values:
        for delta in delta_values:
            # What: only simulate upward-moving rocket
            # Why: negative velocity means already falling — no apogee prediction needed
            if v > 0:
                apogee = predict_apogee(h, v, delta)
 
                # What: only keep physically valid apogees
                # Why: apogee must be above current altitude
                if apogee > h:
                    X.append([h, v, delta])
                    y.append(apogee)
 
            count += 1
            if count % 1000 == 0:
                print(f"  Progress: {count}/{total} ({100*count//total}%)")
 
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
 
print(f"\nValid training samples generated: {len(X)}")
print(f"Apogee range in dataset: {y.min():.0f} m to {y.max():.0f} m")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: SPLIT INTO TRAIN AND TEST SETS
# ─────────────────────────────────────────────────────────────────────────────
# What: split data into 80% training, 20% testing
# Why: test set simulates unseen data — tells us if model generalised
#      or just memorised training examples.
# random_state=42: same split every run for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: NORMALISE INPUTS
# ─────────────────────────────────────────────────────────────────────────────
# What: scale each input feature to mean=0, std=1
# Why: h, v, delta are on completely different scales.
#      Without normalisation the optimiser is dominated by h (largest scale)
#      and barely learns from delta. Normalising puts all features equal.
#
# CRITICAL: fit scaler ONLY on training data.
# Why: fitting on test data leaks future information into training.
#      The scaler must only know statistics from the training period.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
 
# What: also scale output (apogee values)
# Why: large output values (500-20000m) cause large gradients → unstable training
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CONVERT TO PYTORCH TENSORS
# ─────────────────────────────────────────────────────────────────────────────
# What: convert numpy arrays to PyTorch tensors
# Why: PyTorch layers only accept tensors, not numpy arrays.
#      Tensors also support automatic differentiation needed for backpropagation.
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t  = torch.FloatTensor(X_test_scaled)
y_test_t  = torch.FloatTensor(y_test_scaled)
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DEFINE THE NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
# Architecture: 3 → 64 → 64 → 1
#   Input:         3 neurons  (h, v, delta)
#   Hidden layer 1: 64 neurons + ReLU
#   Hidden layer 2: 64 neurons + ReLU
#   Output:        1 neuron   (predicted apogee)
#
# Why 64 neurons: rule of thumb ~20x input size for smooth regression problems
# Why 2 layers: enough to capture nonlinear physics without overfitting
# Why ReLU: introduces nonlinearity — without it stacked linear layers
#           are mathematically equivalent to a single linear layer
# Why no output activation: output is a raw number in metres —
#           any activation would constrain its range wrongly
 
class ApogeeNet(nn.Module):
 
    def __init__(self):
        super(ApogeeNet, self).__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu   = nn.ReLU()
 
    def forward(self, x):
        # Flow: input → layer1 → ReLU → layer2 → ReLU → layer3 → output
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
 
model = ApogeeNet()
print(f"\nNetwork architecture:")
print(model)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: LOSS FUNCTION AND OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
# MSELoss: Mean Squared Error — standard for regression
#          squares errors so large mistakes are penalised more
# Adam: adaptive learning rate optimiser — adjusts step size per weight
#       significantly faster and more robust than basic SGD
# lr=0.001: standard default for Adam — good balance of speed and stability
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
# What: repeatedly show network training data and adjust weights to reduce loss
# Epoch: one complete pass through all training data
# Batch: process 64 samples at a time — memory efficient, adds beneficial noise
# Why 500 epochs: enough for this smooth physics function to converge fully
 
EPOCHS     = 500
BATCH_SIZE = 64
train_losses = []
test_losses  = []
 
print(f"\nTraining for {EPOCHS} epochs...")
start_time = time.time()
 
for epoch in range(EPOCHS):
 
    model.train()
    epoch_loss = 0
    n_batches  = 0
 
    for i in range(0, len(X_train_t), BATCH_SIZE):
        X_batch = X_train_t[i : i + BATCH_SIZE]
        y_batch = y_train_t[i : i + BATCH_SIZE]
 
        # What: zero gradients — PyTorch accumulates by default
        # Why: stale gradients from previous batch corrupt current update
        optimizer.zero_grad()
 
        predictions = model(X_batch).squeeze()
        loss        = criterion(predictions, y_batch)
 
        # What: backpropagation — computes gradient of loss w.r.t every weight
        # Why: tells optimiser which direction to adjust each weight
        loss.backward()
 
        # What: update weights using computed gradients
        # Why: Adam takes a step that reduces loss for each weight independently
        optimizer.step()
 
        epoch_loss += loss.item()
        n_batches  += 1
 
    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)
 
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).squeeze()
        test_loss = criterion(test_pred, y_test_t)
        test_losses.append(test_loss.item())
 
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {test_loss.item():.6f}")
 
training_time = time.time() - start_time
print(f"\nTraining complete in {training_time:.1f} seconds")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: EVALUATE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).squeeze().numpy()
 
# What: inverse transform predictions back to metres
# Why: model was trained on scaled output — must undo scaling to get real values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
 
# R² score: fraction of variance explained by the model
#   R²=1.0 → perfect, R²=0.0 → no better than predicting mean
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2     = 1 - ss_res / ss_tot
 
# MAE: average absolute error in metres — most interpretable metric
mae = np.mean(np.abs(y_test - y_pred))
 
print(f"\n── Neural Network Performance ───────────────────────────")
print(f"R² Score:                    {r2:.6f}  (target: > 0.99)")
print(f"Mean Absolute Error:         {mae:.2f} m")
print(f"Max Error:                   {np.max(np.abs(y_test - y_pred)):.2f} m")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: SINGLE SAMPLE SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
# What: compare single-sample inference time of NN vs physics integrator
# Why: establishes baseline speedup before batching
test_h, test_v, test_delta = H_INIT, V_INIT * 0.5, 25.0
 
N_TRIALS = 1000
start = time.time()
for _ in range(N_TRIALS):
    physics_apogee = predict_apogee(test_h, test_v, test_delta)
physics_time = (time.time() - start) / N_TRIALS * 1000
 
test_input = torch.FloatTensor(scaler_X.transform([[test_h, test_v, test_delta]]))
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(N_TRIALS):
        nn_pred_scaled = model(test_input).item()
    nn_time = (time.time() - start) / N_TRIALS * 1000
 
nn_apogee = scaler_y.inverse_transform([[nn_pred_scaled]])[0][0]
speedup   = physics_time / nn_time
 
print(f"\n── Single Sample Speed Benchmark ────────────────────────")
print(f"Test state: h={test_h}m, v={test_v:.1f}m/s, delta={test_delta}°")
print(f"Physics integrator:  {physics_apogee:.2f} m  ({physics_time:.4f} ms/call)")
print(f"Neural network:      {nn_apogee:.2f} m  ({nn_time:.4f} ms/call)")
print(f"Speedup:             {speedup:.1f}x faster")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: BATCHED SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
# What: predict many states simultaneously in one forward pass
# Why: PyTorch matrix operations barely increase in cost as batch grows
#      but physics simulator scales linearly — this is where 200x comes from
# Real world: RL training evaluates hundreds of states per episode in one batch
 
BATCH_SIZES = [1, 10, 50, 100, 200, 500]
 
print(f"\n── Batched Speed Benchmark ──────────────────────────────")
print(f"{'Batch Size':>12} {'Physics (ms)':>14} {'NN (ms)':>10} {'Speedup':>10}")
 
for batch_size in BATCH_SIZES:
    h_batch     = np.random.uniform(500,    H_INIT * 8, batch_size)
    v_batch     = np.random.uniform(10,     V_INIT,     batch_size)
    delta_batch = np.random.uniform(0,      50,         batch_size)
    states      = list(zip(h_batch, v_batch, delta_batch))
 
    # Physics — must call one sample at a time
    start = time.time()
    for _ in range(100):
        for h, v, d in states:
            predict_apogee(h, v, d)
    physics_total_ms = (time.time() - start) / 100 * 1000
 
    # NN — entire batch in ONE forward pass
    X_batch_np     = np.array([[h, v, d] for h, v, d in states], dtype=np.float32)
    X_batch_scaled = scaler_X.transform(X_batch_np)
    X_batch_tensor = torch.FloatTensor(X_batch_scaled)
 
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            nn_preds = model(X_batch_tensor).squeeze()
        nn_total_ms = (time.time() - start) / 100 * 1000
 
    speedup = physics_total_ms / nn_total_ms
    print(f"{batch_size:>12} {physics_total_ms:>14.4f} {nn_total_ms:>10.4f} {speedup:>10.1f}x")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Neural Network Apogee Predictor', fontsize=14, fontweight='bold')
 
# Plot 1: Loss curve
# Healthy training: both lines decrease together, no divergence
ax1 = axes[0]
ax1.plot(train_losses, color='#2E75B6', label='Train Loss', linewidth=1.5)
ax1.plot(test_losses,  color='#E07B39', label='Test Loss',  linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss (scaled)')
ax1.set_title('Training & Test Loss Curve')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.4)
 
# Plot 2: Predicted vs Actual
# Perfect model = all dots on red diagonal line
ax2 = axes[1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=10, color='#2E75B6')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('Actual Apogee (m)')
ax2.set_ylabel('Predicted Apogee (m)')
ax2.set_title(f'Predicted vs Actual  (R²={r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.4)
 
# Plot 3: Error distribution
# Ideal: symmetric bell curve centred at zero — no systematic bias
errors = y_pred - y_test
ax3 = axes[2]
ax3.hist(errors, bins=40, color='#2E75B6', edgecolor='white', alpha=0.8)
ax3.axvline(0,    color='green',  linewidth=2,   linestyle='--', label='Zero error')
ax3.axvline(mae,  color='orange', linewidth=1.5, linestyle='--', label=f'MAE: {mae:.1f}m')
ax3.axvline(-mae, color='orange', linewidth=1.5, linestyle='--')
ax3.set_xlabel('Prediction Error (m)')
ax3.set_ylabel('Count')
ax3.set_title('Error Distribution')
ax3.legend()
ax3.grid(True, alpha=0.4)
 
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot6_nn_performance.png', dpi=150)
plt.close()
print("\nNN performance plot saved to plots/plot6_nn_performance.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: DROP-IN REPLACEMENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# What: same interface as predict_apogee() but uses NN internally
# Why: drop-in means get_control_action.py needs zero changes to use it
# Same inputs in, same output format out — just faster
 
def nn_predict_apogee(h, v, delta):
    """
    Predicts apogee using the trained neural network.
    Drop-in replacement for predict_apogee().
 
    Args:
        h:     current altitude (m)
        v:     current velocity (m/s)
        delta: airbrake deflection angle (degrees)
 
    Returns:
        predicted apogee (float, metres)
    """
    # Must apply same normalisation used during training
    # Why: model learned a mapping from scaled inputs — must receive scaled inputs
    x_scaled = scaler_X.transform([[h, v, delta]])
    x_tensor = torch.FloatTensor(x_scaled)
 
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).item()
 
    apogee = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return float(apogee)
 
 
if __name__ == '__main__':
    print(f"\n── Drop-in Function Test ────────────────────────────")
    print(f"Using H_INIT={H_INIT}m, V_INIT={V_INIT:.1f}m/s from constants.py")
    print(f"{'h (m)':>8} {'v (m/s)':>8} {'delta':>6} {'Physics':>10} {'NN':>10} {'Error':>8}")
 
    # What: test at states relative to configured initial conditions
    # Why: ensures test cases are always relevant to current constants.py settings
    test_states = [
        (H_INIT,         V_INIT * 0.9, 0),
        (H_INIT,         V_INIT * 0.9, 25),
        (H_INIT * 2,     V_INIT * 0.6, 50),
        (H_INIT * 0.8,   V_INIT,       10),
        (H_INIT * 4,     V_INIT * 0.3, 30),
    ]
    for h, v, d in test_states:
        phys = predict_apogee(h, v, d)
        nn   = nn_predict_apogee(h, v, d)
        err  = abs(phys - nn)
        print(f"{h:>8.0f} {v:>8.1f} {d:>6.0f} {phys:>10.1f} {nn:>10.1f} {err:>8.1f}")
 
    print(f"\nPhase 2B complete.")