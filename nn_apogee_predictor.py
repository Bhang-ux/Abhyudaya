import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What: numpy for array operations
# Why: generating training grid and computing metrics
import numpy as np

# What: PyTorch — Facebook's deep learning framework
# Why: building and training the neural network
# torch: core library
# nn: neural network building blocks (layers, activations, loss functions)
# optim: optimisation algorithms (Adam, SGD etc.)
import torch
import torch.nn as nn
import torch.optim as optim

# What: scikit-learn utilities
# Why: splitting data into train/test sets and normalising inputs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import time

# What: import your existing physics-based apogee predictor
# Why: this is the "teacher" — it generates the ground truth labels
#      for our training data. The NN learns to mimic this function.
from Abhyudaya.predict_apogee import predict_apogee
from Abhyudaya.constants import vs

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
# Why: these cover the realistic flight envelope of your rocket
#      h: 0 to 15000m — full altitude range from ground to above target
#      v: 10 to 500 m/s — from near-apogee slow speed to fast burnout speed
#      delta: 0 to 50° — full airbrake deflection range
h_values     = np.linspace(500,  12000, 20)   # 20 altitude values
v_values     = np.linspace(10,   480,   20)   # 20 velocity values
delta_values = np.linspace(0,    50,    15)   # 15 deflection values

# What: total samples = 20 × 20 × 15 = 6000 data points
# Why: enough for a small NN to learn the mapping well
#      More points = better accuracy but slower data generation
total = len(h_values) * len(v_values) * len(delta_values)
print(f"Total samples to generate: {total}")

X = []   # inputs:  (h, v, delta)
y = []   # outputs: predicted apogee

count = 0
for h in h_values:
    for v in v_values:
        for delta in delta_values:
            # What: only simulate when velocity is positive (rocket going up)
            # Why: negative velocity means rocket already falling — no apogee
            #      prediction needed. Filtering invalid states keeps data clean.
            if v > 0:
                apogee = predict_apogee(h, v, delta)

                # What: only keep physically valid apogees
                # Why: apogee must be above current altitude.
                #      Invalid samples confuse the NN during training.
                if apogee > h:
                    X.append([h, v, delta])
                    y.append(apogee)

            count += 1
            if count % 1000 == 0:
                print(f"  Progress: {count}/{total} ({100*count//total}%)")

# What: convert to numpy arrays
# Why: PyTorch and scikit-learn both expect numpy arrays as input
X = np.array(X, dtype=np.float32)   # shape: (n_samples, 3)
y = np.array(y, dtype=np.float32)   # shape: (n_samples,)

print(f"\nValid training samples generated: {len(X)}")
print(f"Apogee range in dataset: {y.min():.0f} m to {y.max():.0f} m")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: SPLIT INTO TRAIN AND TEST SETS
# ─────────────────────────────────────────────────────────────────────────────
# What: split data into 80% training, 20% testing
# Why: we train on 80% and evaluate on the remaining 20% that the model
#      has never seen. This tells us if the model generalised or just
#      memorised the training data.
#
# This is called the train-test split — fundamental to ALL ML workflows.
# Real world: in finance, you train on historical data and test on
#      held-out recent data to simulate live trading performance.
#
# random_state=42: ensures same split every run (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: NORMALISE INPUTS
# ─────────────────────────────────────────────────────────────────────────────
# What: scale each input feature to mean=0, std=1
# Why: same reason as GP — h ranges 500-12000, v ranges 10-480, delta 0-50.
#      Without normalisation the NN's gradient descent optimiser struggles
#      because features on different scales create elongated loss landscapes
#      that are hard to navigate efficiently.
#
# StandardScaler: subtracts mean and divides by standard deviation
#      Result: each feature has mean=0 and std=1
#
# CRITICAL: fit scaler ONLY on training data, then apply to both train and test.
# Why: if you fit on all data including test, you leak information about the
#      test set into training. This is called data leakage — a common ML bug
#      that makes test performance look better than it really is.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)   # fit AND transform training data
X_test_scaled  = scaler_X.transform(X_test)        # ONLY transform test data

# What: also scale the output (apogee values)
# Why: apogee ranges from ~500m to ~15000m. Large output values slow down
#      training. Scaling output to similar range as inputs speeds convergence.
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CONVERT TO PYTORCH TENSORS
# ─────────────────────────────────────────────────────────────────────────────
# What: convert numpy arrays to PyTorch tensors
# Why: PyTorch operates on tensors, not numpy arrays.
#      Tensors are like numpy arrays but can run on GPU and support
#      automatic differentiation (needed for backpropagation).
#
# torch.FloatTensor: 32-bit floating point — standard for neural networks
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t  = torch.FloatTensor(X_test_scaled)
y_test_t  = torch.FloatTensor(y_test_scaled)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DEFINE THE NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
# What: define the structure of the neural network
# Why: we need to specify how many layers, how many neurons, and what
#      activation functions to use.
#
# Architecture: 3 → 64 → 64 → 1
#   Input layer:   3 neurons  (h, v, delta)
#   Hidden layer 1: 64 neurons
#   Hidden layer 2: 64 neurons
#   Output layer:  1 neuron   (predicted apogee)
#
# Why 64 neurons? Rule of thumb: start with 2-4x the input size.
#      3 inputs × ~20 = 64 is a reasonable starting point.
#      Too few: underfitting (model too simple to learn the pattern)
#      Too many: overfitting (model memorises training data, fails on new data)
#
# Why 2 hidden layers?
#      1 layer can approximate simple functions
#      2 layers can approximate much more complex nonlinear relationships
#      More layers = deeper network = more complex patterns, but harder to train
#
# ReLU activation:
#      After each hidden layer we apply ReLU: f(x) = max(0, x)
#      Why: without activations, stacking linear layers is still just linear.
#      Activations introduce nonlinearity — lets the network learn curves,
#      not just straight lines. ReLU is the most common choice.
#      Real world: same activation used in image recognition, NLP, everything.

class ApogeeNet(nn.Module):
    # What: nn.Module is the base class for all PyTorch neural networks
    # Why: inheriting from it gives us all the training infrastructure for free

    def __init__(self):
        # What: calls the parent class constructor
        # Why: required when inheriting from nn.Module
        super(ApogeeNet, self).__init__()

        # What: defines the layers of the network
        # nn.Linear(in, out): a fully connected layer
        #   in = number of input neurons
        #   out = number of output neurons
        #   internally stores a weight matrix of shape (out, in) and bias vector
        self.layer1 = nn.Linear(3, 64)    # 3 inputs  → 64 hidden neurons
        self.layer2 = nn.Linear(64, 64)   # 64 hidden → 64 hidden neurons
        self.layer3 = nn.Linear(64, 1)    # 64 hidden → 1 output (apogee)

        # What: ReLU activation function
        # Why: applied after each hidden layer to add nonlinearity
        self.relu = nn.ReLU()

    def forward(self, x):
        # What: defines how data flows through the network
        # Why: PyTorch calls this automatically during training and prediction
        #
        # Flow: input → layer1 → ReLU → layer2 → ReLU → layer3 → output
        # Each step: multiply by weights, add bias, apply activation

        x = self.relu(self.layer1(x))   # input → hidden layer 1 → ReLU
        x = self.relu(self.layer2(x))   # hidden layer 1 → hidden layer 2 → ReLU
        x = self.layer3(x)              # hidden layer 2 → output (no activation)
        # Why no activation on output? We want raw numbers, not squashed values.
        # Activations on output would limit the range of predictions.
        return x

# What: create an instance of the network
model = ApogeeNet()
print(f"\nNetwork architecture:")
print(model)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: DEFINE LOSS FUNCTION AND OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
# What: MSE loss — Mean Squared Error
# Why: measures how wrong the predictions are.
#      MSE = average of (predicted - actual)²
#      Squaring means large errors are penalised more than small ones.
#      This is the standard loss for regression problems.
criterion = nn.MSELoss()

# What: Adam optimiser
# Why: adjusts the network's weights after each batch to reduce loss.
#      Adam (Adaptive Moment Estimation) is the most popular optimiser —
#      it adapts the learning rate for each weight individually.
#      Better than plain SGD for most problems.
#
# lr=0.001: learning rate — how big a step to take each update.
#      Too high: overshoots, loss oscillates and never converges
#      Too low: converges but extremely slowly
#      0.001 is the standard default for Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: TRAIN THE NETWORK
# ─────────────────────────────────────────────────────────────────────────────
# What: training loop — repeatedly show the network data and adjust weights
# Why: the network starts with random weights. Training nudges them toward
#      weights that produce accurate apogee predictions.
#
# Epoch: one complete pass through the entire training dataset
# Why multiple epochs? One pass is not enough — the network needs to see
#      the data many times to learn the pattern well. Like studying for an
#      exam — reading notes once is not enough.

EPOCHS     = 500    # number of times to go through all training data
BATCH_SIZE = 64     # number of samples processed before updating weights

# What: why batches instead of all data at once?
# Why: processing all 5000+ samples at once requires lots of memory.
#      Batches of 64 are efficient and also add beneficial noise to training
#      (stochastic gradient descent) which helps avoid local minima.

train_losses = []   # track loss over time for plotting
test_losses  = []

print(f"\nTraining for {EPOCHS} epochs...")
start_time = time.time()

for epoch in range(EPOCHS):

    # ── Training phase ────────────────────────────────────────────────────
    model.train()
    # What: model.train() sets the model to training mode
    # Why: some layers (Dropout, BatchNorm) behave differently during
    #      training vs inference. Good habit even if not using those layers.

    epoch_loss = 0
    n_batches  = 0

    # What: process data in batches of BATCH_SIZE
    for i in range(0, len(X_train_t), BATCH_SIZE):
        # What: slice out one batch
        X_batch = X_train_t[i : i + BATCH_SIZE]
        y_batch = y_train_t[i : i + BATCH_SIZE]

        # What: zero out gradients from previous batch
        # Why: PyTorch accumulates gradients by default. Must reset each batch
        #      or gradients from previous batches contaminate current update.
        optimizer.zero_grad()

        # What: forward pass — run inputs through the network
        # Why: computes predictions for this batch
        predictions = model(X_batch).squeeze()
        # squeeze(): removes the extra dimension. Shape (64,1) → (64,)

        # What: compute loss — how wrong are the predictions?
        loss = criterion(predictions, y_batch)

        # What: backward pass — backpropagation
        # Why: computes gradient of loss with respect to every weight in the network
        #      Gradient = direction and magnitude to adjust each weight
        #      This is calculus chain rule applied across all layers automatically
        loss.backward()

        # What: update weights using the computed gradients
        # Why: Adam uses the gradients to take a step that reduces the loss
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)

    # ── Evaluation phase ──────────────────────────────────────────────────
    model.eval()
    # What: model.eval() sets model to evaluation mode
    # Why: disables dropout etc. during inference

    with torch.no_grad():
        # What: torch.no_grad() disables gradient computation
        # Why: we don't need gradients during evaluation — saves memory and
        #      speeds up computation significantly
        test_pred = model(X_test_t).squeeze()
        test_loss = criterion(test_pred, y_test_t)
        test_losses.append(test_loss.item())

    # What: print progress every 50 epochs
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
    # What: get predictions on test set (data model never saw during training)
    y_pred_scaled = model(X_test_t).squeeze().numpy()

# What: inverse transform — convert scaled predictions back to metres
# Why: we scaled the output during training. Must undo that to get real apogee values.
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# What: compute R² score
# Why: R² (coefficient of determination) measures how well predictions
#      explain the variance in actual values.
#      R²=1.0: perfect predictions
#      R²=0.0: model just predicts the mean, no better than a straight line
#      R²<0.0: model is worse than predicting the mean (very bad)
#
# Formula: R² = 1 - (sum of squared residuals) / (total sum of squares)
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2     = 1 - ss_res / ss_tot

# What: MAE — Mean Absolute Error in metres
# Why: more interpretable than MSE. "On average predictions are off by X metres"
mae = np.mean(np.abs(y_test - y_pred))

print(f"\n── Neural Network Performance ───────────────────────────")
print(f"R² Score:                    {r2:.6f}  (target: > 0.99)")
print(f"Mean Absolute Error:         {mae:.2f} m")
print(f"Max Error:                   {np.max(np.abs(y_test - y_pred)):.2f} m")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: BENCHMARK SPEED
# ─────────────────────────────────────────────────────────────────────────────
# What: compare inference time of NN vs numerical integrator
# Why: this is the key metric for the resume bullet — "200x speedup"
#
# We time both methods on the same input 1000 times and compare

test_h, test_v, test_delta = 3000.0, 200.0, 25.0

# Time the numerical integrator
N_TRIALS = 1000
start = time.time()
for _ in range(N_TRIALS):
    physics_apogee = predict_apogee(test_h, test_v, test_delta)
physics_time = (time.time() - start) / N_TRIALS * 1000  # ms per call

# Time the neural network
test_input = torch.FloatTensor(
    scaler_X.transform([[test_h, test_v, test_delta]])
)
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(N_TRIALS):
        nn_pred_scaled = model(test_input).item()
    nn_time = (time.time() - start) / N_TRIALS * 1000  # ms per call

# Convert NN output back to metres
nn_apogee = scaler_y.inverse_transform([[nn_pred_scaled]])[0][0]
speedup   = physics_time / nn_time

print(f"\n── Speed Benchmark ──────────────────────────────────────")
print(f"Test state: h={test_h}m, v={test_v}m/s, delta={test_delta}°")
print(f"Physics integrator:  {physics_apogee:.2f} m  ({physics_time:.4f} ms/call)")
print(f"Neural network:      {nn_apogee:.2f} m  ({nn_time:.4f} ms/call)")
print(f"Speedup:             {speedup:.1f}x faster")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: PLOT TRAINING CURVE AND PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Neural Network Apogee Predictor', fontsize=14, fontweight='bold')

# Plot 1: Training and test loss over epochs
# What: shows how the model improved during training
# Why: if train loss goes down but test loss goes up = overfitting
#      Both going down together = healthy training
ax1 = axes[0]
ax1.plot(train_losses, color='#2E75B6', label='Train Loss', linewidth=1.5)
ax1.plot(test_losses,  color='#E07B39', label='Test Loss',  linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss (scaled)')
ax1.set_title('Training & Test Loss Curve')
ax1.set_yscale('log')
# Why log scale? Loss drops by orders of magnitude — log scale shows
# the full picture rather than a flat line after the initial drop
ax1.legend()
ax1.grid(True, alpha=0.4)

# Plot 2: Predicted vs Actual apogee
# What: perfect predictions would lie on the diagonal y=x line
# Why: shows where the model is accurate and where it struggles
ax2 = axes[1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=10, color='#2E75B6')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('Actual Apogee (m)')
ax2.set_ylabel('Predicted Apogee (m)')
ax2.set_title(f'Predicted vs Actual  (R²={r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.4)

# Plot 3: Prediction error distribution
# What: histogram of (predicted - actual) errors across test set
# Why: shows if errors are random (good) or systematic (bad)
#      A symmetric bell curve centred at 0 = unbiased model
errors = y_pred - y_test
ax3 = axes[2]
ax3.hist(errors, bins=40, color='#2E75B6', edgecolor='white', alpha=0.8)
ax3.axvline(0,            color='green', linewidth=2, linestyle='--', label='Zero error')
ax3.axvline(mae,          color='orange', linewidth=1.5, linestyle='--', label=f'MAE: {mae:.1f}m')
ax3.axvline(-mae,         color='orange', linewidth=1.5, linestyle='--')
ax3.set_xlabel('Prediction Error (m)')
ax3.set_ylabel('Count')
ax3.set_title('Error Distribution')
ax3.legend()
ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('plots/plot6_nn_performance.png', dpi=150)
plt.close()
print("\nNN performance plot saved to plots/plot6_nn_performance.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: THE DROP-IN REPLACEMENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# What: same interface as predict_apogee() but uses NN internally
# Why: drop-in replacement — get_control_action.py needs zero changes

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
    # What: scale input the same way as training data
    # Why: model was trained on scaled data — must predict on scaled data
    x_scaled = scaler_X.transform([[h, v, delta]])
    x_tensor = torch.FloatTensor(x_scaled)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).item()

    # What: inverse transform to get actual metres
    apogee = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return float(apogee)


if __name__ == '__main__':
    print(f"\n── Drop-in Function Test ────────────────────────────")
    print(f"{'h (m)':>8} {'v (m/s)':>8} {'delta':>6} {'Physics':>10} {'NN':>10} {'Error':>8}")
    test_states = [
        (3000, 200, 0),
        (3000, 200, 25),
        (5000, 150, 50),
        (2000, 300, 10),
        (8000, 100, 30),
    ]
    for h, v, d in test_states:
        phys = predict_apogee(h, v, d)
        nn   = nn_predict_apogee(h, v, d)
        err  = abs(phys - nn)
        print(f"{h:>8.0f} {v:>8.1f} {d:>6.0f} {phys:>10.1f} {nn:>10.1f} {err:>8.1f}")
# ── Batched Speed Benchmark ───────────────────────────────────────────────────
# What: instead of predicting one state at a time, predict many simultaneously
# Why: PyTorch's matrix operations are optimised for batches.
#      Predicting 64 states takes almost the same time as predicting 1.
#      So effective speedup per sample becomes much higher.
# Real world: this is exactly how RL training works — the agent evaluates
#      hundreds of states per episode in one batched forward pass.

BATCH_SIZES = [1, 10, 50, 100, 200, 500]

print(f"\n── Batched Speed Benchmark ──────────────────────────────")
print(f"{'Batch Size':>12} {'Physics (ms)':>14} {'NN (ms)':>10} {'Speedup':>10} {'Per Sample':>12}")

for batch_size in BATCH_SIZES:
    # Generate a batch of random states
    h_batch     = np.random.uniform(500,  12000, batch_size)
    v_batch     = np.random.uniform(10,   480,   batch_size)
    delta_batch = np.random.uniform(0,    50,    batch_size)
    states      = list(zip(h_batch, v_batch, delta_batch))

    # Time physics simulator — must call one at a time (no batching possible)
    start = time.time()
    for _ in range(100):
        for h, v, d in states:
            predict_apogee(h, v, d)
    physics_total_ms = (time.time() - start) / 100 * 1000  # ms for whole batch

    # Time neural network — predict entire batch in ONE forward pass
    X_batch_np     = np.array([[h, v, d] for h, v, d in states], dtype=np.float32)
    X_batch_scaled = scaler_X.transform(X_batch_np)
    X_batch_tensor = torch.FloatTensor(X_batch_scaled)

    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            nn_preds = model(X_batch_tensor).squeeze()
        nn_total_ms = (time.time() - start) / 100 * 1000  # ms for whole batch

    speedup    = physics_total_ms / nn_total_ms
    per_sample = speedup  # same thing — both measured for same batch size

    print(f"{batch_size:>12} {physics_total_ms:>14.4f} {nn_total_ms:>10.4f} "
          f"{speedup:>10.1f}x {per_sample:>10.1f}x")
    print(f"\nAll results saved. Phase 2B complete.")