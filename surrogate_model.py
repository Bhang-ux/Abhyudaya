import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What: imports numpy for array handling
# Why: GP needs data in array format, not plain Python lists
import numpy as np

# What: imports the Gaussian Process regressor from scikit-learn
# Why: this is the ready-made GP implementation we will use
from sklearn.gaussian_process import GaussianProcessRegressor

# What: imports two kernel types
# Why: kernels define how the GP measures similarity between points
# Matern: smooth but flexible curve — better than RBF for physical data
# WhiteKernel: accounts for noise in the data (measurement uncertainty)
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

import matplotlib.pyplot as plt

# ── Step 1: Define the training data ─────────────────────────────────────────
# What: these are your original 9 data points from bilinear_interpolate.py
# Why: this is the sparse CFD-derived drag coefficient table
mach_values = [0.3, 0.6, 0.8]
defl_values = [0, 30, 50]

cx_table = [
    [0.35, 0.55, 0.70],   # Mach 0.3
    [0.40, 0.62, 0.78],   # Mach 0.6
    [0.55, 0.75, 0.92],   # Mach 0.8
]

# What: flatten the 3x3 grid into a list of (mach, deflection) input pairs
# Why: GP takes a 2D input array where each row is one data point
# X_train shape: (9, 2) — 9 points, 2 features each (mach + deflection)
X_train = []
y_train = []

for i, mach in enumerate(mach_values):
    for j, defl in enumerate(defl_values):
        X_train.append([mach, defl])
        y_train.append(cx_table[i][j])

# What: convert to numpy arrays
# Why: scikit-learn requires numpy arrays not plain Python lists
X_train = np.array(X_train)   # shape (9, 2)
y_train = np.array(y_train)   # shape (9,)

# ── Step 2: Normalise inputs ──────────────────────────────────────────────────
# What: scale both input dimensions to [0, 1] range
# Why: GP kernels measure distance between points. If one feature ranges
#      0-50 and another 0.3-0.8, the GP ignores the small-range feature
#      completely. Normalising puts both dimensions on equal footing.
# Real world: normalisation is a fundamental step in EVERY ML pipeline.
#             In finance: normalising returns and volatilities before 
#             feeding them into risk models.
X_min = X_train.min(axis=0)   # [0.3, 0]  — minimum of each column
X_max = X_train.max(axis=0)   # [0.8, 50] — maximum of each column

# What: subtract min, divide by range → all values now between 0 and 1
# Why: Mach 0.3→0.0, 0.8→1.0 and Deflection 0→0.0, 50→1.0
#      now both dimensions are equally important to the GP
X_train_scaled = (X_train - X_min) / (X_max - X_min)

# ── Step 3: Define the kernel ─────────────────────────────────────────────────
# What: anisotropic Matern kernel — separate length scale per input dimension
# Why: even after normalisation, Mach and deflection may influence Cx at
#      different rates. length_scale=[1,1] with bounds lets the GP learn
#      a different sensitivity for each dimension independently.
#
# Matern(nu=2.5):
#   nu controls smoothness. 2.5 means twice differentiable — good for
#   physical systems that are smooth but not perfectly regular.
#
# WhiteKernel:
#   adds a small noise term. Prevents overfitting to potentially imperfect
#   CFD data. Represents real measurement uncertainty.
#
# length_scale_bounds: tells the optimiser to only search within this range
#   for the length scale values. Prevents it finding crazy solutions.
kernel = Matern(
            length_scale=[1.0, 1.0],
            length_scale_bounds=(1e-2, 10.0),
            nu=2.5
         ) + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-10, 1e-1)
         )

# ── Step 4: Create and train the GP model ─────────────────────────────────────
# What: creates the GP regressor object
# n_restarts_optimizer=10: runs the kernel hyperparameter optimiser 10 times
#   with different random starting points and picks the best result.
#   Why 10? Optimisers can get stuck in local minima. Multiple restarts
#   reduces that risk significantly.
# random_state=42: fixes randomness so results are reproducible every run.
#   42 is just a convention in ML — any fixed number works.
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)

# What: fits the GP to your 9 scaled training points
# Why: this is where the GP learns the shape of the Cx surface.
# Under the hood: optimises kernel hyperparameters by maximising the
#   log marginal likelihood — a score measuring how well the kernel
#   explains the observed data.
# After fit(): the GP has learned how fast Cx changes with Mach,
#   how fast it changes with deflection, and the noise level.
gp.fit(X_train_scaled, y_train)

print("GP model trained successfully.")
print(f"Optimised kernel: {gp.kernel_}")

# ── Step 5: Evaluate accuracy on training points ──────────────────────────────
# What: predict Cx at the same 9 training points and check accuracy
# Why: if the GP cannot fit its own training data well, something is wrong
# return_std=True: also returns uncertainty (standard deviation) at each point
#   At training points uncertainty should be near zero — GP saw these points
cx_pred, cx_std = gp.predict(X_train_scaled, return_std=True)

print("\n── Training Point Accuracy ──────────────────────────────")
print(f"{'Mach':>6} {'Defl':>6} {'Actual':>8} {'Predicted':>10} {'Std Dev':>8}")
for i in range(len(X_train)):
    print(f"{X_train[i,0]:>6.2f} {X_train[i,1]:>6.1f} {y_train[i]:>8.4f} "
          f"{cx_pred[i]:>10.4f} {cx_std[i]:>8.6f}")

# What: RMSE — Root Mean Squared Error
# Why: standard metric for regression accuracy. Lower is better.
#      0 means perfect prediction. 
#      Formula: sqrt(average of (predicted - actual)^2)
#      Squaring penalises large errors more than small ones.
rmse = np.sqrt(np.mean((cx_pred - y_train)**2))
print(f"\nRMSE on training data: {rmse:.6f}")
print(f"(Should be close to 0 — GP should fit its own training data well)")

# ── Step 6: Generate smooth prediction surface ────────────────────────────────
# What: create a fine 50x50 grid across the full (Mach, deflection) space
# Why: this is what the GP is for — smooth predictions between sparse points
#      The bilinear method gives straight lines. GP gives smooth curves.
mach_grid = np.linspace(0.3, 0.8, 50)
defl_grid = np.linspace(0, 50, 50)

# What: meshgrid creates all 50x50=2500 combinations of mach and deflection
# Why: we want to evaluate the GP everywhere on the surface, not just at
#      the 9 training points
M, D = np.meshgrid(mach_grid, defl_grid)

# What: flatten the 2D grid into a list of input pairs for the GP
# Why: GP predict() expects shape (n_points, 2), not a 2D grid
X_grid = np.column_stack([M.ravel(), D.ravel()])

# What: apply the SAME normalisation used during training
# Why: this is critical. The model was trained on scaled data.
#      If you feed unscaled data at prediction time, results will be wrong.
#      This is one of the most common bugs in ML pipelines.
X_grid_scaled = (X_grid - X_min) / (X_max - X_min)

# What: predict Cx and uncertainty at all 2500 grid points
cx_surface, cx_uncertainty = gp.predict(X_grid_scaled, return_std=True)

# What: reshape back to 50x50 for plotting
cx_surface     = cx_surface.reshape(M.shape)
cx_uncertainty = cx_uncertainty.reshape(M.shape)

# ── Step 7: Plot the surface ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Gaussian Process Surrogate Model — Aerodynamic Drag Surface',
             fontsize=14, fontweight='bold')

# Plot 1: Cx prediction surface
# What: filled contour plot showing predicted Cx across the whole space
# Why: visualises the smooth surface the GP learned from just 9 points
ax1 = axes[0]
cf1 = ax1.contourf(M, D, cx_surface, levels=20, cmap='viridis')
plt.colorbar(cf1, ax=ax1, label='Cx (drag coefficient)')
ax1.scatter(X_train[:,0], X_train[:,1], c='white', s=80,
            zorder=5, label='Training points (CFD data)')
ax1.set_xlabel('Mach Number')
ax1.set_ylabel('Airbrake Deflection (°)')
ax1.set_title('Predicted Cx Surface (GP Mean)')
ax1.legend()

# Plot 2: Uncertainty surface
# What: shows where the GP is confident vs uncertain
# Why: KEY advantage of GP over bilinear — it knows what it doesn't know.
#      High uncertainty = far from training data = less reliable prediction.
#      Low uncertainty = near training data = very confident.
# Real world: in finance this is like knowing your model's confidence
#             intervals — critical for risk management
ax2 = axes[1]
cf2 = ax2.contourf(M, D, cx_uncertainty, levels=20, cmap='Reds')
plt.colorbar(cf2, ax=ax2, label='Uncertainty (std dev)')
ax2.scatter(X_train[:,0], X_train[:,1], c='white', s=80,
            zorder=5, label='Training points')
ax2.set_xlabel('Mach Number')
ax2.set_ylabel('Airbrake Deflection (°)')
ax2.set_title('GP Uncertainty (High = Less Confident)')
ax2.legend()

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/plot5_gp_surrogate.png', dpi=150)
plt.close()
print("\nGP surface plot saved to plots/plot5_gp_surrogate.png")

# ── Step 8: Drop-in replacement function ─────────────────────────────────────
# What: same interface as bilinear_interpolate() but uses GP internally
# Why: drop-in replacement means acceleration.py needs zero changes —
#      just swap the import. Same inputs in, same output format out.
# This is called the Liskov Substitution Principle in software engineering:
#   a better implementation can replace an old one without breaking anything.
def gp_predict_cx(mach, deflection):
    """
    Predicts drag coefficient Cx using the trained GP surrogate model.
    Drop-in replacement for bilinear_interpolate().

    Args:
        mach:       Mach number (float, clamped to [0.3, 0.8])
        deflection: airbrake deflection angle in degrees (float, clamped to [0, 50])

    Returns:
        cx:          predicted drag coefficient (float)
        uncertainty: standard deviation of prediction — how confident the GP is (float)
    """
    # What: clip inputs to training data bounds
    # Why: GP extrapolation far outside training range is unreliable.
    #      Clipping is safer than letting it guess wildly.
    mach       = np.clip(mach, 0.3, 0.8)
    deflection = np.clip(deflection, 0, 50)

    # What: apply same normalisation as training — ALWAYS required
    x = np.array([[mach, deflection]])
    x_scaled = (x - X_min) / (X_max - X_min)

    cx, std = gp.predict(x_scaled, return_std=True)
    return float(cx[0]), float(std[0])


if __name__ == '__main__':
    print("\n── Drop-in Function Test ────────────────────────────")
    print(f"{'Mach':>6} {'Defl':>6} {'Cx Pred':>10} {'Uncertainty':>12}")
    test_cases = [(0.3, 0), (0.6, 30), (0.8, 50), (0.45, 15), (0.7, 40)]
    for mach, defl in test_cases:
        cx, unc = gp_predict_cx(mach, defl)
        print(f"{mach:>6.2f} {defl:>6.1f} {cx:>10.4f} {unc:>12.6f}")