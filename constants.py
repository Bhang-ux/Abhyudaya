import numpy as np
import matplotlib.pyplot as plt
ROCKET_MASS = 50.0  # kg
ROCKET_DIAMETER = 0.160  # m
ROCKET_AREA = np.pi * (ROCKET_DIAMETER / 2)**2  # m^2, ~0.0201
TARGET_APOGEE = 10000.0  # m
GRAVITY = 9.81  # m/s^2
vs=343 #m/s
H_INIT = 3000        # burnout altitude (m)
V_INIT = 2.5 * vs    # burnout velocity (m/s) — Mach 1.3