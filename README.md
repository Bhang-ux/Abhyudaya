Abhyudaya 🚀
Show Image
Show Image
Show Image
Show Image
Show Image
A closed-loop rocket apogee control system that uses real aerodynamic physics and a proportional controller to precisely hit a target altitude by automatically deploying airbrakes during the coast phase of flight.

Motivation
In competitive rocketry (e.g. Spaceport America Cup), hitting a precise target apogee is a scored objective — missing by even a few hundred metres can mean disqualification. Rockets cannot throttle their motors mid-flight, so the only way to reduce apogee after burnout is aerodynamic drag via deployable airbrakes.
This project simulates exactly that problem: given a rocket coasting upward after motor burnout, how do you control airbrake deflection in real time to land as close to the target altitude as possible?
The same closed-loop control + uncertainty quantification framework applies directly to algorithmic trading, financial risk modelling, and autonomous systems.

System Architecture
constants.py
(mass, area, target, vs)
        │
        ├──────────────────────────────┐
        ▼                              ▼
Atmosphere.py               bilinear_interpolate.py
(air density ρ at h)        (Cx from Mach + deflection table)
        │                              │
        └──────────────┬───────────────┘
                       ▼
               acceleration.py
          (F_drag + gravity → net decel)
                       │
                       ▼
               predict_apogee.py
          (forward Euler integrator → predicted apogee)
                       │
                       ▼
            get_control_action.py
         (P-controller → airbrake angle δ)
                       │
                       ▼
              run_simulation.py
          (full flight loop + logging)

Key Results
MetricUncontrolledControlledApogee9,674.72 m5,000.06 mError vs Target (5,000 m)4,674.72 m0.06 mPercentage Error93.49 %0.001 %Max Airbrake Deflection0°50°Overshoot Reduction—4,674.66 m (99.99%)

Resume bullet: "Developed a closed-loop rocket apogee control simulation in Python; P-controller reduced altitude overshoot from 4,674 m to 0.06 m (99.99% improvement) against a 5,000 m target across 500+ simulated timesteps."


Plots
Altitude vs. Time
Show Image
Airbrake Deflection vs. Time
Show Image
Phase Portrait — Velocity vs. Altitude
Show Image
Controller Error over Time
Show Image

How to Run
1. Clone the repository
bashgit clone https://github.com/Bhang-ux/Abhyudaya.git
cd Abhyudaya
2. Install dependencies
bashpip install -r requirements.txt
3. Run the simulation
bashpython Abhyudaya/run_simulation.py
4. Generate plots
bashpython Abhyudaya/visualise.py

Project Structure
Abhyudaya/
├── Abhyudaya/
│   ├── __init__.py                 # Package entry point
│   ├── constants.py                # Rocket physical parameters and global constants
│   ├── Atmosphere.py               # ISA air density model as a function of altitude
│   ├── bilinear_interpolate.py     # 2D drag coefficient lookup table interpolation
│   ├── acceleration.py             # Net deceleration from aerodynamic drag + gravity
│   ├── predict_apogee.py           # Forward Euler integrator — predicts apogee from current state
│   ├── get_control_action.py       # P-controller — computes airbrake deflection angle
│   ├── run_simulation.py           # Main simulation loop — runs controlled vs uncontrolled
│   └── visualise.py                # Generates and saves all 4 diagnostic plots
├── plots/
│   ├── plot1_altitude_vs_time.png
│   ├── plot2_airbrake_deflection.png
│   ├── plot3_phase_portrait.png
│   └── plot4_controller_error.png
├── requirements.txt
├── LICENSE
└── README.md

Physics Model
The simulation models the coast phase of flight (after motor burnout) using:
Air density (ISA standard atmosphere):
T   = 15.04 - 0.00649 × h        (°C)
ρ   = k × (T + 273)^4.256        (kg/m³)
Aerodynamic drag:
F_drag = 0.5 × ρ × v² × |Cx| × A
Where Cx is the axial drag coefficient looked up from a bilinear interpolation table over Mach number and airbrake deflection angle.
Net acceleration:
a = -F_drag / m - g
P-Controller:
error = predict_apogee(h, v, δ=0) - target
δ     = clip(Kp × error, 0°, 50°)

Future Work
PhaseDescriptionStatusPhase 2AReplace bilinear lookup with Gaussian Process surrogate modelPlannedPhase 2BTrain neural network apogee predictor (3→64→64→1) for 200x inference speedupPlannedPhase 2CReplace P-controller with Reinforcement Learning agentPlannedPhase 2DBayesian hyperparameter optimisation of Kp via OptunaPlannedPhase 3AMonte Carlo uncertainty analysis — 1,000 trial apogee distribution + VaR metricsPlannedPhase 3BMission cost-benefit and sensitivity analysisPlannedPhase 4Streamlit interactive web dashboard — live parameter tuning and scenario plannerPlanned

License
MIT License — see LICENSE for details.