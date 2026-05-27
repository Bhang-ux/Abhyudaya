import os

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Abhyudaya — Rocket Apogee Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚀 Abhyudaya — ML-Adaptive Rocket Apogee Control")
st.markdown("""
A closed-loop rocket flight simulation with **Physics-Based Modelling**,
**Gaussian Process** aerodynamic surrogate, **Neural Network** apogee predictor,
**Reinforcement Learning** controller, and **Bayesian-optimised PID** control.
""")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Flight Configuration")
st.sidebar.markdown("Adjust parameters and click **Run Simulation**")

from Abhyudaya.constants import vs

h_init = st.sidebar.slider(
    "Burnout Altitude H_INIT (m)",
    min_value=500, max_value=5000,
    value=3000, step=100
)

v_factor = st.sidebar.slider(
    "Burnout Velocity (× speed of sound)",
    min_value=0.8, max_value=2.0,
    value=2.5, step=0.05
)
v_init = v_factor * vs

target = st.sidebar.slider(
    "Target Apogee (m)",
    min_value=3000, max_value=20000,
    value=10000, step=500
)

st.sidebar.divider()
st.sidebar.markdown("**Controllers to compare:**")
show_uncontrolled = st.sidebar.checkbox("Uncontrolled", value=True)
show_p            = st.sidebar.checkbox("P-Controller (Kp=5)", value=True)
show_pid          = st.sidebar.checkbox("PID (Optuna-tuned)", value=True)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY CURRENT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Burnout Altitude", f"{h_init:,} m")
col2.metric("Burnout Velocity", f"{v_init:.0f} m/s  (Mach {v_factor:.1f})")
col3.metric("Target Apogee",    f"{target:,} m")

# Quick physics check
max_possible = h_init + v_init**2 / (2 * 9.81)
if target > max_possible * 0.95:
    st.warning(f"⚠️ Target may be unreachable. Max possible apogee ≈ {max_possible:.0f}m with these conditions.")
else:
    overshoot = max_possible - target
    col4.metric("Expected Overshoot", f"≈ {overshoot:.0f} m")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Simulation",
    "🎲 Monte Carlo Robustness",
    "🧠 ML Components"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    if not run_button:
        st.info("👈 Set your flight parameters in the sidebar and click **Run Simulation**")
        st.markdown("""
        **What this simulation does:**
        - Models rocket coast phase from burnout to apogee
        - Uses ISA standard atmosphere for air density
        - Gaussian Process surrogate for aerodynamic drag coefficients
        - Compares uncontrolled flight vs P-controller vs Optuna-tuned PID
        """)

    else:
        with st.spinner("Running simulations..."):

            results = {}

            # Import physics modules
            from Abhyudaya.acceleration import acceleration
            from Abhyudaya.pid_controller import run_pid_simulation
            from Abhyudaya.optimise_pid import find_optimal_pid

            def run_generic_simulation(h0, v0, kp, target_apogee, dt=0.1):
                """Simple P-controller simulation."""
                from Abhyudaya.predict_apogee import predict_apogee
                h, v, t = h0, v0, 0.0
                times, alts, deltas = [t], [h], []
                while v > 0:
                    pred  = predict_apogee(h, v, 0)
                    error = pred - target_apogee
                    delta = float(np.clip(kp * error, 0, 50))
                    a     = acceleration(h, v, delta)
                    v    += a * dt
                    h    += v * dt
                    t    += dt
                    times.append(t)
                    alts.append(h)
                    deltas.append(delta)
                return np.array(times), np.array(alts), np.array(deltas)

            def run_uncontrolled(h0, v0, dt=0.1):
                h, v, t = h0, v0, 0.0
                times, alts = [t], [h]
                while v > 0:
                    a  = acceleration(h, v, 0)
                    v += a * dt
                    h += v * dt
                    t += dt
                    times.append(t)
                    alts.append(h)
                return np.array(times), np.array(alts)

            # Run selected simulations
            if show_uncontrolled:
                t_u, a_u = run_uncontrolled(h_init, v_init)
                results['Uncontrolled'] = {
                    'times': t_u, 'alts': a_u,
                    'apogee': a_u.max(),
                    'error': abs(a_u.max() - target),
                    'color': 'gray', 'deltas': None
                }

            if show_p:
                t_p, a_p, d_p = run_generic_simulation(h_init, v_init, 5, target)
                results['P-Controller'] = {
                    'times': t_p, 'alts': a_p,
                    'apogee': a_p.max(),
                    'error': abs(a_p.max() - target),
                    'color': '#E07B39', 'deltas': d_p
                }

            if show_pid:
                with st.spinner("Finding optimal PID gains (Optuna — ~2 min)..."):
                    best_params, _ = find_optimal_pid(
                        h_init, v_init, target,
                        n_trials=50, verbose=False
                    )
                Kp = best_params['Kp']
                Ki = best_params['Ki']
                Kd = best_params['Kd']
                t_pid, a_pid, d_pid, apogee_pid, err_pid = run_pid_simulation(
                    Kp, Ki, Kd, h_init, v_init, target
                )
                results['PID (Optuna)'] = {
                    'times': t_pid, 'alts': a_pid,
                    'apogee': apogee_pid,
                    'error': err_pid,
                    'color': '#2E75B6', 'deltas': d_pid,
                    'gains': (Kp, Ki, Kd)
                }

        # ── Metrics row ───────────────────────────────────────────────────────
        st.subheader("Results")
        cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            with cols[i]:
                if name == 'Uncontrolled':
                    st.metric(name, f"{data['apogee']:.0f} m",
                              f"Overshoot: {data['error']:.0f} m")
                else:
                    st.metric(name, f"{data['apogee']:.2f} m",
                              f"Error: {data['error']:.4f} m")

        # ── Altitude plot ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        for name, data in results.items():
            ls = '--' if name == 'Uncontrolled' else '-'
            ax1.plot(data['times'], data['alts'],
                     color=data['color'], linewidth=2,
                     linestyle=ls, label=f"{name} ({data['apogee']:.0f}m)")
        ax1.axhline(target, color='green', linewidth=1.5,
                    linestyle='--', label=f'Target: {target}m')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Altitude vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        # ── Deflection plot ───────────────────────────────────────────────────
        ax2 = axes[1]
        for name, data in results.items():
            if data['deltas'] is not None and len(data['deltas']) > 0:
                t_arr = data['times'][:len(data['deltas'])]
                ax2.plot(t_arr, data['deltas'],
                         color=data['color'], linewidth=2, label=name)
        ax2.axhline(50, color='red', linewidth=1,
                    linestyle='--', label='Max 50°')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Airbrake Deflection (°)')
        ax2.set_title('Airbrake Deflection vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── PID gains display ─────────────────────────────────────────────────
        if show_pid and 'PID (Optuna)' in results:
            Kp, Ki, Kd = results['PID (Optuna)']['gains']
            st.subheader("Optimal PID Gains Found by Bayesian Optimisation")
            g1, g2, g3 = st.columns(3)
            g1.metric("Kp (Proportional)", f"{Kp:.6f}")
            g2.metric("Ki (Integral)",     f"{Ki:.6f}")
            g3.metric("Kd (Derivative)",   f"{Kd:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Monte Carlo Robustness Analysis")
    st.markdown("""
    Tests controller performance under **real-world uncertainty**.
    Each trial randomly varies physical parameters within realistic tolerances
    and measures how consistently the controller hits the target.
    """)

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        n_mc_trials = st.slider("Number of trials", 100, 1000, 500, step=100)
        mass_unc    = st.slider("Mass uncertainty (%)", 1, 10, 3)
    with mc_col2:
        vel_unc     = st.slider("Velocity uncertainty (%)", 1, 5, 2)
        cx_unc      = st.slider("Drag coefficient uncertainty (%)", 1, 10, 5)

    run_mc = st.button("🎲 Run Monte Carlo Analysis", type="primary")

    if run_mc:
        with st.spinner(f"Running {n_mc_trials} Monte Carlo trials..."):

            from Abhyudaya.constants import ROCKET_MASS
            from Abhyudaya.acceleration import acceleration
            from Abhyudaya.bilinear_interpolate import bilinear_interpolate

            def mc_acceleration(h, v, delta, mass, cx_mult):
                from Abhyudaya.constants import GRAVITY, ROCKET_AREA
                T   = 15.04 - 0.00649 * h
                k   = 1.2050
                rho = max(k * ((T + 273) / 288.08) ** 4.256, 0.001)
                mach = abs(v) / vs
                cx   = abs(bilinear_interpolate(mach, delta)) * cx_mult
                f_drag = 0.5 * rho * v**2 * cx * ROCKET_AREA
                return -f_drag / mass - GRAVITY

            def mc_predict_apogee(h, v, mass, cx_mult, dt=0.1):
                while v > 0:
                    a  = mc_acceleration(h, v, 0, mass, cx_mult)
                    v += a * dt
                    h += v * dt
                return h

            # Get optimal PID gains for nominal conditions
            best_params, _ = find_optimal_pid(
                h_init, v_init, target,
                n_trials=50, verbose=False
            )
            Kp = best_params['Kp']
            Ki = best_params['Ki']
            Kd = best_params['Kd']

            np.random.seed(42)
            pid_apogees = []
            unc_apogees = []

            progress = st.progress(0)
            for i in range(n_mc_trials):
                mass    = np.random.normal(ROCKET_MASS, ROCKET_MASS * mass_unc/100)
                v_pert  = np.random.normal(v_init, v_init * vel_unc/100)
                cx_mult = np.random.normal(1.0, cx_unc/100)
                mass    = np.clip(mass,    ROCKET_MASS*0.8, ROCKET_MASS*1.2)
                v_pert  = np.clip(v_pert,  v_init*0.8,      v_init*1.2)
                cx_mult = np.clip(cx_mult, 0.7,              1.3)

                # PID flight
                h, v  = h_init, v_pert
                integ = prev_err = 0.0
                max_h = h
                while v > 0:
                    pred   = mc_predict_apogee(h, v, mass, cx_mult)
                    err    = pred - target
                    integ  = np.clip(integ + err * 0.1, -10000, 10000)
                    deriv  = (err - prev_err) / 0.1
                    prev_err = err
                    delta  = float(np.clip(Kp*err + Ki*integ + Kd*deriv, 0, 50))
                    a      = mc_acceleration(h, v, delta, mass, cx_mult)
                    v     += a * 0.1
                    h     += v * 0.1
                    max_h  = max(max_h, h)
                pid_apogees.append(max_h)

                # Uncontrolled flight
                h, v  = h_init, v_pert
                max_h = h
                while v > 0:
                    a    = mc_acceleration(h, v, 0, mass, cx_mult)
                    v   += a * 0.1
                    h   += v * 0.1
                    max_h = max(max_h, h)
                unc_apogees.append(max_h)

                progress.progress((i+1) / n_mc_trials)

        pid_apogees = np.array(pid_apogees)
        unc_apogees = np.array(unc_apogees)
        pid_errors  = np.abs(pid_apogees - target)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Error",         f"{np.mean(pid_errors):.2f} m")
        m2.metric("Within 50m",         f"{np.mean(pid_errors<50)*100:.1f}%")
        m3.metric("P5-P95 Range",       f"{np.percentile(pid_apogees,95)-np.percentile(pid_apogees,5):.2f} m")
        m4.metric("VaR (95%)",          f"{np.percentile(pid_errors,95):.2f} m")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(unc_apogees, bins=40, alpha=0.5, color='gray',
                     label='Uncontrolled', density=True)
        axes[0].hist(pid_apogees, bins=40, alpha=0.7, color='#2E75B6',
                     label='PID Controller', density=True)
        axes[0].axvline(target, color='green', linewidth=2,
                        linestyle='--', label=f'Target: {target}m')
        axes[0].set_xlabel('Final Apogee (m)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Apogee Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.4)

        sorted_e = np.sort(pid_errors)
        cdf      = np.arange(1, len(sorted_e)+1) / len(sorted_e) * 100
        axes[1].plot(sorted_e, cdf, color='#2E75B6', linewidth=2)
        axes[1].axhline(95, color='red', linewidth=1.5,
                        linestyle='--', label='95th percentile')
        axes[1].axvline(50, color='green', linewidth=1,
                        linestyle=':', label='50m threshold')
        axes[1].set_xlabel('Apogee Error (m)')
        axes[1].set_ylabel('Cumulative % of Flights')
        axes[1].set_title('Cumulative Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ML Components Overview")

    exp1 = st.expander("📊 Gaussian Process Surrogate Model", expanded=True)
    with exp1:
        st.markdown("""
        **Problem:** Aerodynamic drag coefficient (Cx) is only known at 9 sparse CFD data points
        across a (Mach, deflection) grid.

        **Solution:** Gaussian Process regression learns a smooth Cx surface from these 9 points,
        enabling predictions at any (Mach, deflection) combination with uncertainty estimates.

        **Key results:**
        - RMSE on training data: 0.0026
        - Uncertainty quantification at every prediction point
        - Smooth physically-realistic interpolation vs straight-line bilinear method
        """)
        if os.path.exists('plots/plot5_gp_surrogate.png'):
            st.image('plots/plot5_gp_surrogate.png')

    exp2 = st.expander("🧠 Neural Network Apogee Predictor")
    with exp2:
        st.markdown("""
        **Problem:** Predicting apogee requires running a 200-step numerical integration loop —
        called thousands of times during RL training, making it very slow.

        **Solution:** A feedforward neural network (3→64→64→1, 4,481 parameters) trained on
        6,000 synthetic flight states learns to predict apogee in a single forward pass.

        **Key results:**
        - R² = 0.9999
        - MAE = 12.7m
        - 200x inference speedup at batch size 10
        """)
        if os.path.exists('plots/plot6_nn_performance.png'):
            st.image('plots/plot6_nn_performance.png')

    exp3 = st.expander("🤖 Reinforcement Learning Controller")
    with exp3:
        st.markdown("""
        **Approach:** Q-learning agent learns airbrake control policy purely from simulated
        experience — no explicit physics knowledge encoded.

        **Key insight:** RL demonstrates the model-free learning approach. While the
        Optuna-tuned PID achieves lower error (physics knowledge + Bayesian optimisation
        is hard to beat for deterministic systems), the RL agent shows that a viable
        control policy can emerge purely from trial and error.

        **Hyperparameters tuned automatically** using Bayesian optimisation (Optuna).
        """)
        if os.path.exists('plots/plot7_rl_training.png'):
            st.image('plots/plot7_rl_training.png')

    exp4 = st.expander("⚡ Bayesian PID Optimisation")
    with exp4:
        st.markdown("""
        **Problem:** PID gains (Kp, Ki, Kd) that minimise apogee error are different for
        every flight condition. Manual tuning is imprecise and does not generalise.

        **Solution:** Optuna's Tree Parzen Estimator (Bayesian optimisation) automatically
        searches the (Kp, Ki, Kd) space, intelligently focusing on promising regions.

        **Key results:**
        - 100 trials finds gains achieving < 0.001m error
        - 139,256x improvement over hand-tuned P-controller
        - Fully automatic — works for any H_INIT, V_INIT, TARGET_APOGEE
        """)
        if os.path.exists('plots/plot9_pid_comparison.png'):
            st.image('plots/plot9_pid_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em'>
Abhyudaya — Rocket Apogee Control System |
Physics Simulation + Gaussian Process + Neural Network + RL + Bayesian PID
</div>
""", unsafe_allow_html=True)