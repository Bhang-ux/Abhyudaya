import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Abhyudaya.predict_apogee import predict_apogee
from Abhyudaya.constants import TARGET_APOGEE

def get_pid_action(h, v, target, Kp, Ki, Kd, dt, integral, prev_error):
    """
    Computes airbrake deflection using a PID controller.

    What: combines three control terms to compute the optimal airbrake angle
    Why: PID fixes problems that P-controller alone cannot handle:
         - P alone has steady state error (never fully reaches target)
         - I eliminates steady state error by accumulating past mistakes
         - D prevents overshoot by reacting to rate of change

    Args:
        h:          current altitude (m)
        v:          current velocity (m/s)
        target:     target apogee (m)
        Kp:         proportional gain
        Ki:         integral gain
        Kd:         derivative gain
        dt:         timestep (s) — needed for integral and derivative
        integral:   running sum of past errors (carry between timesteps)
        prev_error: error from previous timestep (carry between timesteps)

    Returns:
        delta:       airbrake deflection angle (degrees, clipped to [0, 50])
        integral:    updated integral for next timestep
        prev_error:  current error to pass as prev_error next timestep

    Why return integral and prev_error?
        PID needs memory across timesteps. The integral accumulates every
        timestep. The derivative needs the previous error to compute change.
        We return them so the simulation loop can pass them back in.
        This is called stateful control — the controller remembers history.
    """

    # ── Compute predicted apogee ──────────────────────────────────────────
    # What: predict where rocket will end up with brakes closed
    # Why: same as P-controller — base prediction with no brakes
    #      gives us the error we need to correct
    predicted_apogee = predict_apogee(h, v, 0)

    # ── Compute error ─────────────────────────────────────────────────────
    # What: difference between predicted apogee and target
    # Why: positive error means rocket will overshoot — need to open brakes
    #      negative error means rocket will undershoot — keep brakes closed
    error = predicted_apogee - target

    # ── P term ────────────────────────────────────────────────────────────
    # What: proportional response to current error
    # Why: immediate correction — larger error = larger response
    #      This is exactly your existing P-controller
    p_term = Kp * error

    # ── I term ────────────────────────────────────────────────────────────
    # What: accumulate error over time
    # Why: if rocket consistently overshoots by small amount each step,
    #      P term alone becomes too small to fix it.
    #      Integral grows until correction is strong enough.
    # dt multiplication: integral is area under error curve = error × time
    #      Without dt, integral would depend on timestep size — wrong.
    #      With dt, integral has units of metres × seconds — physically correct.
    integral  += error * dt
    i_term     = Ki * integral

    # ── Anti-windup ───────────────────────────────────────────────────────
    # What: clip integral to prevent it growing too large
    # Why: if airbrakes are clipped at 50° for many timesteps, the integral
    #      keeps growing even though we cannot act on it.
    #      When brakes finally close, the huge integral causes massive
    #      overcorrection. Anti-windup prevents this.
    # Real world: anti-windup is used in every industrial PID controller —
    #      motor drives, chemical plants, aircraft autopilots
    integral = np.clip(integral, -10000, 10000)

    # ── D term ────────────────────────────────────────────────────────────
    # What: rate of change of error
    # Why: if error is decreasing fast the rocket is already correcting well
    #      D term reduces correction to prevent overshoot.
    #      If error is increasing fast, D term adds extra correction.
    # Division by dt: converts raw error difference to rate per second
    #      Without dt, derivative would depend on timestep — wrong.
    derivative = (error - prev_error) / dt
    d_term     = Kd * derivative

    # ── Combine all three terms ───────────────────────────────────────────
    # What: sum of P + I + D gives total correction signal
    # Why: each term handles a different aspect of the error:
    #      P = where are you now
    #      I = where have you been
    #      D = where are you going
    delta = p_term + i_term + d_term

    # ── Clip to physical limits ───────────────────────────────────────────
    # What: airbrakes cannot go below 0° or above 50°
    # Why: physical constraint of the mechanism
    #      Also prevents negative deflection (brakes cannot push rocket faster)
    delta = float(np.clip(delta, 0, 50))

    # ── Update state for next timestep ───────────────────────────────────
    prev_error = error

    return delta, integral, prev_error


def run_pid_simulation(Kp, Ki, Kd, h_init, v_init, target, dt=0.1):
    """
    Runs a complete flight simulation using PID controller.

    What: simulates entire coast phase with PID airbrake control
    Why: needed by Optuna (step 2) to evaluate any (Kp, Ki, Kd) combination
         and by smart_simulation.py for final comparison

    Args:
        Kp, Ki, Kd: PID gains
        h_init:     starting altitude (m)
        v_init:     starting velocity (m/s)
        target:     target apogee (m)
        dt:         timestep (s)

    Returns:
        times:      array of time values
        altitudes:  array of altitude values
        deltas:     array of airbrake angles
        apogee:     maximum altitude reached (m)
        error:      |apogee - target| in metres
    """
    from Abhyudaya.acceleration import acceleration

    # What: initialise simulation state
    h          = h_init
    v          = v_init
    t          = 0.0
    integral   = 0.0      # PID memory — starts at 0 (no accumulated error yet)
    prev_error = 0.0      # PID memory — starts at 0 (no previous error yet)

    # What: storage for logging
    times     = [t]
    altitudes = [h]
    deltas    = []

    # What: main simulation loop — identical to run_simulation.py
    # Runs until velocity goes negative (rocket starts falling = apogee reached)
    while v > 0:

        # What: get PID control action
        # Why: pass current integral and prev_error so PID has memory
        delta, integral, prev_error = get_pid_action(
            h, v, target, Kp, Ki, Kd, dt, integral, prev_error
        )

        # What: simulate one timestep of physics
        # Why: same forward Euler integration as existing simulation
        accel  = acceleration(h, v, delta)
        v     += accel * dt
        h     += v * dt
        t     += dt

        times.append(t)
        altitudes.append(h)
        deltas.append(delta)

    apogee = max(altitudes)
    error  = abs(apogee - target)

    return (
        np.array(times),
        np.array(altitudes),
        np.array(deltas),
        apogee,
        error
    )


if __name__ == '__main__':
    # What: quick test of PID controller with hand-tuned gains
    # Why: verify it works before plugging into Optuna
    from Abhyudaya.constants import H_INIT, V_INIT, TARGET_APOGEE

    print("Testing PID controller with initial gains...")
    print(f"H_INIT={H_INIT}m | V_INIT={V_INIT:.1f}m/s | TARGET={TARGET_APOGEE}m")
    print()

    # What: test three different gain combinations to see sensitivity
    # Why: gives you intuition for how each gain affects performance
    test_gains = [
        (0.05, 0.0,    0.0,   "P only (no I, no D)"),
        (0.05, 0.001,  0.0,   "PI controller"),
        (0.05, 0.001,  0.01,  "Full PID"),
        (0.1,  0.002,  0.05,  "More aggressive PID"),
    ]

    print(f"{'Gains':<30} {'Apogee':>10} {'Error':>10} {'Error %':>10}")
    print("-" * 65)

    for Kp, Ki, Kd, label in test_gains:
        _, _, _, apogee, error = run_pid_simulation(
            Kp, Ki, Kd, H_INIT, V_INIT, TARGET_APOGEE
        )
        pct = error / TARGET_APOGEE * 100
        print(f"{label:<30} {apogee:>10.2f}m {error:>10.2f}m {pct:>10.3f}%")

    print()
    print("Run optimise_pid.py next to find optimal gains automatically.")