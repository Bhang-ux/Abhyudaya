import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Abhyudaya import predict_apogee


def get_control_action(h, v, target_apogee, Kp):
    """
    Determines the required airbrake deflection angle.
    """
    # Predict apogee with brakes closed for a stable baseline
    predicted_apogee = predict_apogee.predict_apogee(h, v, 0)
    
    error = predicted_apogee - target_apogee
    
    delta = Kp * error
    
    # Saturate the output to physical limits [0, 50] degrees
    delta_saturated = np.clip(delta, 0, 50)
    
    return delta_saturated