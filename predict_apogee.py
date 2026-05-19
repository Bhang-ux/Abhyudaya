import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Abhyudaya import acceleration



def predict_apogee(current_h, current_v, current_delta):
   
   # Simulates the flight from the current state to apogee.
    
    DT=0.1
    h_sim = current_h
    v_sim = current_v
    
    while v_sim > 0:
        accel_sim = acceleration.acceleration(h_sim, v_sim, current_delta)
        v_sim += accel_sim * DT
        h_sim += v_sim * DT
        
    return h_sim