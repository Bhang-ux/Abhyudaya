import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Atmosphere
import bilinear_interpolate
from constants import GRAVITY, ROCKET_AREA, ROCKET_MASS, vs

def acceleration(h, vr, defl):
    
    rho = Atmosphere.atmosphere(h)
    mach = vr / vs
    
    cx = bilinear_interpolate.bilinear_interpolate(mach, defl)
    
    drag_force = 0.5 * rho * vr**2 * abs(cx) * ROCKET_AREA
    
    # F_net = -F_drag - F_gravity

    acceleration = -drag_force / ROCKET_MASS - GRAVITY
    
    return acceleration