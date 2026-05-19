import math
def atmosphere(h):  

    # constants
    k = 101.29 / (0.2869*(288.15 ** 5.256))   

    # Temperature model (in °C)
    T = 15.04 - 0.00649 * h

    # Density (kg/m^3)
    rho = k * (T + 273) ** 4.256
    return rho

