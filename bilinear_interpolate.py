import numpy as np

# Data (Deflection, Mach, Cx)


def bilinear_interpolate(mach, defl):
    data = {
    0:  {0.3: -0.6846, 0.6: -0.6597, 0.8: -0.7154},
    30: {0.3: -1.4306, 0.6: -1.3486, 0.8: -1.4127},
    50: {0.3: -2.1023, 0.6: -2.2383, 0.8: -2.3991}
}
        # Get the sorted list of Mach and Deflection points from the data
    mach_points = sorted(list(data[0].keys()))
    defl_points = sorted(list(data.keys()))

 
    # "clip" it to the nearest edge value. This prevents out-of-bounds errors.
    
    
    if mach < mach_points[0]:
        mach = mach_points[0]  # If mach is too low, use the lowest mach data
    elif mach > mach_points[-1]:
        mach = mach_points[-1] # If mach is too high, use the highest mach data

    if defl < defl_points[0]:
        defl = defl_points[0]
    elif defl > defl_points[-1]:
        defl = defl_points[-1]

    # Nearest deflections
    defls = sorted(data.keys())
    for i in range(len(defls)-1):
        if defls[i] <= defl <= defls[i+1]:
            d1, d2 = defls[i], defls[i+1]
            break
    else:
        raise ValueError(f"Deflection {defl} out of bounds {defls}")
    
    # Nearest machs
    machs = sorted(data[d1].keys())
    for j in range(len(machs)-1):
        if machs[j] <= mach <= machs[j+1]:
            m1, m2 = machs[j], machs[j+1]
            break
    else:
        raise ValueError(f"Mach {mach} out of bounds {machs}")

    # Corner values
    Q11 = data[d1][m1]
    Q12 = data[d1][m2]
    Q21 = data[d2][m1]
    Q22 = data[d2][m2]

    # Bilinear interpolation
    cx = (Q11 * (d2-defl) * (m2-mach) +
          Q21 * (defl-d1) * (m2-mach) +
          Q12 * (d2-defl) * (mach-m1) +
          Q22 * (defl-d1) * (mach-m1)) / ((d2-d1)*(m2-m1))

    return cx

