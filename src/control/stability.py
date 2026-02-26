def is_stable(viscosity, target, tolerance):
    return abs(viscosity - target) <= tolerance