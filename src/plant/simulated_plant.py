import numpy as np

class SimulatedPlant:
    def __init__(self):
        self.tau_y = 15
        self.K = 8
        self.n = 0.6

    def step(self, temp, pressure, flow):
        shear_rate = flow * 10
        viscosity = self.K * (shear_rate ** (self.n - 1))
        shear_stress = self.tau_y + self.K * (shear_rate ** self.n)

        # temperature effect
        viscosity *= np.exp(-0.02 * (temp - 70))

        return {
            "viscosity": float(viscosity),
            "shear_stress": float(shear_stress)
        }