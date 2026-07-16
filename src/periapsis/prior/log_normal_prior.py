from .prior import Prior
import numpy as np
from scipy.special import ndtri


class LogNormalPrior(Prior):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

        self.min = 10**(mean - 10*std)
        self.max = 10**(mean + 10*std)

    def sample(self, random_state, size=1):
        return 10**random_state.normal(loc=self.mean, scale=self.std, size=size)

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        log_x = np.log10(x)
        return (
            -0.5*np.log(2*np.pi*self.std**2)
            - 0.5*((log_x - self.mean)/self.std)**2
            - np.log(x*np.log(10))
        )

    def unp(self, u):
        return 10**(self.mean + self.std * ndtri(u))
