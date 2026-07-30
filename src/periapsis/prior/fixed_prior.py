from .prior import Prior
import numpy as np

class FixedPrior(Prior):
    def __init__(self, value: float):
        self.value = value
        self.min = value
        self.max = value

    def sample(self, random_state, size=1):
        return np.ones(size) * self.value

    def logpdf(self, x):
        raise NotImplementedError("FixedPrior does not have a logpdf. It is a fixed value.")
        
    def unp(self, u):
        return np.ones_like(u) * self.value