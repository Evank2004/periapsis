from .prior import Prior
import numpy as np

class FixedPrior(Prior):
    def __init__(self, value: float):
        self.value = value

    def sample(self, random_state, size=1):
        return np.ones(size) * self.value

    def logpdf(self, x):
        if x == self.value:
            return 0.0
        else:
            return -float('inf')
        
    def unp(self, u):
        return np.ones_like(u) * self.value