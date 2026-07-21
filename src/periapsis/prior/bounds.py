from .prior import Prior
import numpy as np

class Bounds(Prior):
    def __init__(self, lower: float = None, upper: float = None):
        self.lower = lower
        self.upper = upper
        assert lower is not None or upper is not None, "At least one of `lower` or `upper` must be specified for a Bounds."

    def sample(self, random_state, size=1):
        raise NotImplementedError("Bounds does not support sampling.")

    def logpdf(self, x):
        if self.lower is not None and x < self.lower:
            return -np.inf
        if self.upper is not None and x > self.upper:
            return -np.inf
        return 0.0

    def unp(self, u):
        raise NotImplementedError("Bounds does not support unp transformation.")