from .prior import Prior
import numpy as np

class UniformPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.min = lower_bound
        self.max = upper_bound

        self.density = -np.log(upper_bound - lower_bound)

    def sample(self, random_state, size=1):
        return random_state.uniform(low=self.lower_bound, high=self.upper_bound, size=size)

    def logpdf(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return self.density
        return -np.inf
    
    #this next function is for the Ultranest conversions
    def unp(self,u):
        return self.lower_bound + u * (self.upper_bound - self.lower_bound)