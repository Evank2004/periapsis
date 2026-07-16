from .prior import Prior
import numpy as np

class LogUniformPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.min = lower_bound
        self.max = upper_bound

    def sample(self, random_state, size=1):
        return 10**random_state.uniform(low=np.log10(self.lower_bound), high=np.log10(self.upper_bound), size=size)

    def logpdf(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return np.log(1/(x * np.log(self.upper_bound / self.lower_bound)))
        return -np.inf
    
    #this next function is for the Ultranest conversions
    def unp(self,u):
        return 10**(np.log10(self.lower_bound) + u * (np.log10(self.upper_bound) - np.log10(self.lower_bound)))