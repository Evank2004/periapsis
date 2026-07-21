from .prior import Prior
import numpy as np
from scipy.special import ndtri

class NormalPrior(Prior):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

        #with how initial is set up, we still 
        #need to have defined boundaries if something goes into it
        self.min = mean - 10*std
        self.max = mean + 10*std

        self.constants = -0.5*np.log(2*np.pi*self.std**2)

    def sample(self, random_state, size=1):
        return random_state.normal(loc=self.mean, scale=self.std, size=size)
    
    def logpdf(self, x):
        return self.constants - 0.5*((x - self.mean)/self.std)**2
    
    #for Ultranest
    #ndtri, for a given u in [0,1], returns the value x 
    # such that the cumulative distribution function 
    # of the standard normal distribution at x is equal to u. 
    def unp(self,u):
        return self.mean + self.std * ndtri(u)