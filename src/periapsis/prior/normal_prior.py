from periapsis.params.units import get_canonical_unit_scale

from .prior import Prior
import numpy as np
from scipy.special import ndtri

class NormalPrior(Prior):
    def __init__(self, mean: float, std: float, unit=None):
        if std <= 0:
            raise ValueError("Standard deviation must be positive for a NormalPrior.")
        self.mean = mean
        self.std = std

        #with how initial is set up, we still 
        #need to have defined boundaries if something goes into it
        self.min = mean - 10*std
        self.max = mean + 10*std

        self.constants = -0.5*np.log(2*np.pi*self.std**2)

        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)

    def __repr__(self):
        return f"NormalPrior(mean={self.mean}, std={self.std}, unit={self.unit if self.unit is not None else 'canonical'})"


    @property
    def canonical(self) -> NormalPrior:
        return self._canonical

    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        if unit is None:
            self._canonical = self
        else:
            factor = get_canonical_unit_scale(unit)
            self._canonical = NormalPrior(self.mean*factor, self.std*factor, unit=None)

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