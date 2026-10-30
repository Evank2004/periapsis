from __future__ import annotations

from .prior import Prior
import numpy as np
from periapsis.params import get_canonical_unit_scale

class UniformPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, unit = None):
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound for a UniformPrior.")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.min = lower_bound
        self.max = upper_bound

        self.density = -np.log(upper_bound - lower_bound)
        
        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)

    def __repr__(self):
        return f"UniformPrior(lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, unit={self.unit if self.unit is not None else 'canonical'})"

    @property
    def canonical(self) -> UniformPrior:
        return self._canonical

    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        factor = get_canonical_unit_scale(unit)
        self._canonical = UniformPrior(self.lower_bound*factor, self.upper_bound*factor, unit=None)

    def sample(self, random_state, size=1):
        return random_state.uniform(low=self.lower_bound, high=self.upper_bound, size=size)

    def logpdf(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return self.density
        return -np.inf
    
    #this next function is for the Ultranest conversions
    def unp(self,u):
        return self.lower_bound + u * (self.upper_bound - self.lower_bound)