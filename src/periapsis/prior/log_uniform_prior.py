from __future__ import annotations

from periapsis.params.units import get_canonical_unit_scale

from .prior import Prior
import numpy as np

class LogUniformPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, unit = None):
        if lower_bound <= 0 or upper_bound <= 0:
            raise ValueError("Bounds must be positive for a LogUniformPrior.")
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound for a LogUniformPrior.")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.min = lower_bound
        self.max = upper_bound

        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)

    def __repr__(self):
        return f"LogUniformPrior(lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, unit={self.unit if self.unit is not None else 'canonical'})"

    @property
    def canonical(self) -> LogUniformPrior:
        return self._canonical

    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        if unit is None:
            self._canonical = self
        else:
            factor = get_canonical_unit_scale(unit)
            self._canonical = LogUniformPrior(self.lower_bound*factor, self.upper_bound*factor, unit=None)

    def sample(self, random_state, size=1):
        return 10**random_state.uniform(low=np.log10(self.lower_bound), high=np.log10(self.upper_bound), size=size)

    def logpdf(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return np.log(1/(x * np.log(self.upper_bound / self.lower_bound)))
        return -np.inf
    
    #this next function is for the Ultranest conversions
    def unp(self,u):
        return 10**(np.log10(self.lower_bound) + u * (np.log10(self.upper_bound) - np.log10(self.lower_bound)))