from periapsis.params.units import get_canonical_unit_scale

from .prior import Prior
import numpy as np
from scipy.special import ndtri


class LogNormalPrior(Prior):
    def __init__(self, mean: float, std: float, unit = None):
        if std <= 0:
            raise ValueError("Standard deviation must be positive for a LogNormalPrior.")
        if mean <= 0:
            raise ValueError("Mean must be positive for a LogNormalPrior.")
        self.mean = mean
        self.std = std

        self.min = 10**(mean - 10*std)
        self.max = 10**(mean + 10*std)

        self.constants = -0.5*np.log(2*np.pi*self.std**2)

        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)

    def __repr__(self):
        return f"LogNormalPrior(mean={self.mean}, std={self.std}, unit={self.unit if self.unit is not None else 'canonical'})"

    @property
    def canonical(self) -> LogNormalPrior:
        return self._canonical
    
    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        if unit is None:
            self._canonical = self
        else:
            factor = get_canonical_unit_scale(unit)
            self._canonical = LogNormalPrior(self.mean + np.log10(factor), self.std + np.log10(factor), unit=None)


    def sample(self, random_state, size=1):
        return 10**random_state.normal(loc=self.mean, scale=self.std, size=size)

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        log_x = np.log10(x)
        return (
            self.constants
            - 0.5*((log_x - self.mean)/self.std)**2
            - np.log(x*np.log(10))
        )

    def unp(self, u):
        return 10**(self.mean + self.std * ndtri(u))
