from __future__ import annotations

from periapsis.params.units import get_canonical_unit_scale

from .prior import Prior
import numpy as np

class FixedPrior(Prior):
    def __init__(self, value: float, unit = None):
        self.value = value
        self.min = value
        self.max = value
        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)


    def __repr__(self):
        return f"FixedPrior(value={self.value}, unit={self.unit if self.unit is not None else 'canonical'})"

    @property
    def canonical(self) -> FixedPrior:
        return self._canonical

    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        if unit is None:
            self._canonical = self
        else:
            self._canonical = FixedPrior(self.value*get_canonical_unit_scale(unit), unit=None)

    def sample(self, random_state, size=1):
        return np.ones(size) * self.value

    def logpdf(self, x):
        raise NotImplementedError("FixedPrior does not have a logpdf. It is a fixed value.")
        
    def unp(self, u):
        return np.ones_like(u) * self.value