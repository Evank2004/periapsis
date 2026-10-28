from __future__ import annotations

from periapsis.params.units import get_canonical_unit_scale

from .prior import Prior
import numpy as np

class Bounds(Prior):
    def __init__(self, lower: float = None, upper: float = None, unit = None):
        self.lower = lower
        self.upper = upper
        if lower is None and upper is None:
            raise ValueError("At least one of `lower` or `upper` must be specified for a Bounds.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Lower bound cannot be greater than upper bound.")

        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)
            

    @property
    def canonical(self) -> Bounds:
        return self._canonical

    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        self.unit = unit
        factor = get_canonical_unit_scale(unit)
        self._canonical = Bounds(
            lower=self.lower * factor if self.lower is not None else None,
            upper=self.upper * factor if self.upper is not None else None,
            unit=None
        )

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