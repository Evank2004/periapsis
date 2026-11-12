from __future__ import annotations

import numpy as np
from scipy.special import betaln, betaincinv
from scipy.special import xlogy, xlog1py # These help with numerical stability for logpdf calculations and NAN from log(0) issues

from periapsis.params.units import get_canonical_unit_scale
from .prior import Prior

class BetaPrior(Prior):
    def __init__(self, alpha: float, beta: float, lower_bound: float = 0.0, upper_bound: float = 1.0, unit=None):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta parameters must be positive for a BetaPrior.")
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound for a BetaPrior.")
        
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.min = lower_bound
        self.max = upper_bound
        self.width = upper_bound - lower_bound
        self._log_width = np.log(self.width)
       
        

        # Precompute the log normalization constant for efficiency
        self.constants = betaln(self.alpha, self.beta)

        if unit is None:
            self.unit = None
            self._canonical = self
        else:
            self.set_unit(unit)

    def __repr__(self):
        unit = self.unit if self.unit is not None else 'canonical'
        return (
            f"BetaPrior(alpha={self.alpha}, beta={self.beta}, "
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound}, unit={unit})"
        )

    @property
    def canonical(self) -> BetaPrior:
        return self._canonical

    def set_unit(self, unit):
        self.unit = unit
        if unit is None:
            self._canonical = self
            return

        factor = get_canonical_unit_scale(unit)
        self._canonical = BetaPrior(self.alpha, self.beta, self.lower_bound * factor, self.upper_bound * factor,unit=None)

    def sample(self, random_state, size=1):
        normalized_samps = random_state.beta(self.alpha, self.beta, size=size)
        return self.lower_bound + normalized_samps * self.width

    def logpdf(self, x):
        if np.any(x < self.lower_bound) or np.any(x > self.upper_bound):
            return -np.inf

        normalized_x = (x - self.lower_bound) / self.width    

        log_pdf = xlogy(self.alpha -1, normalized_x) + xlog1py(self.beta - 1, -normalized_x)  # log(1 - normalized_x) = log1p(-normalized_x)

        return log_pdf - self.constants - self._log_width
        
    def unp(self, u):
        normalized_x = betaincinv(self.alpha, self.beta, u)
        return self.lower_bound + normalized_x * self.width
    