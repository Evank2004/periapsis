from __future__ import annotations

from abc import ABC, abstractmethod

class Prior(ABC):
    @abstractmethod
    def sample(self, random_state, size=1):
        pass

    @abstractmethod
    def logpdf(self, x):
        pass

    @abstractmethod
    def unp(self, u):
        pass

    @abstractmethod
    def set_unit(self, unit):
        """Overrides the unit of the prior, discarding the current unit"""
        pass

    @property
    @abstractmethod
    def canonical(self) -> Prior:
        """Returns a version of the prior in canonical units."""
        pass