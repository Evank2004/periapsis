from abc import ABC, abstractmethod

class Prior(ABC):
    @abstractmethod
    def scale(self, factor):
        """Scale this numeric prior into canonical units."""
        pass

    @abstractmethod
    def sample(self, random_state, size=1):
        pass

    @abstractmethod
    def logpdf(self, x):
        pass

    @abstractmethod
    def unp(self, u):
        pass