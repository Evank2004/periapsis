from abc import ABC, abstractmethod

class Prior(ABC):
    @abstractmethod
    def sample(self, random_state, size=1):
        pass

    @abstractmethod
    def logpdf(self, x):
        pass