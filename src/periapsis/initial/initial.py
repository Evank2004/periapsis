from abc import abstractmethod, ABC
import numpy as np
from periapsis.data import Data
from periapsis.prior import Prior

class InitialGuess(ABC):
    """Class for obtaining an intial guess on sampled parameters"""
    def __init__(self, data: Data, rng: np.random.RandomState, **priors: Prior):
        self.data = data
        self.rng = rng
        self.priors = priors

    @abstractmethod
    def get_initial_guess(self, param_order, nwalkers: int) -> np.ndarray:
        """
        Returns an intial guess on sampled parameters based on the data
        """
        pass