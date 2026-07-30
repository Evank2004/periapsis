from .initial import InitialGuess
from .astrometry_initial import AstrometryInitialGuess
from .rv_initial import RVInitialGuess
from .gaia_initial import GaiaInitialGuess
from periapsis.data import JointData, GaiaData, RadialVelocityData, AstrometryData

import numpy as np

class JointInitialGuess(InitialGuess):
    def __init__(self, data: JointData, rng: np.random.RandomState, **priors):
        super().__init__(data, rng, **priors)
        self.initial_guesses = []
        for sub_data in data.datas:
            if isinstance(sub_data, GaiaData):
                self.initial_guesses.append(GaiaInitialGuess(sub_data, rng, **priors))
            elif isinstance(sub_data, RadialVelocityData):
                self.initial_guesses.append(RVInitialGuess(sub_data, rng, **priors))
            elif isinstance(sub_data, AstrometryData):
                self.initial_guesses.append(AstrometryInitialGuess(sub_data, rng, **priors))
            else:
                raise ValueError(f"Unsupported data type: {type(sub_data)}")

    def get_initial_guess(self, param_order, nwalkers: int) -> np.ndarray:
        initial_points = []
        for initial_guess in self.initial_guesses:
            initial_points.append(initial_guess.get_initial_guess(param_order, nwalkers))
        return np.mean(initial_points, axis=0)