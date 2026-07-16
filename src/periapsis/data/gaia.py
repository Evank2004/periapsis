from .data import Data
from orbit_package.model.orbit import Orbit


class GaiaData(Data):
    def __init__(self, t, *args):
        raise NotImplementedError("GaiaData is not implemented yet. This is a placeholder for the actual implementation of Gaia data handling.")

    def chi2(self, orbit: Orbit):
        raise NotImplementedError("GaiaData chi2 method is not implemented yet")