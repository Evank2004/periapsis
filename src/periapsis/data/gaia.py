from .common import SystemData
from periapsis.model.orbit import Orbit


class GaiaData(SystemData):
    def __init__(self, t, *args, system=None):
        super().__init__(system)
        raise NotImplementedError("GaiaData is not implemented yet. This is a placeholder for the actual implementation of Gaia data handling.")

    def chi2(self, orbit: Orbit):
        raise NotImplementedError("GaiaData chi2 method is not implemented yet")