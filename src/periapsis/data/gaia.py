from .data import Data
from periapsis.model.orbit import Orbit


class GaiaData(Data):
    def __init__(self, spsi,cpsi,t,plx_fac,x,err):
        self.spsi = spsi
        self.cpsi = cpsi
        self.t = t
        self.plx_fac = plx_fac
        self.x = x
        self.err = err

    def chi2(self, orbit: Orbit):
        raise NotImplementedError("GaiaData chi2 method is not implemented yet")