from .data import Data
from orbit_package.model.orbit import Orbit
import numpy as np

class AstrometryData(Data):
    def __init__(self, t, x, y, x_err, y_err,ref_epoch=None,mu_x=None, mu_y=None):
        self.t = t
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

        if ref_epoch is None:
            self.ref_epoch = np.mean(t)
        else:
            self.ref_epoch = ref_epoch

        if mu_x is not None and mu_y is not None:
            self.mu_x = mu_x
            self.mu_y = mu_y
        else:
            self.mu_x = None
            self.mu_y = None

    def chi2(self, orbit: Orbit):
        x, y = orbit.astrometry(self.t)
        chi2_x = np.sum(((self.x - x) / self.x_err) ** 2)
        chi2_y = np.sum(((self.y - y) / self.y_err) ** 2)
        return chi2_x + chi2_y
    
    def t_series(self):
        return self.x, self.y,None, self.t
    

class RadialVelocityData(Data):
    def __init__(self, t, rv, rv_err):
        self.t = t
        self.rv = rv
        self.rv_err = rv_err

    def chi2(self, orbit: Orbit):
        vz = orbit.radial_velocity(self.t)
        chi2_rv = np.sum(((self.rv - vz) / self.rv_err) ** 2)
        return chi2_rv
    
    def t_series(self):
        return None, None,self.rv, self.t


class AstroRVData(Data):
    def __init__(self, t, x, x_err, y, y_err, rv, rv_err):
        self.t = t
        self.x = x
        self.x_err = x_err
        self.y = y
        self.y_err = y_err
        self.rv = rv
        self.rv_err = rv_err

    def chi2(self, orbit: Orbit):
        x, y = orbit.astrometry(self.t)
        vz = orbit.radial_velocity(self.t)
        chi2_x = np.sum(((self.x - x) / self.x_err) ** 2)
        chi2_y = np.sum(((self.y - y) / self.y_err) ** 2)
        chi2_rv = np.sum(((self.rv - vz) / self.rv_err) ** 2)
        return chi2_x + chi2_y + chi2_rv
    
    def t_series(self):
        return self.x, self.y,self.rv, self.t
    
    