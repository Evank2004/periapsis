from .data import Data
from periapsis.model.orbit import Orbit
import numpy as np

class SystemData(Data):
    def __init__(self, system):
        if system is None:
            raise ValueError(f"`system` must be provided for {self.__class__.__name__}. It can be either '1', '2', or 'relative'.")
        self.system = str(system)
        if self.system not in ['1', '2', 'relative']:
            raise ValueError(f"`system` must be either '1', '2', or 'relative'. Got '{system}' instead.")
        

    def _astrometry(self, orbit: Orbit):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _radial_velocity(self, orbit: Orbit):
        raise NotImplementedError("This method should be implemented in subclasses.")

class AstrometryData(SystemData):
    def __init__(self, t, x, y, x_err, y_err,ref_epoch=None,mu_x=None, mu_y=None, system=None):
        super().__init__(system)
        self.t = np.atleast_1d(t)
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.x_err = np.atleast_1d(x_err)
        self.y_err = np.atleast_1d(y_err)
        if self.x.shape != self.t.shape or self.y.shape != self.t.shape:
            raise ValueError("x and y must have the same shape as t")
        if self.x_err.shape != self.x.shape:
            self.x_err = np.broadcast_to(self.x_err, self.x.shape)
        if self.y_err.shape != self.y.shape:
            self.y_err = np.broadcast_to(self.y_err, self.y.shape)

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

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        return 2*len(self.t)

    def chi2(self, orbit: Orbit):
        x, y = orbit.astrometry(self.t, system=self.system)
        chi2_x = np.sum(((self.x - x) / self.x_err) ** 2)
        chi2_y = np.sum(((self.y - y) / self.y_err) ** 2)
        return chi2_x + chi2_y

    def _astrometry(self, orbit: Orbit):
        return self.x, self.y
    
    def t_series(self):
        return self.x, self.y,None, self.t
    

class RadialVelocityData(SystemData):
    def __init__(self, t, rv, rv_err, system=None):
        super().__init__(system)
        self.t = np.atleast_1d(t)
        self.rv = np.atleast_1d(rv)
        self.rv_err = np.atleast_1d(rv_err)
        if self.rv.shape != self.t.shape:
            raise ValueError("rv must have the same shape as t")
        if self.rv_err.shape != self.rv.shape:
            self.rv_err = np.broadcast_to(self.rv_err, self.rv.shape)

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        return len(self.t)

    def chi2(self, orbit: Orbit):
        vz = orbit.rv(self.t, system=self.system)
        chi2_rv = np.sum(((self.rv - vz) / self.rv_err) ** 2)
        return chi2_rv

    def _radial_velocity(self, orbit: Orbit):
        return self.rv
    
    def t_series(self):
        return None, None,self.rv, self.t
    
    