from .data import Data
from periapsis.model.orbit import Orbit
from periapsis.params.units import UnitSystem
from periapsis.params.params import flux_parameter
from astropy import units as u
from typing import Mapping, Any, Optional
import numpy as np

class SystemData(Data):
    def __init__(self, system, units: Optional[Mapping[str, Any]] = None):
        if system is None:
            raise ValueError(f"`system` must be provided for {self.__class__.__name__}. It can be either '1', '2', or 'relative'.")
        self.system = str(system)
        if self.system not in ['1', '2', 'relative']:
            raise ValueError(f"`system` must be either '1', '2', or 'relative'. Got '{system}' instead.")
        self.unit_system = UnitSystem(units)

    def parameter_unit(self,param_name):
        ''''Return the user-facing unit associated with model
        or nuisance parameter'''
        if param_name == 'astro_jitter' or param_name.startswith('astro_'):
            return self.unit_system.units['x']
        if param_name == 'rv_jitter' or param_name.startswith('rv_'):
            return self.unit_system.units['rv']

        return self.unit_system.get_parameter_unit(param_name)

    def parameter_dimension(self,param_name):
        ''''Return the user-facing dimension associated with model
        or nuisance parameter'''
        if param_name == 'astro_jitter' or param_name.startswith('astro_'):
            return self.unit_system.get_parameter_dimension('x')
        if param_name == 'rv_jitter' or param_name.startswith('rv_'):
            return self.unit_system.get_parameter_dimension('rv')

        return self.unit_system.get_parameter_dimension(param_name)
    
def _gaussian_log_likelihood(residuals,err,jitter):
    """ Computes the log-likelihood of a Gaussian distribution given residuals, errors, and jitter.
    """
    if jitter is not None:
        err = np.sqrt(err**2 + jitter**2)
        log_likelihood = -0.5 * np.sum((residuals / err) ** 2 + np.log(2 * np.pi * err ** 2))
    else:
        log_likelihood = -0.5 * np.sum((residuals/err)**2)
    return log_likelihood

def _orbit_value(orbit, name, default=None):
    if name in orbit.params:
        return orbit.params[name]
    if name in orbit.derived_params:
        return orbit.derived_params[name]
    return default

class AstrometryData(SystemData):
    def __init__(self, t, x, y, x_err, y_err,plxf_x=None,plxf_y=None,ref_epoch=None, units: Optional[Mapping[str, Any]] = None,
                system=None, instrument=None,band=None):
        if units is None:
            raise ValueError("Units must be provided for AstrometryData.")
        super().__init__(system, units)
        required_units = {"t", "x", "y", "x_err", "y_err"}
        missing_units = required_units - self.unit_system.units.keys()
        if missing_units:
            raise ValueError(f"Missing units for astrometry data: {sorted(missing_units)}")
        if self.unit_system.get_parameter_dimension("t") != "time":
            raise ValueError("Unit for 't' must represent time.")
        if self.unit_system.get_parameter_dimension("x") not in {"angle", "length"}:
            raise ValueError("Unit for 'x' must represent an angle or length.")

        self.t = np.atleast_1d(self.unit_system.convert_to_canonical("t", t))
        self.x = np.atleast_1d(self.unit_system.convert_to_canonical("x", x))
        self.y = np.atleast_1d(self.unit_system.convert_to_canonical("y", y))
        self.x_err = np.atleast_1d(self.unit_system.convert_to_canonical("x_err", x_err))
        self.y_err = np.atleast_1d(self.unit_system.convert_to_canonical("y_err", y_err))
        
        
        
        self.plxf_x = plxf_x
        self.plxf_y = plxf_y
        
        if self.x.shape != self.t.shape or self.y.shape != self.t.shape:
            raise ValueError("x and y must have the same shape as t")
        if self.x_err.shape != self.x.shape:
            self.x_err = np.broadcast_to(self.x_err, self.x.shape)
        if self.y_err.shape != self.y.shape:
            self.y_err = np.broadcast_to(self.y_err, self.y.shape)

        if ref_epoch is None:
            self.ref_epoch = np.mean(self.t)
        else:
            self.ref_epoch = self.unit_system.convert_to_canonical("t", ref_epoch)

        if plxf_x is not None and plxf_y is not None:
            self.plxf_x = np.atleast_1d(plxf_x)
            self.plxf_y = np.atleast_1d(plxf_y)
            if self.plxf_x.shape != self.t.shape:
                self.plxf_x = np.broadcast_to(self.plxf_x, self.t.shape)
            if self.plxf_y.shape != self.t.shape:
                self.plxf_y = np.broadcast_to(self.plxf_y, self.t.shape)
            self.parallax = True
        else:
            self.parallax = False

        
        self.instrument = instrument
        self.band = band
        

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        return 2*len(self.t)

    def chi2(self, orbit: Orbit,jitter_x=None,jitter_y=None):
        x, y = orbit.astrometry(self.t,self.plxf_x,self.plxf_y, system=self.system)

        if jitter_x is not None:
            total_x_err = np.sqrt(self.x_err**2 + jitter_x**2)
        else:
            total_x_err = self.x_err
        if jitter_y is not None:
            total_y_err = np.sqrt(self.y_err**2 + jitter_y**2)
        else:
            total_y_err = self.y_err
        chi2_x = np.sum(((self.x - x) / total_x_err) ** 2)
        chi2_y = np.sum(((self.y - y) / total_y_err) ** 2)
        return chi2_x + chi2_y

    def log_likelihood(self, orbit: Orbit,offset_names=None):
        flux_name = flux_parameter(self.band)
        flux_ratio = (
            orbit.params[flux_name]
            if flux_name is not None and flux_name in orbit.params
            else None
        )
        x, y = orbit.astrometry(
            self.t,
            self.plxf_x,
            self.plxf_y,
            system=self.system,
            flux_ratio=flux_ratio,
        )
        offset_names = set() if offset_names is None else set(offset_names)
        if f"astro_{self.instrument}_x_offset" in offset_names:
            x_offset = _orbit_value(orbit, f"astro_{self.instrument}_x_offset", default=0.0)
            x += x_offset
        if f"astro_{self.instrument}_y_offset" in offset_names:
            y_offset = _orbit_value(orbit, f"astro_{self.instrument}_y_offset", default=0.0)
            y += y_offset
        residuals_x = self.x - x
        residuals_y = self.y - y

        jitter_name = (
            'astro_jitter'
            if self.instrument is None
            else f"astro_{self.instrument}_jitter"
        )
        if jitter_name in orbit.params:
            jitter = orbit.params[jitter_name]
            ln_like_x = _gaussian_log_likelihood(residuals_x, self.x_err, jitter)
            ln_like_y = _gaussian_log_likelihood(residuals_y, self.y_err, jitter)
        else:
            ln_like_x = -0.5 * np.sum((residuals_x / self.x_err) ** 2)
            ln_like_y = -0.5 * np.sum((residuals_y / self.y_err) ** 2)
            
        return ln_like_x + ln_like_y

    def has_astrometry(self) -> bool:
        return True

    def has_radial_velocity(self) -> bool:
        return False

    def _astrometry(self, orbit: Orbit):
        return self.x, self.y

    def _radial_velocity(self, orbit: Orbit):
        raise NotImplementedError
    
    def t_series(self):
        return self.x, self.y,None, self.t
    

class RadialVelocityData(SystemData):
    def __init__(self, t, rv, rv_err, units: Optional[Mapping[str, Any]] = None, system=None,rv_trend=False,instrument=None):
        if units is None:
            raise ValueError("Units must be provided for RadialVelocityData.")
        super().__init__(system, units)
        required_units = {"t", "rv", "rv_err"}
        missing_units = required_units - self.unit_system.units.keys()
        if missing_units:
            raise ValueError(f"Missing units for radial velocity data: {sorted(missing_units)}")
        if self.unit_system.get_parameter_dimension("t") != "time":
            raise ValueError("Unit for 't' must represent time.")
        if self.unit_system.get_parameter_dimension("rv") != "velocity":
            raise ValueError("Unit for 'rv' must represent velocity.")

        self.t = np.atleast_1d(self.unit_system.convert_to_canonical("t", t))
        self.rv = np.atleast_1d(self.unit_system.convert_to_canonical("rv", rv))
        self.rv_err = np.atleast_1d(self.unit_system.convert_to_canonical("rv_err", rv_err))
        if self.rv.shape != self.t.shape:
            raise ValueError("rv must have the same shape as t")
        if self.rv_err.shape != self.rv.shape:
            self.rv_err = np.broadcast_to(self.rv_err, self.rv.shape)
        
        self.rv_trend = rv_trend
        self.instrument = instrument
        

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        return len(self.t)

    def chi2(self, orbit: Orbit,jitter=None):
        vz = orbit.rv(self.t, system=self.system)
        if jitter is not None:
            total_err = np.sqrt(self.rv_err**2 + jitter**2)
        else:
            total_err = self.rv_err
        chi2_rv = np.sum(((self.rv - vz) / total_err) ** 2)
        return chi2_rv

    def log_likelihood(self, orbit: Orbit,offset_names=None):
        vz = orbit.rv(self.t, system=self.system)
        offset_names = set() if offset_names is None else set(offset_names)
        if f"rv_{self.instrument}_offset" in offset_names:
            rv_offset = _orbit_value(orbit, f"rv_{self.instrument}_offset", default=0.0)
            vz += rv_offset
        residuals_rv = self.rv - vz

        jitter_name = (
            'rv_jitter'
            if self.instrument is None
            else f"rv_{self.instrument}_jitter"
        )
        if jitter_name in orbit.params:
            jitter = orbit.params[jitter_name]
            ln_like_rv = _gaussian_log_likelihood(residuals_rv, self.rv_err, jitter)
        else:
            ln_like_rv = -0.5 * np.sum((residuals_rv / self.rv_err) ** 2 )
        

        return ln_like_rv

    def has_astrometry(self) -> bool:
        return False

    def has_radial_velocity(self) -> bool:
        return True

    def _radial_velocity(self, orbit: Orbit):
        return self.rv

    def _astrometry(self, orbit: Orbit):
        raise NotImplementedError
    
    def t_series(self):
        return None, None,self.rv, self.t
    
    