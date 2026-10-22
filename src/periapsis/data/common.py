from .data import Data
from periapsis.model.orbit import Orbit
from periapsis.params.units import UnitSystem, is_quantity
from periapsis.params.params import flux_parameter
from typing import Mapping, Any, Optional
import numpy as np

class SystemData(Data):
    def __init__(self, system, units: Optional[Mapping[str, Any]] = None):
        if system is None:
            raise ValueError(f"`system` must be provided for {self.__class__.__name__}. It can be either '1', '2', or 'relative'.")
        self.system = str(system)
        if self.system not in ['1', '2', 'relative']:
            raise ValueError(f"`system` must be either '1', '2', or 'relative'. Got '{system}' instead.")
        if units is None:
            raise ValueError(f"Units must be provided for {self.__class__.__name__}.")
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
    
    return log_likelihood

def _orbit_value(orbit, name, default=None):
    if name in orbit.params:
        return orbit.params[name]
    if name in orbit.derived_params:
        return orbit.derived_params[name]
    return default


class _AstrometryDataBase(SystemData):
    def __init__(self, t, x, y, x_err, y_err,plxf_x=None,plxf_y=None,ref_epoch=None, units: Optional[Mapping[str, Any]] = None,
                system=None, instrument=None,band=None):
        if units is None:
            if is_quantity(t) and is_quantity(x) and is_quantity(y) and is_quantity(x_err) and is_quantity(y_err):
                units = {
                    "t": t.unit,
                    "x": x.unit,
                    "y": y.unit,
                    "x_err": x_err.unit,
                    "y_err": y_err.unit,
                }
                t = t.value
                x = x.value
                y = y.value
                x_err = x_err.value
                y_err = y_err.value
        super().__init__(system, units)
        required_units = {"t", "x", "y", "x_err", "y_err"}
        missing_units = required_units - self.unit_system.units.keys()
        if missing_units:
            raise ValueError(f"Missing units for astrometry data: {sorted(missing_units)}")
        if self.unit_system.get_parameter_dimension("t") != "time":
            raise ValueError("Unit for 't' must represent time.")
        if self.unit_system.get_parameter_dimension("x") not in {"angle", "length"}:
            raise ValueError("Unit for 'x' must represent an angle or length.")
        if self.unit_system.get_parameter_dimension("y") not in {"angle", "length"}:
            raise ValueError("Unit for 'y' must represent an angle or length.")
        if self.unit_system.get_parameter_dimension("x_err") not in {"angle", "length"}:
            raise ValueError("Unit for 'x_err' must represent an angle or length.")
        if self.unit_system.get_parameter_dimension("y_err") not in {"angle", "length"}:
            raise ValueError("Unit for 'y_err' must represent an angle or length.")

        self.t = np.atleast_1d(t)
        self.t_canonical = self.unit_system.convert_to_canonical("t", self.t)

        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.x_err = np.atleast_1d(x_err)
        self.y_err = np.atleast_1d(y_err)
        if ref_epoch is None:
            self.ref_epoch = np.mean(self.t)
        else:
            self.ref_epoch = ref_epoch
        self.ref_epoch_canonical = self.unit_system.convert_to_canonical("t", self.ref_epoch)
        
        self.plxf_x = plxf_x
        self.plxf_y = plxf_y
        
        if self.x.shape != self.t.shape or self.y.shape != self.t.shape:
            raise ValueError("x and y must have the same shape as t")
        if self.x_err.shape != self.x.shape:
            self.x_err = np.broadcast_to(self.x_err, self.x.shape)
        if self.y_err.shape != self.y.shape:
            self.y_err = np.broadcast_to(self.y_err, self.y.shape)

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
        return 2*len(self.t_canonical)

    def has_astrometry(self) -> bool:
        return True

    def has_radial_velocity(self) -> bool:
        return False

    def _radial_velocity(self, orbit: Orbit):
        raise NotImplementedError


class AstrometryData(_AstrometryDataBase):
    """
    Class for handling 2D astrometry data (on-sky positions and their uncertainties) for an individual system.
    """
    def __new__(cls, t, x, y, x_err, y_err, plxf_x=None, plxf_y=None, ref_epoch=None, units: Optional[Mapping[str, Any]] = None,
                system=None, instrument=None, band=None) -> _AstrometryDataBase:
        if units is None:
            if is_quantity(t) and is_quantity(x) and is_quantity(y) and is_quantity(x_err) and is_quantity(y_err):
                units = {
                    "t": t.unit,
                    "x": x.unit,
                    "y": y.unit,
                    "x_err": x_err.unit,
                    "y_err": y_err.unit,
                }
            else:
                raise ValueError(f"Units must be provided for {cls.__name__}.")
        unit_system = UnitSystem(units)
        if unit_system.get_parameter_dimension("x") == "length" and unit_system.get_parameter_dimension("y") == "length":
            return PhysicalAstrometryData(t, x, y, x_err, y_err, plxf_x, plxf_y, ref_epoch, units, system, instrument, band)
        elif unit_system.get_parameter_dimension("x") == "angle" and unit_system.get_parameter_dimension("y") == "angle":
            return super().__new__(cls)
        else:
            raise ValueError("Units for 'x' and 'y' must either both represent a physical length or both represent a projected angular length.")

    def __init__(self, t, x, y, x_err, y_err, plxf_x=None, plxf_y=None, ref_epoch=None, units: Mapping[str, Any] = None,
                    system=None, instrument=None, band=None):
        super().__init__(t, x, y, x_err, y_err, plxf_x, plxf_y, ref_epoch, units, system, instrument, band)
        if self.unit_system.get_parameter_dimension("x") != "angle" or self.unit_system.get_parameter_dimension("y") != "angle":
            raise ValueError("Units for 'x' and 'y' must represent an angle.")

        self.x_ang_canonical = self.unit_system.convert_to_canonical("x", self.x)
        self.y_ang_canonical = self.unit_system.convert_to_canonical("y", self.y)
        self.x_err_ang_canonical = self.unit_system.convert_to_canonical("x_err", self.x_err)
        self.y_err_ang_canonical = self.unit_system.convert_to_canonical("y_err", self.y_err)

    # FIXME test chi2 method
    def chi2(self, orbit: Orbit,jitter_x=None,jitter_y=None):
        x_ang, y_ang = orbit.astrometry_ang(self.t_canonical, self.plxf_x, self.plxf_y, system=self.system)
    
        if jitter_x is not None:
            total_x_ang_err = np.sqrt(self.x_err_ang_canonical**2 + jitter_x**2)
        else:
            total_x_ang_err = self.x_err_ang_canonical
        if jitter_y is not None:
            total_y_ang_err = np.sqrt(self.y_err_ang_canonical**2 + jitter_y**2)
        else:
            total_y_ang_err = self.y_err_ang_canonical
        chi2_x = np.sum(((self.x_ang_canonical - x_ang) / total_x_ang_err) ** 2)
        chi2_y = np.sum(((self.y_ang_canonical - y_ang) / total_y_ang_err) ** 2)
        return chi2_x + chi2_y

    #FIXME test log_likelihood method
    def log_likelihood(self, orbit: Orbit, offset_names=None):
        flux_name = flux_parameter(self.band)
        flux_ratio = (
            orbit.params[flux_name]
            if flux_name is not None and flux_name in orbit.params
            else None
        )
        x_ang, y_ang = orbit.astrometry_ang(
            self.t_canonical,
            self.plxf_x,
            self.plxf_y,
            system=self.system,
            flux_ratio=flux_ratio,
        )
        offset_names = set() if offset_names is None else set(offset_names)
        if f"astro_{self.instrument}_x_offset" in offset_names:
            x_offset = _orbit_value(orbit, f"astro_{self.instrument}_x_offset", default=0.0)
            x_ang += x_offset
        if f"astro_{self.instrument}_y_offset" in offset_names:
            y_offset = _orbit_value(orbit, f"astro_{self.instrument}_y_offset", default=0.0)
            y_ang += y_offset
        residuals_x = self.x_ang_canonical - x_ang
        residuals_y = self.y_ang_canonical - y_ang

        jitter_name = (
            'astro_jitter'
            if self.instrument is None
            else f"astro_{self.instrument}_jitter"
        )

        jitter = orbit.params.get(jitter_name)

        ln_like_x = _gaussian_log_likelihood(residuals_x,self.x_err_ang_canonical,jitter)
        ln_like_y = _gaussian_log_likelihood(residuals_y,self.y_err_ang_canonical,jitter)
            
        return ln_like_x + ln_like_y

    def _astrometry(self, orbit: Orbit):
        return self.x_ang_canonical, self.y_ang_canonical


class PhysicalAstrometryData(_AstrometryDataBase):
    def __init__(self, t, x, y, x_err, y_err, plxf_x=None, plxf_y=None, ref_epoch=None, units: Mapping[str, Any] = None,
                 system=None, instrument=None, band=None):
        super().__init__(t, x, y, x_err, y_err, plxf_x, plxf_y, ref_epoch, units, system, instrument, band)
        if self.unit_system.get_parameter_dimension("x") != "length" or self.unit_system.get_parameter_dimension("y") != "length":
            raise ValueError("Units for 'x' and 'y' must represent a physical length.")

        self.x_canonical = self.unit_system.convert_to_canonical("x", self.x)
        self.y_canonical = self.unit_system.convert_to_canonical("y", self.y)
        self.x_err_canonical = self.unit_system.convert_to_canonical("x_err", self.x_err)
        self.y_err_canonical = self.unit_system.convert_to_canonical("y_err", self.y_err)

    def chi2(self, orbit: Orbit,jitter_x=None,jitter_y=None):
        x, y = orbit.astrometry(self.t_canonical, self.plxf_x, self.plxf_y, system=self.system)
    
        if jitter_x is not None:
            total_x_err = np.sqrt(self.x_err_canonical**2 + jitter_x**2)
        else:
            total_x_err = self.x_err_canonical
        if jitter_y is not None:
            total_y_err = np.sqrt(self.y_err_canonical**2 + jitter_y**2)
        else:
            total_y_err = self.y_err_canonical
        chi2_x = np.sum(((self.x_canonical - x) / total_x_err) ** 2)
        chi2_y = np.sum(((self.y_canonical - y) / total_y_err) ** 2)
        return chi2_x + chi2_y


    def log_likelihood(self, orbit: Orbit, offset_names=None):
            flux_name = flux_parameter(self.band)
            flux_ratio = (
                orbit.params[flux_name]
                if flux_name is not None and flux_name in orbit.params
                else None
            )
            x, y = orbit.astrometry(
                self.t_canonical,
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
            residuals_x = self.x_canonical - x
            residuals_y = self.y_canonical - y
    
            jitter_name = (
                'astro_jitter'
                if self.instrument is None
                else f"astro_{self.instrument}_jitter"
            )
    
            jitter = orbit.params.get(jitter_name)
    
            ln_like_x = _gaussian_log_likelihood(residuals_x,self.x_err,jitter)
            ln_like_y = _gaussian_log_likelihood(residuals_y,self.y_err,jitter)
                
            return ln_like_x + ln_like_y

    def _astrometry(self, orbit: Orbit):
        return self.x_canonical, self.y_canonical
    

class RadialVelocityData(SystemData):
    def __init__(self, t, rv, rv_err, units: Optional[Mapping[str, Any]] = None, system=None,instrument=None):
        if units is None:
            if is_quantity(t) and is_quantity(rv) and is_quantity(rv_err):
                units = {
                    "t": t.unit,
                    "rv": rv.unit,
                    "rv_err": rv_err.unit,
                }
                t = t.value
                rv = rv.value
                rv_err = rv_err.value
        super().__init__(system, units)
        required_units = {"t", "rv", "rv_err"}
        missing_units = required_units - self.unit_system.units.keys()
        if missing_units:
            raise ValueError(f"Missing units for radial velocity data: {sorted(missing_units)}")
        if self.unit_system.get_parameter_dimension("t") != "time":
            raise ValueError("Unit for 't' must represent time.")
        if self.unit_system.get_parameter_dimension("rv") != "velocity":
            raise ValueError("Unit for 'rv' must represent velocity.")
        if self.unit_system.get_parameter_dimension("rv_err") != "velocity":
            raise ValueError("Unit for 'rv_err' must represent velocity.")

        self.t = np.atleast_1d(t)
        self.rv = np.atleast_1d(rv)
        self.rv_err = np.atleast_1d(rv_err)
        if self.rv.shape != self.t.shape:
            raise ValueError("rv must have the same shape as t")
        if self.rv_err.shape != self.rv.shape:
            self.rv_err = np.broadcast_to(self.rv_err, self.rv.shape)

        self.t_canonical = self.unit_system.convert_to_canonical("t", self.t)
        self.rv_canonical = self.unit_system.convert_to_canonical("rv", self.rv)
        self.rv_err_canonical = self.unit_system.convert_to_canonical("rv_err", self.rv_err) 
        
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

        jitter = orbit.params.get(jitter_name)
        ln_like_rv = _gaussian_log_likelihood(residuals_rv,self.rv_err,jitter)
        
        
        return ln_like_rv

    def has_astrometry(self) -> bool:
        return False

    def has_radial_velocity(self) -> bool:
        return True

    def _radial_velocity(self, orbit: Orbit):
        return self.rv

    def _astrometry(self, orbit: Orbit):
        raise NotImplementedError
    
    