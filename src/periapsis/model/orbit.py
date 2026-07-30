from periapsis.params import covered_parameters, build_transform_functions
from periapsis.utils.solvers import solve_kepler
import numpy as np
from types import MappingProxyType

_astrometry_param_names = {
    "": {'P', 'e', 'Tp', 'Tepoch', 'A', 'B', 'F', 'G', 'dx', 'dy', 'dpmra', 'dpmdec'}, # Relative astrometry
    "1": {'P', 'e', 'Tp', 'Tepoch', 'A1', 'B1', 'F1', 'G1', 'dx', 'dy', 'dpmra', 'dpmdec'}, # Primary astrometry
    "2": {'P', 'e', 'Tp', 'Tepoch', 'A2', 'B2', 'F2', 'G2', 'dx', 'dy', 'dpmra', 'dpmdec'}, # Secondary astrometry
}
_rv_param_names = {
    "": {'K', 'e', 'Tp', 'Tepoch', 'P', 'omega', 'systemic_velocity'}, # Relative RVs
    "1": {'K1', 'e', 'Tp', 'Tepoch', 'P', 'omega1', 'systemic_velocity'}, # Primary RVs
    "2": {'K2', 'e', 'Tp', 'Tepoch', 'P', 'omega2', 'systemic_velocity'} # Secondary RVs
}
_xyz_param_names = {
    "": {"P", "e", "a", "Tp", "Tepoch", "omega", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'}, # Relative 3D position
    "1": {"P", "e", "a1", "Tp", "Tepoch", "omega1", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'}, # Primary 3D position
    "2": {"P", "e", "a2", "Tp", "Tepoch", "omega2", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'} # Secondary 3D position
}
_vxyz_param_names = {
    "": {"P", "e", "a", "Tp", "Tepoch", "omega", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'}, # Relative 3D velocity
    "1": {"P", "e", "a1", "Tp", "Tepoch", "omega1", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'}, # Primary 3D velocity
    "2": {"P", "e", "a2", "Tp", "Tepoch", "omega2", "Omega", "i", 'dx', 'dy', 'distance', 'dpmra', 'dpmdec', 'systemic_velocity'} # Secondary 3D velocity
}

_gaia_param_names = {
    "": {"P", "e", "Tp",  "A", "B", "F", "G", "parallax", "dalpha", "ddelta", "mu_alpha", "mu_delta"}, # Relative Gaia astrometry
    "1": {"P", "e", "Tp", "A1", "B1", "F1", "G1", "parallax", "dalpha", "ddelta", "mu_alpha", "mu_delta"}, # Primary Gaia astrometry
    "2": {"P", "e", "Tp", "A2", "B2", "F2", "G2", "parallax", "dalpha", "ddelta", "mu_alpha", "mu_delta"} # Secondary Gaia astrometry
}

class Orbit():
    """Class representing an orbit defined by a set of orbital elements."""
    def __init__(self, *, velocity_ratio=None, **kwparams):
        """
        A Keplerian orbit is defined by orbital parameters. 7 independent parameters are required to fully specify a 3D orbit, but less may be needed in some contexts.

        velocity_ratio optionally gives the conversion factor to convert from units of (time/distance) to the preferred units for velocity. If none is specified, a warning will be raised whenever a method that computes velocity is called.
        """
        self._params = kwparams
        if 'Tepoch' not in self._params:
            self._params['Tepoch'] = 0.0
        self._covered_params = covered_parameters(set(self.params.keys()))
        self.astrometry_param_sets = {"": None, "1": None, "2": None}
        self.rv_param_sets = {"": None, "1": None, "2": None}
        self.xyz_param_sets = {"": None, "1": None, "2": None}
        self.vxyz_param_sets = {"": None, "1": None, "2": None}
        self._derived_params = dict(self.params)
        self._velocity_ratio = velocity_ratio
        self._check_params()

    @property
    def params(self):
        return MappingProxyType(self._params)

    @property
    def covered_params(self):
        return frozenset(self._covered_params)

    @property
    def derived_params(self):
        return MappingProxyType(self._derived_params)

    @property
    def velocity_ratio(self):
        return self._velocity_ratio

    def __getitem__(self, key):
        if key in self.params:
            return self.params[key]
        elif key in self.derived_params:
            return self.derived_params[key]
        elif key in self.covered_params:
            transform = build_transform_functions(self.params, [key])
            self._derived_params[key] = transform(**self.params)[key]
            return self._derived_params[key]
        else:
            raise KeyError(f"Parameter '{key}' is not defined by the provided parameters.")


    def __setitem__(self, key, value):
        raise TypeError("Orbit parameters are read-only. To modify parameters, create a new Orbit instance with the desired parameters.")

    def __contains__(self, key):
        return key in self.params or key in self.derived_params or key in self.covered_params

    def _check_params(self):
        if len(self.params) <= 1:
            raise ValueError("Insufficient orbital parameters provided.")

        # TODO more efficient check for whether any of the required parameter sets are covered by the provided parameters.
        #     missing_astrometry_params = {"relative" if sys == "" else sys: _astrometry_param_names[sys] - self.covered_params for sys in ("", "1", "2")}
        #     missing_rv_params = {"relative" if sys == "" else sys: _rv_param_names[sys] - self.covered_params for sys in ("", "1", "2")}
        #     missing_xyz_params = {"relative" if sys == "" else sys: _xyz_param_names[sys] - self.covered_params for sys in ("", "1", "2")}
        #     missing_vxyz_params = {"relative" if sys == "" else sys: _vxyz_param_names[sys] - self.covered_params for sys in ("", "1", "2")}
        #     raise ValueError("Insufficient orbital parameters provided to define any orbit. Missing parameters:\n"
        #                      f"  - Astrometry: {missing_astrometry_params}\n"
        #                      f"  - RV: {missing_rv_params}\n"
        #                      f"  - XYZ: {missing_xyz_params}\n"
        #                      f"  - VXYZ: {missing_vxyz_params}")
        # TODO add more validation to check for valid parameter values (e.g. e should be between 0 and 1)

    def _ensure_derived_params(self, required_params):
        missing_params = required_params - frozenset(self.derived_params.keys())
        if missing_params:
            transform = build_transform_functions(self.params, sorted(missing_params))
            self._derived_params.update(transform(**self.params))

    def astrometry(self, t, system=None):
        """
        Computes the astrometric position of the orbit at time(s) t. 
        
        Returns the position as a tuple (x, y) in the plane of the sky where x and y are floats or an array of floats matching the shape of t.
        """
        if system is None or str(system) not in {'1', '2', 'relative'}:
            raise ValueError(f"`system` must be provided for astrometry. It can be either '1', '2', or 'relative'.")
        system = "" if system == "relative" else str(system)

        self._ensure_derived_params(_astrometry_param_names[system])

        ref_epoch = self.derived_params['Tepoch']
        
        dt = np.asarray(t) - ref_epoch

        M = 2 * np.pi / self.derived_params['P'] * (dt - (self.derived_params['Tp']-ref_epoch))
        E = solve_kepler(M, self.derived_params['e'])
        X = (np.cos(E) - self.derived_params['e'])
        Y = (np.sqrt(1 - self.derived_params['e']**2) * np.sin(E))
        alpha = self.derived_params[f'A{system}'] * X + self.derived_params[f'F{system}'] * Y
        delta = self.derived_params[f'B{system}'] * X + self.derived_params[f'G{system}'] * Y

        alpha = alpha + self.derived_params['dx'] + self.derived_params['dpmra'] * dt
        delta = delta + self.derived_params['dy'] + self.derived_params['dpmdec'] * dt
        return alpha, delta

    def gaia_astrometry(self, t,spsi,cpsi,par_factor, system=None):
        """
        Computes astrometric position of orbit at time(s) t for Gaia data
        """
        if system is None or str(system) not in {'1', '2', 'relative'}:
            raise ValueError(f"`system` must be provided for astrometry. It can be either '1', '2', or 'relative'.")
        system = "" if system == "relative" else str(system)

        self._ensure_derived_params(_gaia_param_names[system])

        t = np.asarray(t)
        M = 2 * np.pi / self.derived_params['P'] * (t - self.derived_params['Tp'])
        E = solve_kepler(M, self.derived_params['e'])
        X = (np.cos(E) - self.derived_params['e'])
        Y = (np.sqrt(1 - self.derived_params['e']**2) * np.sin(E))

        wss = ((self.derived_params['dalpha']+self.derived_params['mu_alpha']*t)*spsi
                + (self.derived_params['ddelta']+self.derived_params['mu_delta']*t)*cpsi
                + self.derived_params['parallax']*par_factor)

        wk = ((self.derived_params[f'B{system}']*X + self.derived_params[f'G{system}']*Y)*spsi
              + (self.derived_params[f'A{system}']*X + self.derived_params[f'F{system}']*Y)*cpsi)

        return wss + wk


    def rv(self, t, system=None):
        """
        Computes the radial velocity of the orbit at time(s) t. 
        
        Returns the radial velocity as a float or array of floats matching the shape of t.
        """
        if system is None or str(system) not in {'1', '2', 'relative'}:
            raise ValueError(f"`system` must be provided for radial velocity. It can be either '1', '2', or 'relative'.")
        system = "" if system == "relative" else str(system)

        self._ensure_derived_params(_rv_param_names[system])
        
        ref_epoch = self.derived_params['Tepoch']
        dt = np.asarray(t) - ref_epoch

        M = 2 * np.pi / self.derived_params['P'] * (dt - (self.derived_params['Tp']-ref_epoch))
        E = solve_kepler(M, self.derived_params['e'])
        true_anomaly = 2 * np.arctan2(np.sqrt(1 + self.derived_params['e']) * np.sin(E / 2), np.sqrt(1 - self.derived_params['e']) * np.cos(E / 2))
        rv = self.derived_params[f'K{system}'] * (np.cos(true_anomaly + self.derived_params[f'omega{system}']) + self.derived_params[f'e'] * np.cos(self.derived_params[f'omega{system}']))
        rv *= self.velocity_ratio if self.velocity_ratio is not None else 1.0
        # if self.velocity_ratio is None:
            # print("Warning: velocity_ratio is not set. Radial velocity will be returned in units of (time/distance).")
        if system != "":
            rv += self.derived_params['systemic_velocity']
        return rv
    
    def xyz(self, t, system=None):
        """
        Computes the 3D position of the orbit at time(s) t, where the x-y plane is the plane of the sky. 
        
        Returns the position as a tuple (x, y, z) where x, y, and z are floats or arrays of floats matching the shape of t.
        """

        if system is None or str(system) not in {'1', '2', 'relative'}:
            raise ValueError(f"`system` must be provided for 3D position. It can be either '1', '2', or 'relative'.")
        system = "" if system == "relative" else str(system)

        self._ensure_derived_params(_xyz_param_names[system])

        ref_epoch = self.derived_params['Tepoch']

        raise NotImplementedError("The xyz method is not implemented yet.")
        x=float('nan')
        y=float('nan')
        z=float('nan')
        return x, y, z
    
    def vxyz(self, t, system=None):
        """
        Computes the 3D velocity of the orbit at time(s) t, where the x-y plane is the plane of the sky. 
        
        Returns the velocity as a tuple (vx, vy, vz) where vx, vy, and vz are floats or arrays of floats matching the shape of t.
        """

        if system is None or str(system) not in {'1', '2', 'relative'}:
            raise ValueError(f"`system` must be provided for 3D velocity. It can be either '1', '2', or 'relative'.")
        system = "" if system == "relative" else str(system)

        self._ensure_derived_params(_vxyz_param_names[system])
        # raise ValueError(f"Insufficient orbital parameters provided to compute 3D velocity. Unable to calculate parameters for system {system}: {_vxyz_param_names[system] - self.covered_params}")
        
        ref_epoch = self.derived_params['Tepoch']

        if self.velocity_ratio is None:
            print("Warning: velocity_ratio is not set. Velocities will be returned in units of (time/distance).")

        raise NotImplementedError("The vxyz method is not implemented yet.")
        vx=float('nan')
        vy=float('nan')
        vz=float('nan')
        return vx, vy, vz
        
