from astropy import units as u
from typing import Any, Dict, Mapping, Optional
import numpy as np
from periapsis.params import params as par
import warnings

#Define Canonical Units
CanonicalUnits = {
    'time': u.yr,
    'length':u.AU,
    'mass': u.M_sun,
    'angle': u.rad,
    'velocity':u.AU / u.yr,
    "angular_velocity": u.rad / u.yr,
    "acceleration": u.AU / u.yr**2,
    "angular_acceleration": u.rad / u.yr**2,
    "gravitational_parameter": u.AU**3 / u.yr**2,
    "dimensionless": u.dimensionless_unscaled,
}

#Aliases for units
angle_aliases = {
    'rad': u.rad,
    'radian': u.rad,
    'deg': u.deg,
    'degree': u.deg,
    'arcsec': u.arcsec,
    'arcsecond': u.arcsec,
    'arcmin': u.arcmin,
    'arcminute': u.arcmin,
    'as': u.arcsec,
    'mas': u.mas,
    'milliarcsecond': u.mas,
}

linear_aliases = {
    'au': u.AU,
    "AU": u.AU,
    'pc': u.pc,
    'parsec': u.pc,
    'ly': u.lyr,
    'lightyear': u.lyr,
    'm': u.m,
    'meter': u.m,
    'km': u.km,
    'kilometer': u.km,
    'cm': u.cm,
    'centimeter': u.cm,
}

time_aliases = {
    's': u.s,
    'sec': u.s,
    'second': u.s,
    'min': u.min,
    'minute': u.min,
    'h': u.h,
    'hr': u.h,
    'hour': u.h,
    'd': u.d,
    'day': u.d,
    'yr': u.yr,
    'year': u.yr,
    'yrs': u.yr,
}

mass_aliases = {
    'kg': u.kg,
    'msun': u.M_sun,
    'Msun': u.M_sun,
    'm_sun': u.M_sun,
    'solar_mass': u.M_sun,
    'g': u.g,
    'mjup': u.M_jup,
    'Mjup': u.M_jup,
    'm_jup': u.M_jup,
    'mearth': u.M_earth,
    'Mearth': u.M_earth,
    'm_earth': u.M_earth,
}

velocity_aliases = {
    'm/s': u.m / u.s,
    'cm/s': u.cm / u.s,
    'km/s': u.km / u.s,
    'au/yr': u.AU / u.yr,
    'AU/yr': u.AU / u.yr,
    'pc/yr': u.pc / u.yr,
}

aliases = {**angle_aliases, **linear_aliases, **time_aliases, **mass_aliases, **velocity_aliases}

def normalize_unit(value: Any) -> u.Unit:
    """
    Convert a string representation of a unit to an astropy Unit object.
    """
    if value is None:
        raise ValueError("Unit value cannot be None")
    
    if isinstance(value, u.UnitBase):
        return value

    if isinstance(value, str):
        # Check for aliases
        if value in aliases:
            return aliases[value]
        try:
            return u.Unit(value)
        except ValueError as e:
            raise ValueError(f"Invalid unit string: {value}") from e

    raise ValueError(f"Invalid unit value: {value!r}")


def is_quantity(value: Any) -> bool:
    """
    Check if the value is an astropy Quantity.
    """
    return hasattr(value, 'unit') and hasattr(value, 'value')


def parse_units(unit_dict: Optional[Mapping[str, Any]]) -> Dict[str, u.Unit]:
    '''
    Parse and return the canonical units as a dictionary of astropy Unit objects.
    '''
    if unit_dict is None:
        raise ValueError("Units dictionary cannot be None")

    parsed_units = {}
    for key, value in unit_dict.items():
        parsed_units[key] = normalize_unit(value)

    return parsed_units


def validate_units(units:Dict[str,u.Unit]) -> None:

    sim_groups = {
        'astrometry_position_group': ['x', 'y', 'x_err', 'y_err'],
        'x_group': ['x','x_err'],
        'y_group': ['y','y_err'],
        'mu_group':['mu_x','mu_y'],
        'rv_group':['rv','rv_err']
    }

    for group, quantities in sim_groups.items():
        present_units = {q: units[q] for q in quantities if q in units}
        if not present_units:
            continue
        ref_unit = list(present_units.values())[0]
        for qty, unit_obj in present_units.items():
            try:
                ref_unit.to(unit_obj)
            except u.UnitConversionError:
                raise ValueError(
                    f"Incompatible units in group '{group}': "
                    f"'{qty}' has unit '{unit_obj}' but other quantities have '{ref_unit}'. "
                    f"All of {list(present_units.keys())} must be dimensionally compatible."
                )

    for name, unit in units.items():
        if name in _PARAMETER_DIMENSIONS:
            expected_dimension = _PARAMETER_DIMENSIONS[name]
            actual_dimension = get_unit_type(unit)
            if expected_dimension != actual_dimension:
                if name in par.ang_parameters and (
                    (expected_dimension == "length" and actual_dimension == "angle") or
                    (expected_dimension == "velocity" and actual_dimension == "angular_velocity") or
                    (expected_dimension == "acceleration" and actual_dimension == "angular_acceleration")
                ):
                    raise ValueError(
                        f"Unit for '{name}' has dimension '{actual_dimension}', "
                        f"but expected '{expected_dimension}'. "
                        f"You probably want to use '{name}_ang', which is the version of this parameter in angular rather than physical units."
                    )
                raise ValueError(
                    f"Unit for '{name}' has dimension '{actual_dimension}', "
                    f"but expected '{expected_dimension}'."
                )


_DATA_DIMENSIONS = {
    "t": "time",
    "parallax_factor": "dimensionless",
}

_PARAMETER_DIMENSIONS = {
    par.a: "length",
    par.a1: "length",
    par.a1sini: "length",
    par.a2: "length",
    par.a2sini: "length",
    par.A: "length",
    par.A1: "length",
    par.A2: "length",
    par.b: "length",
    par.b1: "length",
    par.b2: "length",
    par.B: "length",
    par.B1: "length",
    par.B2: "length",
    par.c: "velocity",
    par.c1: "velocity",
    par.c2: "velocity",
    par.C: "length",
    par.C1: "length",
    par.C2: "length",
    par.cosi: "dimensionless",
    par.dalpha: "length",
    par.ddelta: "length",
    par.dgamma: "acceleration",
    par.distance: "length",
    par.dmu_alpha: "acceleration",
    par.dmu_delta: "acceleration",
    par.e: "dimensionless",
    par.E0: "angle",
    par.f1: "mass",
    par.f2: "mass",
    par.F: "length",
    par.F1: "length",
    par.F2: "length",
    par.gamma: "velocity",
    par.G: "length",
    par.G1: "length",
    par.G2: "length",
    par.h: "velocity",
    par.h1: "velocity",
    par.h2: "velocity",
    par.H: "length",
    par.H1: "length",
    par.H2: "length",
    par.i: "angle",
    par.K: "velocity",
    par.K1: "velocity",
    par.K2: "velocity",
    par.l0: "angle",
    par.l01: "angle",
    par.l02: "angle",
    par.L0: "angle",
    par.M1: "mass",
    par.M2: "mass",
    par.minM1: "mass",
    par.minM2: "mass",
    par.Mtot: "mass",
    par.M0: "angle",
    par.Msini: "mass",
    par.M1sini: "mass",
    par.M2sini: "mass",
    par.mu: "gravitational_parameter",
    par.mu_alpha: "velocity",
    par.mu_delta: "velocity",
    par.n: "angular_velocity",
    par.nu0: "angle",
    par.Omega: "angle",
    par.Omega1: "angle",
    par.Omega2: "angle",
    par.omega: "angle",
    par.omega1: "angle",
    par.omega2: "angle",
    par.p: "length",
    par.p1: "length",
    par.p2: "length",
    par.P: "time",
    par.parallax: "angle",
    par.piomega: "angle",
    par.piomega1: "angle",
    par.piomega2: "angle",
    par.q: "dimensionless",
    par.r_a: "length",
    par.r_a1: "length",
    par.r_a2: "length",
    par.r_p: "length",
    par.r_p1: "length",
    par.r_p2: "length",
    par.sini: "dimensionless",
    par.t0: "dimensionless",
    par.Tp: "time",
    par.Tepoch: "time",
    par.u0: "angle",
    par.u01: "angle",
    par.u02: "angle",
    par.uM0: "angle",
    par.uM01: "angle",
    par.uM02: "angle",
}
for param in par.ang_parameters:
    current_dim = _PARAMETER_DIMENSIONS[param]
    if current_dim == "length":
        _PARAMETER_DIMENSIONS[par.ang(param)] = "angle"
    elif current_dim == "velocity":
        _PARAMETER_DIMENSIONS[par.ang(param)] = "angular_velocity"
    elif current_dim == "acceleration":
        _PARAMETER_DIMENSIONS[par.ang(param)] = "angular_acceleration"
for param in list(_PARAMETER_DIMENSIONS.keys()):
    _PARAMETER_DIMENSIONS[par.log(param)] = _PARAMETER_DIMENSIONS[param]


def get_default_dimension(param_name):
    """
    Get the default dimension for a model parameter.
    """
    if param_name not in _PARAMETER_DIMENSIONS:
        raise KeyError(f"No default dimension defined for parameter '{param_name}'.")
    return _PARAMETER_DIMENSIONS[param_name]

def get_unit_type(unit):
    """Determine the physical dimension represented by an Astropy unit."""
    unit = normalize_unit(unit)

    if unit.is_equivalent(u.rad): # Angle check before dimensionless, since radians are also dimensionless
        return "angle"
    if unit.is_equivalent(u.m / u.s):
        return "velocity"
    if unit.is_equivalent(u.rad / u.s):
        return "angular_velocity"
    if unit.is_equivalent(u.s):
        return "time"
    if unit.is_equivalent(u.m):
        return "length"
    if unit.is_equivalent(u.kg):
        return "mass"
    if unit.is_equivalent(u.AU**3 / u.yr**2):
        return "gravitational_parameter"
    if unit.is_equivalent(u.m / u.s**2):
        return "acceleration"
    if unit.is_equivalent(u.rad / u.s**2):
        return "angular_acceleration"
    if unit.is_equivalent(u.dimensionless_unscaled):
        return "dimensionless"

    raise ValueError(f"Unknown unit type for unit: {unit}")


def get_canonical_unit_scale(unit):
    """Return the scale factor to convert a unit to its canonical unit."""
    unit = normalize_unit(unit)
    unit_type = get_unit_type(unit)
    canonical_unit = CanonicalUnits[unit_type]
    return unit.to(canonical_unit)


class UnitSystem:
    """Units and conversions for data and model parameters"""

    def __init__(self, units, astrometry_dimension=None):
        self.units = parse_units(units)
        validate_units(self.units)

        if astrometry_dimension is None and "x" in self.units:
            astrometry_dimension = get_unit_type(self.units["x"])

        if astrometry_dimension is not None and astrometry_dimension not in {
            "angle", "length"
        }:
            raise ValueError(
                "astrometry_dimension must be either 'angle' or 'length'."
            )

        self.astrometry_dimension = astrometry_dimension
        self._dimensionality = {
            name: self._infer_dimension(name, unit)
            for name, unit in self.units.items()
        }
        self._canonical_units = {
            name: CanonicalUnits[dimension]
            for name, dimension in self._dimensionality.items()
        }
        self._conversion_factors = {
            name: unit.to(self._canonical_units[name])
            for name, unit in self.units.items()
        }

    def __repr__(self):
        return f"UnitSystem(units={self.units}, astrometry_dimension={self.astrometry_dimension})"

    def merge(self, other: UnitSystem) -> UnitSystem:
        """Merge another UnitSystem into this one, returning a new UnitSystem."""
        if self.astrometry_dimension is not None and other.astrometry_dimension is not None and self.astrometry_dimension != other.astrometry_dimension:
            raise ValueError(
                "Cannot merge UnitSystems with conflicting astrometry dimensions."
            )

        # Emit warning if there are conflicting units for the same parameter
        for name in set(self.units.keys()).intersection(other.units.keys()):
            if self.units[name] != other.units[name]:
                warnings.warn(
                    f"Conflicting units for parameter '{name}': "
                    f"{self.units[name]} vs {other.units[name]}. "
                    f"Using {self.units[name]} from the first UnitSystem."
                )
        merged_astrometry_dimension = self.astrometry_dimension if self.astrometry_dimension is not None else other.astrometry_dimension
        merged_units = {**other.units, **self.units}
        return UnitSystem(merged_units, astrometry_dimension=merged_astrometry_dimension)

    def _infer_dimension(self, name, unit):
        # if name in par.ang_parameters: FIXME
        #     if self.astrometry_dimension == "length":
        #         return _PARAMETER_DIMENSIONS[name]
        #     else:
        #         return _PARAMETER_DIMENSIONS[par.ang(name)]

        return _DATA_DIMENSIONS.get(name, get_unit_type(unit))

    def get_parameter_dimension(self, param_name):
        if param_name in self._dimensionality:
            return self._dimensionality[param_name]
        if param_name.startswith("log"):
            return "dimensionless"
        if "rv" in param_name and "jitter" in param_name:
            return "velocity"
        if "astro" in param_name and "jitter" in param_name:
            if self.astrometry_dimension == "length":
                return "length"
            elif self.astrometry_dimension == "angle":
                return "angle"
        # if param_name in par.ang_parameters:
        #     return self.astrometry_dimension or "length"
        if param_name in _DATA_DIMENSIONS:
            return _DATA_DIMENSIONS[param_name]
        if param_name in _PARAMETER_DIMENSIONS:
            return _PARAMETER_DIMENSIONS[param_name]
        raise KeyError(f"No dimension known for '{param_name}'.")

    def get_parameter_unit(self, param_name) -> u.UnitBase:
        """Return the user-facing unit associated with a model parameter."""
        if param_name in self.units:
            return self.units[param_name]

        dimension = self.get_parameter_dimension(param_name)
        if dimension == "time":
            return self.units["t"]
        if dimension == "angle" and self.astrometry_dimension == "angle" and "x" in self.units and "_ang" in param_name:
            return self.units["x"]
        if dimension == "length" and self.astrometry_dimension == "length" and "x" in self.units:
            return self.units["x"]
        if dimension == "velocity":
            if "rv" in self.units:
                return self.units["rv"]
            elif "x" in self.units and "t" in self.units and self.astrometry_dimension == "length":
                return self.units["x"] / self.units["t"]
        if dimension == "angular_velocity" and "x" in self.units and "t" in self.units and self.astrometry_dimension == "angle":
            return self.units["x"] / self.units["t"]
        if dimension == "acceleration":
            if "rv" in self.units and "t" in self.units:
                return self.units["rv"] / self.units["t"]
            if "x" in self.units and "t" in self.units and self.astrometry_dimension == "length":
                return self.units["x"] / (self.units["t"] ** 2)
        if dimension == "angular_acceleration" and "x" in self.units and "t" in self.units and self.astrometry_dimension == "angle":
            return self.units["x"] / (self.units["t"] ** 2)
        if dimension == "gravitational_parameter":
            if "x" in self.units and "t" in self.units and self.astrometry_dimension == "length":
                return self.units["x"] ** 3 / (self.units["t"] ** 2)
        return CanonicalUnits[dimension]

    def get_parameter_canonical_unit(self, param_name) -> u.UnitBase:
        """Return the canonical unit associated with a model parameter."""
        dimension = self.get_parameter_dimension(param_name)
        return CanonicalUnits[dimension]

    def convert_to_canonical(self, param_name, value):
        if param_name not in self._conversion_factors:
            self._conversion_factors[param_name] = self.get_parameter_unit(param_name).to(CanonicalUnits[self.get_parameter_dimension(param_name)])
        return value * self._conversion_factors[param_name]

    def convert_from_canonical(self, param_name, value, unit=None):
        # Use provided unit if given, otherwise use the default unit from the unit system
        if unit is not None:
            conversion_factor = normalize_unit(unit).to(CanonicalUnits[self.get_parameter_dimension(param_name)])
            return value / conversion_factor
        if param_name not in self._conversion_factors:
            self._conversion_factors[param_name] = self.get_parameter_unit(param_name).to(CanonicalUnits[self.get_parameter_dimension(param_name)])
        return value / self._conversion_factors[param_name]