from astropy import units as u
from typing import Any, Dict, Mapping, Optional
import numpy as np

#Define Canonical Units
CanonicalUnits = {
    'time': u.yr,
    'length':u.AU,
    'mass': u.M_sun,
    'angle': u.rad,
    'velocity':u.AU / u.yr,
    'distance': u.pc,
    "angular_velocity": u.rad / u.yr,
    "linear_velocity": u.AU / u.yr,
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


def unit_from_string(value: Any) -> u.Unit:
    """
    Convert a string representation of a unit to an astropy Unit object.
    """
    if value is None:
        raise ValueError("Unit value cannot be None")
    
    if isinstance(value, u.UnitBase):
        return value

    if isinstance(value, str):
        # Check for aliases
        for alias in [angle_aliases, linear_aliases, time_aliases, mass_aliases, velocity_aliases]:
            if value in alias:
                return alias[value]  
        
        try:
            return u.Unit(value)
        except ValueError as e:
            raise ValueError(f"Invalid unit string: {value}") from e

    raise ValueError(f"Invalid unit value: {value!r}")

def parse_units(unit_dict: Optional[Mapping[str, Any]]) -> Dict[str, u.Unit]:
    '''
    Parse and return the canonical units as a dictionary of astropy Unit objects.
    '''
    if unit_dict is None:
        raise ValueError("Units dictionary cannot be None")

    parsed_units = {}
    for key, value in unit_dict.items():
        parsed_units[key] = unit_from_string(value)

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

_PARAMETER_DIMENSIONS = {
    "P": "time",
    "Tp": "time",
    "Tepoch": "time",
    "t": "time",
    "n": "angular_velocity",
    "i": "angle",
    "cosi": "dimensionless",
    "sini": "dimensionless",
    "omega": "angle",
    "omega1": "angle",
    "omega2": "angle",
    "Omega": "angle",
    "Omega1": "angle",
    "Omega2": "angle",
    "piomega": "angle",
    "piomega1": "angle",
    "piomega2": "angle",
    "parallax": "angle",
    "M0": "angle",
    "E0": "angle",
    "nu0": "angle",
    "L0": "angle",
    "l0": "angle",
    "l01": "angle",
    "l02": "angle",
    "u0": "angle",
    "u01": "angle",
    "u02": "angle",
    "uM0": "angle",
    "uM01": "angle",
    "uM02": "angle",
    "M1": "mass",
    "M2": "mass",
    "Mtot": "mass",
    "distance": "distance",
    "K": "velocity",
    "K1": "velocity",
    "K2": "velocity",
    "c": "velocity",
    "c1": "velocity",
    "c2": "velocity",
    "h": "velocity",
    "h1": "velocity",
    "h2": "velocity",
    "rv": "velocity",
    "gamma": "velocity",
    "e": "dimensionless",
    "q": "dimensionless",
    "parallax_factor": "dimensionless",
}

_PROJECTED_POSITION_PARAMETERS = {
    "a", "b", "p", "r_a", "r_p",
    "a1", "b1", "p1", "r_a1", "r_p1",
    "a2", "b2", "p2", "r_a2", "r_p2",
    "A", "B", "F", "G",
    "A1", "B1", "F1", "G1",
    "A2", "B2", "F2", "G2",
    "dalpha", "ddelta",
}

_PROPER_MOTION_PARAMETERS = {"mu_alpha", "mu_delta"}



def get_unit_type(unit):
    """Determine the physical dimension represented by an Astropy unit."""
    unit = unit_from_string(unit)

    # Check angle before dimensionless because radians are dimensionless in SI.
    if unit.is_equivalent(u.rad):
        return "angle"
    if unit.is_equivalent(u.m / u.s):
        return "velocity"
    if unit.is_equivalent(u.rad / u.s):
        return "angular_velocity"
    if unit.is_equivalent(u.s):
        return "time"
    if unit in {u.pc, u.kpc, u.Mpc, u.Gpc, u.lyr}:
        return "distance"
    if unit.is_equivalent(u.m):
        return "length"
    if unit.is_equivalent(u.kg):
        return "mass"
    if unit.is_equivalent(u.dimensionless_unscaled):
        return "dimensionless"

    raise ValueError(f"Unknown unit type for unit: {unit}")


class UnitSystem:
    """Cached units and conversions for one fitting problem."""

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
        self.dimensionality = {
            name: self._infer_dimension(name, unit)
            for name, unit in self.units.items()
        }
        self.canonical_units = {
            name: CanonicalUnits[dimension]
            for name, dimension in self.dimensionality.items()
        }
        self.conversion_factors = {
            name: unit.to(self.canonical_units[name])
            for name, unit in self.units.items()
        }

    def _infer_dimension(self, name, unit):
        if name in _PROJECTED_POSITION_PARAMETERS:
            if self.astrometry_dimension is not None:
                return self.astrometry_dimension
            return get_unit_type(unit)

        if name in _PROPER_MOTION_PARAMETERS:
            if self.astrometry_dimension == "length":
                return "linear_velocity"
            return "angular_velocity"

        return _PARAMETER_DIMENSIONS.get(name, get_unit_type(unit))

    def get_parameter_dimension(self, param_name):
        if param_name in self.dimensionality:
            return self.dimensionality[param_name]
        if param_name in _PROJECTED_POSITION_PARAMETERS:
            return self.astrometry_dimension or "length"
        if param_name in _PROPER_MOTION_PARAMETERS:
            return (
                "linear_velocity"
                if self.astrometry_dimension == "length"
                else "angular_velocity"
            )
        if param_name in _PARAMETER_DIMENSIONS:
            return _PARAMETER_DIMENSIONS[param_name]
        raise KeyError(f"No dimension known for '{param_name}'.")

    def get_parameter_unit(self, param_name):
        """Return the user-facing unit associated with a model parameter."""
        if param_name in self.units:
            return self.units[param_name]

        dimension = self.get_parameter_dimension(param_name)
        if dimension == "time":
            return self.units["t"]
        if dimension in {"angle", "length"}:
            if self.astrometry_dimension == dimension and "x" in self.units:
                return self.units["x"]
            return CanonicalUnits[dimension]
        if dimension in {"angular_velocity", "linear_velocity"}:
            if "x" not in self.units or "t" not in self.units:
                return CanonicalUnits[dimension]
            return self.units["x"] / self.units["t"]
        if dimension == "velocity" and "rv" in self.units:
            return self.units["rv"]
        return CanonicalUnits[dimension]

    def convert_to_canonical(self, param_name, value):
        if param_name not in self.conversion_factors:
            raise KeyError(f"No input unit was supplied for '{param_name}'.")
        return np.asarray(value) * self.conversion_factors[param_name]

    def convert_from_canonical(self, param_name, value):
        if param_name not in self.conversion_factors:
            raise KeyError(f"No input unit was supplied for '{param_name}'.")
        return np.asarray(value) / self.conversion_factors[param_name]

    def convert_prior_to_canonical(self, param_name, value):
        unit = self.get_parameter_unit(param_name)
        return np.asarray(value) * unit.to(CanonicalUnits[self.get_parameter_dimension(param_name)])

    def convert_prior_from_canonical(self, param_name, value):
        unit = self.get_parameter_unit(param_name)
        return np.asarray(value) / unit.to(CanonicalUnits[self.get_parameter_dimension(param_name)])