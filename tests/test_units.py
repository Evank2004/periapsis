import numpy as np
import pytest
from astropy import units as u

from periapsis.params.units import (
    CanonicalUnits,
    UnitSystem,
    get_unit_type,
    parse_units,
    normalize_unit,
    validate_units,
    get_default_dimension,
)

from periapsis.params.params import all_parameters, ang_parameters


def test_unit_aliases_parse_to_astropy_units():
    assert normalize_unit("mas") == u.mas
    assert normalize_unit("AU/yr") == u.AU / u.yr
    assert normalize_unit("Msun") == u.M_sun


def test_parse_units_returns_astropy_units():
    parsed = parse_units({"x": "mas", "t": "yr"})

    assert parsed == {"x": u.mas, "t": u.yr}


def test_validate_units_requires_compatible_astrometry_positions():
    with pytest.raises(ValueError, match="astrometry_position_group"):
        validate_units({"x": u.mas, "y": u.AU})


def test_validate_units_allows_different_but_compatible_position_units():
    validate_units({"x": u.mas, "y": u.arcsec, "x_err": u.mas})


def test_angular_astrometry_sets_projected_parameters_to_radians():
    units = UnitSystem(
        {
            "x": "mas",
            "y": "mas",
            "x_err": "mas",
            "y_err": "mas",
            "A": "mas",
            "mu_alpha": "mas/yr",
            "parallax": "mas",
        }
    )

    assert units.astrometry_dimension == "angle"
    assert units.get_parameter_dimension("A") == "angle"
    assert units.get_parameter_dimension("mu_alpha") == "angular_velocity"
    assert units.get_parameter_dimension("parallax") == "angle"
    assert units.canonical_units["A"] == u.rad
    assert units.canonical_units["parallax"] == u.rad


def test_orbital_angles_remain_radians_when_astrometry_is_in_mas():
    units = UnitSystem({"x": "mas", "y": "mas", "parallax": "mas"})

    assert units.get_parameter_unit("omega1") == u.rad
    assert units.get_parameter_unit("Omega1") == u.rad
    assert units.get_parameter_unit("i") == u.rad
    assert units.get_parameter_unit("M0") == u.rad
    assert units.get_parameter_unit("a1") == u.mas
    assert units.get_parameter_unit("parallax") == u.mas


def test_linear_astrometry_sets_projected_parameters_to_au():
    units = UnitSystem(
        {
            "x": "AU",
            "y": "AU",
            "A": "AU",
            "mu_alpha": "AU/yr",
        }
    )

    assert units.astrometry_dimension == "length"
    assert units.get_parameter_dimension("A") == "length"
    assert units.get_parameter_dimension("mu_alpha") == "velocity"
    assert units.canonical_units["A"] == u.AU


def test_unit_system_converts_arrays_to_and_from_canonical_units():
    units = UnitSystem({"x": "mas", "y": "mas"})
    values = np.array([1.0, 2.0])

    canonical = units.convert_to_canonical("x", values)
    restored = units.convert_from_canonical("x", canonical)

    np.testing.assert_allclose(canonical, values * u.mas.to(u.rad))
    np.testing.assert_allclose(restored, values)


def test_parallax_is_an_angular_quantity_in_the_unit_system():
    units = UnitSystem({"x": "mas", "y": "mas", "parallax": "mas"})

    assert units.get_parameter_dimension("parallax") == "angle"
    assert units.convert_to_canonical("parallax", 1) == pytest.approx(
        u.mas.to(u.rad)
    )

def test_canonical_units_include_velocity():
    assert CanonicalUnits["velocity"] == u.AU / u.yr

def test_all_parameters_map_to_dimension():
    for param in all_parameters:
        dimension = get_default_dimension(param)
        assert dimension is not None, f"Parameter {param} does not map to a dimension"
