import numpy as np
import pytest
from astropy import units as u

from periapsis.params.units import (
    CanonicalUnits,
    UnitSystem,
    get_unit_type,
    parse_units,
    unit_from_string,
    validate_units,
)


def test_unit_aliases_parse_to_astropy_units():
    assert unit_from_string("mas") == u.mas
    assert unit_from_string("AU/yr") == u.AU / u.yr
    assert unit_from_string("Msun") == u.M_sun


def test_parse_units_returns_astropy_units():
    parsed = parse_units({"x": "mas", "t": "yr"})

    assert parsed == {"x": u.mas, "t": u.yr}


def test_validate_units_requires_compatible_astrometry_positions():
    with pytest.raises(ValueError, match="astrometry_position_group"):
        validate_units({"x": u.mas, "y": u.AU})


def test_validate_units_allows_different_but_compatible_position_units():
    validate_units({"x": u.mas, "y": u.arcsec, "x_err": u.mas})


def test_get_unit_type_distinguishes_distance_from_length():
    assert get_unit_type(u.pc) == "distance"
    assert get_unit_type(u.AU) == "length"


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
    assert units.get_parameter_dimension("mu_alpha") == "linear_velocity"
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


def test_canonical_units_include_linear_velocity():
    assert CanonicalUnits["linear_velocity"] == u.AU / u.yr