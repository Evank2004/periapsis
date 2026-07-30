import pytest
import numpy as np
from periapsis.model.orbit import Orbit


def test_orbit_requires_parameters():
    with pytest.raises(ValueError):
        Orbit()

    orb = Orbit(P=10.0)
    assert 'P' in orb.params
    assert 'P' in orb.covered_params
    assert 'P' in orb.derived_params


def test_orbit_contains_method():
    orb = Orbit(P=10.0)
    assert 'P' in orb.covered_params
    assert 'P' in orb
    assert 'e' not in orb.covered_params
    assert 'e' not in orb


def test_orbit_derives_params():
    orb = Orbit(P=10.0)
    assert 'n' in orb


def test_orbit_getitem_method():
    orb = Orbit(P=10.0)
    assert orb['P'] == 10.0


def test_orbit_transforms_n_correctly():
    orb = Orbit(P=10.0)
    assert orb['n'] == 2 * 3.141592653589793 / 10.0


def test_lookup_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb['e']


def test_orbit_astrometry_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb.astrometry(0.0, system='1')


def test_orbit_rv_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb.rv(0.0, system='1')


def test_orbit_xyz_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb.xyz(0.0, system='1')


def test_orbit_vxyz_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb.vxyz(0.0, system='1')


def test_orbit_gaia_astrometry_requires_sufficient_parameters():
    orb = Orbit(P=10.0)
    with pytest.raises(KeyError):
        orb.gaia_astrometry(0.0, spsi=1.0, cpsi=0.0, par_factor=1.0, system='1')


def test_orbit_contains_param_not_in_params():
    orb = Orbit(P=10.0)
    assert 'n' in orb
    assert 'n' not in orb.params


def test_orbit_lazily_computes_derived_params():
    orb = Orbit(P=10.0)
    assert 'n' not in orb.derived_params
    assert orb['n'] == 2 * 3.141592653589793 / 10.0
    assert 'n' in orb.derived_params


def test_orbit_derived_params_are_cached():
    orb = Orbit(P=10.0)
    n1 = orb['n']
    n2 = orb['n']
    assert n1 is n2


def test_orbit_params_cannot_be_set_directly():
    orb = Orbit(P=10.0)
    with pytest.raises(AttributeError):
        orb.params = {'P': 20.0}
    with pytest.raises(TypeError):
        orb['P'] = 20.0


def test_orbit_derived_params_cannot_be_set_directly():
    orb = Orbit(P=10.0)
    with pytest.raises(AttributeError):
        orb.derived_params = {'n': 0.5}


def test_orbit_covered_params_cannot_be_set_directly():
    orb = Orbit(P=10.0)
    with pytest.raises(AttributeError):
        orb.covered_params = {'P', 'n'}


def test_orbit_velocity_ratio_is_read_only():
    orb = Orbit(P=10.0)
    with pytest.raises(AttributeError):
        orb.velocity_ratio = 2.0


def test_orbit_parameter_mappings_are_read_only():
    orb = Orbit(P=10.0)

    with pytest.raises(TypeError):
        orb.params['P'] = 20.0
    with pytest.raises(TypeError):
        orb.derived_params['n'] = 0.5


def test_orbit_covered_params_are_read_only():
    orb = Orbit(P=10.0)

    with pytest.raises(AttributeError):
        orb.covered_params.add('e')


def test_orbit_defaults_tepoch_to_zero():
    assert Orbit(P=10.0)['Tepoch'] == 0.0


def test_no_system_provided():
    orb = Orbit(P=10.0)
    with pytest.raises(ValueError):
        orb.astrometry(0.0)
    with pytest.raises(ValueError):
        orb.rv(0.0)
    with pytest.raises(ValueError):
        orb.xyz(0.0)
    with pytest.raises(ValueError):
        orb.vxyz(0.0)
    with pytest.raises(ValueError):
        orb.gaia_astrometry(0.0, spsi=1.0, cpsi=0.0, par_factor=1.0)


def test_nondefined_system():
    orb = Orbit(P=10.0)
    with pytest.raises(ValueError):
        orb.astrometry(0.0, system='3')
    with pytest.raises(ValueError):
        orb.rv(0.0, system='3')
    with pytest.raises(ValueError):
        orb.xyz(0.0, system='3')
    with pytest.raises(ValueError):
        orb.vxyz(0.0, system='3')
    with pytest.raises(ValueError):
        orb.gaia_astrometry(0.0, spsi=1.0, cpsi=0.0, par_factor=1.0, system='3')


def test_orbit_astrometry_computes_for_fully_defined_orbit():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0)
    t = 0.0
    result = orb.astrometry(t, system='relative')
    assert isinstance(result, tuple)
    assert len(result) == 2
    result1 = orb.astrometry(t, system='1')
    assert isinstance(result1, tuple)
    assert len(result1) == 2
    result2 = orb.astrometry(t, system='2')
    assert isinstance(result2, tuple)
    assert len(result2) == 2


def test_orbit_rv_computes_for_fully_defined_orbit():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t = 0.0
    result = orb.rv(t, system='relative')
    assert isinstance(result, float)
    result1 = orb.rv(t, system='1')
    assert isinstance(result1, float)
    result2 = orb.rv(t, system='2')
    assert isinstance(result2, float)


def test_orbit_rv_applies_velocity_ratio_and_component_systemic_velocity():
    common = dict(P=1.0, e=0.0, Tp=0.0, Tepoch=0.0, systemic_velocity=30.0)
    relative = Orbit(K=4.0, omega=0.0, velocity_ratio=2.5, **common)
    primary = Orbit(K1=4.0, omega1=0.0, velocity_ratio=2.5, **common)

    assert relative.rv(0.0, system='relative') == pytest.approx(10.0)
    assert primary.rv(0.0, system='1') == pytest.approx(40.0)
    assert relative.rv(0.25, system='relative') == pytest.approx(0.0)


def test_orbit_astrometry_applies_offsets_and_proper_motion():
    orb = Orbit(
        P=1.0, e=0.0, Tp=10.0, Tepoch=10.0,
        A1=0.0, B1=0.0, F1=0.0, G1=0.0,
        dx=2.0, dy=-3.0, dpmra=0.5, dpmdec=1.0,
    )

    assert orb.astrometry(12.0, system='1') == pytest.approx((3.0, -1.0))


def test_orbit_method_derivations_are_cached():
    orb = Orbit(
        P=1.0, e=0.0, Tp=0.0, Tepoch=0.0,
        a1=1.0, omega1=0.0, Omega=0.0, i=0.0,
        dx=0.0, dy=0.0, distance=1.0, dpmra=0.0, dpmdec=0.0,
    )

    assert 'A1' not in orb.derived_params
    orb.astrometry(0.0, system='1')
    assert {'A1', 'B1', 'F1', 'G1'}.issubset(orb.derived_params)


def test_orbit_xyz_computes_for_fully_defined_orbit():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t = 0.0
    with pytest.raises(NotImplementedError):
        orb.xyz(t, system='relative')


def test_orbit_vxyz_computes_for_fully_defined_orbit():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t = 0.0
    with pytest.raises(NotImplementedError):
        orb.vxyz(t, system='relative')


def test_orbit_gaia_astrometry_computes_for_fully_defined_orbit():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dalpha=0.0, ddelta=0.0, distance=1000.0, mu_alpha=0.0, mu_delta=0.0, systemic_velocity=100.0)
    t = 0.0
    spsi = 1.0
    cpsi = 0.0
    par_factor = 1.0
    result = orb.gaia_astrometry(t, spsi=spsi, cpsi=cpsi, par_factor=par_factor, system='relative')
    assert isinstance(result, float)
    result1 = orb.gaia_astrometry(t, spsi=spsi, cpsi=cpsi, par_factor=par_factor, system='1')
    assert isinstance(result1, float)
    result2 = orb.gaia_astrometry(t, spsi=spsi, cpsi=cpsi, par_factor=par_factor, system='2')
    assert isinstance(result2, float)


def test_orbit_gaia_astrometry_uses_orbital_phase_and_all_projection_terms():
    orb = Orbit(
        P=2.0, e=0.0, Tp=1.0, Tepoch=0.0,
        A=2.0, B=3.0, F=5.0, G=7.0,
        dalpha=11.0, ddelta=13.0, mu_alpha=17.0, mu_delta=19.0,
        parallax=23.0,
    )

    # At t=1 (periapsis), X=1 and Y=0.
    result = orb.gaia_astrometry(1.0, spsi=2.0, cpsi=3.0, par_factor=29.0, system='relative')
    expected = (11.0 + 17.0) * 2.0 + (13.0 + 19.0) * 3.0 + 23.0 * 29.0 + 3.0 * 2.0 + 2.0 * 3.0
    assert result == pytest.approx(expected)


def test_orbit_astrometry_with_t_array():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t_array = [0.0, 1.0, 2.0]
    result = orb.astrometry(t_array, system='relative')
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert len(result[0]) == len(t_array)
    assert isinstance(result[1], np.ndarray)
    assert len(result[1]) == len(t_array)


def test_orbit_rv_with_t_array():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t_array = [0.0, 1.0, 2.0]
    result = orb.rv(t_array, system='relative')
    assert isinstance(result, np.ndarray)
    assert len(result) == len(t_array)


def test_orbit_xyz_with_t_array():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t_array = [0.0, 1.0, 2.0]
    with pytest.raises(NotImplementedError):
        orb.xyz(t_array, system='relative')


def test_orbit_vxyz_with_t_array():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dx=0.0, dy=0.0, distance=1000.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=100.0)
    t_array = [0.0, 1.0, 2.0]
    with pytest.raises(NotImplementedError):
        orb.vxyz(t_array, system='relative')


def test_orbit_gaia_astrometry_with_t_array():
    orb = Orbit(P=10.0, e=0.5, Tp=0.0, M1=1.0, M2=2.0, i=0.1, Omega=0.2, omega=0.3, dalpha=0.0, ddelta=0.0, distance=1000.0, mu_alpha=0.0, mu_delta=0.0, systemic_velocity=100.0)
    t_array = np.array([0.0, 1.0, 2.0])
    spsi = np.array([1.0, 0.0, -1.0])
    cpsi = np.array([0.0, 1.0, 0.0])
    par_factor = 1.0
    result = orb.gaia_astrometry(t_array, spsi=spsi, cpsi=cpsi, par_factor=par_factor, system='relative')
    assert isinstance(result, np.ndarray)
    assert len(result) == len(t_array)


def test_basic_orbit_parameters():
    orb = Orbit(P=1.0, e=0.0, a1=1.0, Tp=0.0, omega1=0.0, Omega=0.0, i=0.0, dx=0.0, dy=0.0, distance=1.0, dpmra=0.0, dpmdec=0.0)
    assert orb['P'] == 1.0
    assert orb['r_p1'] == 1.0
    assert orb['r_a1'] == 1.0
    time = 0.0
    x, y = orb.astrometry(time, system='1')
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_basic_coordinate_frame():
    orb = Orbit(P=1.0, e=0.0, a1=1.0, Tp=0.0, omega1=0.0, Omega=0.0, i=0.0, dx=0.0, dy=0.0, distance=1.0, dpmra=0.0, dpmdec=0.0)
    time = 0.0
    x, y = orb.astrometry(time, system='1')
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)

    time = 0.25
    x, y = orb.astrometry(time, system='1')
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(1.0)

    time = 0.5
    x, y = orb.astrometry(time, system='1')
    assert x == pytest.approx(-1.0)
    assert y == pytest.approx(0.0)

    time = 0.75
    x, y = orb.astrometry(time, system='1')
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(-1.0)

    time = 1.0
    x, y = orb.astrometry(time, system='1')
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
