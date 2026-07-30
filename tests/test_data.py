from types import SimpleNamespace

import numpy as np
import pytest

import periapsis.data as data_package
from periapsis.data import (
    AstrometryData,
    Data,
    GaiaData,
    JointData,
    RadialVelocityData,
)


class DummyOrbit:
    def __init__(self, x=(), y=(), rv=(), gaia=(), derived_params=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.rv_values = np.asarray(rv)
        self.gaia = np.asarray(gaia)
        self.derived_params = {} if derived_params is None else derived_params
        self.calls = []

    def astrometry(self, t, system=None):
        self.calls.append(("astrometry", t, system))
        return self.x, self.y

    def rv(self, t, system=None):
        self.calls.append(("rv", t, system))
        return self.rv_values

    def gaia_astrometry(self, t, spsi, cpsi, plx_fac, system=None):
        self.calls.append(
            ("gaia_astrometry", t, spsi, cpsi, plx_fac, system)
        )
        return self.gaia


def make_astrometry(**overrides):
    arguments = {
        "t": np.array([1.0, 2.0, 3.0]),
        "x": np.array([2.0, 4.0, 6.0]),
        "y": np.array([-1.0, 1.0, 3.0]),
        "x_err": np.array([1.0, 2.0, 1.0]),
        "y_err": np.array([2.0, 1.0, 2.0]),
        "system": "1",
    }
    arguments.update(overrides)
    return AstrometryData(**arguments)


def make_rv(**overrides):
    arguments = {
        "t": np.array([1.0, 2.0, 3.0]),
        "rv": np.array([10.0, 12.0, 14.0]),
        "rv_err": np.array([1.0, 2.0, 1.0]),
        "system": "2",
    }
    arguments.update(overrides)
    return RadialVelocityData(**arguments)


def make_gaia(**overrides):
    arguments = {
        "spsi": np.array([0.0, 1.0, 0.0]),
        "cpsi": np.array([1.0, 0.0, -1.0]),
        "t": np.array([0.0, 1.0, 2.0]),
        "plx_fac": np.array([0.2, 0.3, 0.4]),
        "x": np.array([1.0, 2.0, 3.0]),
        "err": np.array([1.0, 2.0, 1.0]),
        "system": "1",
    }
    arguments.update(overrides)
    return GaiaData(**arguments)


def test_data_package_exports_all_public_data_classes():
    assert data_package.__all__ == [
        "Data",
        "AstrometryData",
        "RadialVelocityData",
        "JointData",
        "GaiaData",
    ]


def test_data_base_class_is_abstract():
    with pytest.raises(TypeError):
        Data()


@pytest.mark.parametrize(
    "data_class",
    [AstrometryData, RadialVelocityData, GaiaData],
)
def test_system_data_requires_a_system(data_class):
    if data_class is AstrometryData:
        arguments = dict(t=0.0, x=0.0, y=0.0, x_err=1.0, y_err=1.0)
    elif data_class is RadialVelocityData:
        arguments = dict(t=0.0, rv=0.0, rv_err=1.0)
    else:
        arguments = dict(
            spsi=0.0,
            cpsi=1.0,
            t=0.0,
            plx_fac=0.0,
            x=0.0,
            err=1.0,
        )

    with pytest.raises(ValueError, match="system"):
        data_class(**arguments)


@pytest.mark.parametrize("system", [1, 2, "relative"])
def test_system_is_normalized_to_a_string(system):
    data = make_rv(system=system)

    assert data.system == str(system)


def test_invalid_system_is_rejected():
    with pytest.raises(ValueError, match="system"):
        make_rv(system="3")


def test_astrometry_converts_scalar_inputs_to_one_dimensional_arrays():
    data = AstrometryData(
        t=2.0,
        x=3.0,
        y=4.0,
        x_err=0.5,
        y_err=0.75,
        system=1,
    )

    for value in (data.t, data.x, data.y, data.x_err, data.y_err):
        assert value.shape == (1,)


def test_astrometry_broadcasts_scalar_uncertainties():
    data = make_astrometry(x_err=0.5, y_err=2.0)

    np.testing.assert_allclose(data.x_err, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(data.y_err, [2.0, 2.0, 2.0])
    assert not data.x_err.flags.writeable
    assert not data.y_err.flags.writeable


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", [1.0, 2.0]),
        ("y", [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_astrometry_rejects_coordinate_shape_mismatches(field, value):
    with pytest.raises(ValueError, match="same shape"):
        make_astrometry(**{field: value})


def test_astrometry_rejects_nonbroadcastable_uncertainties():
    with pytest.raises(ValueError):
        make_astrometry(x_err=np.ones(2))


def test_astrometry_defaults_reference_epoch_to_mean_time():
    data = make_astrometry(t=np.array([2.0, 5.0, 11.0]))

    assert data.ref_epoch == pytest.approx(6.0)


def test_astrometry_uses_explicit_reference_epoch():
    data = make_astrometry(ref_epoch=2.5)

    assert data.ref_epoch == 2.5


def test_astrometry_stores_proper_motion_only_when_both_components_exist():
    complete = make_astrometry(mu_x=1.2, mu_y=-0.4)
    partial = make_astrometry(mu_x=1.2)

    assert complete.mu_x == 1.2
    assert complete.mu_y == -0.4
    assert partial.mu_x is None
    assert partial.mu_y is None


def test_astrometry_chi2_uses_both_weighted_coordinates_and_system():
    data = make_astrometry()
    orbit = DummyOrbit(
        x=[1.0, 2.0, 5.0],
        y=[1.0, 0.0, 1.0],
    )

    result = data.chi2(orbit)

    expected_x = 1.0**2 + 1.0**2 + 1.0**2
    expected_y = 1.0**2 + 1.0**2 + 1.0**2
    assert result == pytest.approx(expected_x + expected_y)
    method, times, system = orbit.calls[0]
    assert method == "astrometry"
    assert times is data.t
    assert system == "1"


def test_astrometry_observation_accessors_return_original_arrays():
    data = make_astrometry()

    x, y = data._astrometry(DummyOrbit())
    t_series = data.t_series()

    assert x is data.x
    assert y is data.y
    assert t_series[0] is data.x
    assert t_series[1] is data.y
    assert t_series[2] is None
    assert t_series[3] is data.t


def test_radial_velocity_converts_scalars_and_broadcasts_uncertainty():
    scalar = RadialVelocityData(t=1.0, rv=2.0, rv_err=0.5, system=1)
    broadcast = make_rv(rv_err=0.25)

    assert scalar.t.shape == scalar.rv.shape == scalar.rv_err.shape == (1,)
    np.testing.assert_allclose(broadcast.rv_err, [0.25, 0.25, 0.25])


def test_radial_velocity_rejects_value_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        make_rv(rv=[1.0, 2.0])


def test_radial_velocity_rejects_nonbroadcastable_uncertainty():
    with pytest.raises(ValueError):
        make_rv(rv_err=np.ones(2))


def test_radial_velocity_chi2_is_weighted_and_passes_system():
    data = make_rv()
    orbit = DummyOrbit(rv=[9.0, 10.0, 13.0])

    result = data.chi2(orbit)

    assert result == pytest.approx(3.0)
    method, times, system = orbit.calls[0]
    assert method == "rv"
    assert times is data.t
    assert system == "2"


def test_radial_velocity_observation_accessors_return_original_arrays():
    data = make_rv()

    assert data._radial_velocity(DummyOrbit()) is data.rv
    x, y, rv, times = data.t_series()
    assert x is None
    assert y is None
    assert rv is data.rv
    assert times is data.t


def test_gaia_converts_scalars_to_one_dimensional_arrays():
    data = GaiaData(
        spsi=0.0,
        cpsi=1.0,
        t=2.0,
        plx_fac=0.5,
        x=3.0,
        err=0.1,
        system=1,
    )

    for value in (
        data.spsi,
        data.cpsi,
        data.t,
        data.plx_fac,
        data.x,
        data.err,
    ):
        assert value.shape == (1,)


def test_gaia_broadcasts_scalar_scan_geometry_and_uncertainty():
    data = make_gaia(spsi=0.5, cpsi=-0.5, plx_fac=0.25, err=0.1)

    np.testing.assert_allclose(data.spsi, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(data.cpsi, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(data.plx_fac, [0.25, 0.25, 0.25])
    np.testing.assert_allclose(data.err, [0.1, 0.1, 0.1])


def test_gaia_rejects_observation_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        make_gaia(x=[1.0, 2.0])


@pytest.mark.parametrize("field", ["spsi", "cpsi", "plx_fac", "err"])
def test_gaia_rejects_nonbroadcastable_inputs(field):
    with pytest.raises(ValueError):
        make_gaia(**{field: np.ones(2)})


def test_gaia_chi2_uses_reported_uncertainties_without_jitter():
    data = make_gaia()
    orbit = DummyOrbit(gaia=[0.0, 0.0, 0.0])

    result = data.chi2(orbit)

    assert result == pytest.approx(11.0)
    call = orbit.calls[0]
    assert call[0] == "gaia_astrometry"
    assert call[1] is data.t
    assert call[2] is data.spsi
    assert call[3] is data.cpsi
    assert call[4] is data.plx_fac
    assert call[5] == "1"


def test_gaia_chi2_combines_explicit_jitter_in_quadrature():
    data = make_gaia(x=[2.0, 2.0, 2.0], err=1.0)
    orbit = DummyOrbit(gaia=[0.0, 0.0, 0.0])

    result = data.chi2(orbit, jitter=1.0)

    assert result == pytest.approx(6.0)


def test_gaia_orbit_jitter_takes_precedence_over_explicit_jitter():
    data = make_gaia(x=[3.0, 3.0, 3.0], err=1.0)
    orbit = DummyOrbit(
        gaia=[0.0, 0.0, 0.0],
        derived_params={"jitter": 2.0},
    )

    result = data.chi2(orbit, jitter=10.0)

    assert result == pytest.approx(27.0 / 5.0)


def test_gaia_basic_observation_accessors():
    data = make_gaia()

    assert data.t_series() is data.t
    assert data._radial_velocity(DummyOrbit()) is None


def test_gaia_astrometry_decomposition_has_expected_keys_and_shapes():
    data = make_gaia()
    params = {
        "P": 1.0,
        "Tp": 0.0,
        "e": 0.0,
        "A1": 0.0,
        "B1": 0.0,
        "F1": 0.0,
        "G1": 0.0,
        "dalpha": 0.0,
        "ddelta": 0.0,
        "mu_alpha": 0.0,
        "mu_delta": 0.0,
        "parallax": 0.0,
    }
    orbit = DummyOrbit(gaia=np.zeros(3), derived_params=params)

    result = data._astrometry(orbit)

    assert set(result) == {
        "ra_obs",
        "dec_obs",
        "ra_orb",
        "dec_orb",
        "ra_orb_obs",
        "dec_orb_obs",
        "ra_lin",
        "dec_lin",
        "ra_full",
        "dec_full",
        "ra_sky",
        "dec_sky",
        "ra_peri",
        "dec_peri",
        "t_smooth",
        "ra_sky_data",
        "dec_sky_data",
    }
    np.testing.assert_allclose(result["ra_obs"], data.x * data.spsi)
    np.testing.assert_allclose(result["dec_obs"], data.x * data.cpsi)
    np.testing.assert_allclose(result["ra_orb"], 0.0)
    np.testing.assert_allclose(result["dec_orb"], 0.0)
    assert result["t_smooth"].shape == (1000,)
    assert result["ra_sky_data"].shape == data.t.shape
    assert result["dec_sky_data"].shape == data.t.shape


def test_joint_data_stores_components_and_sums_chi2():
    calls = []

    class Component:
        def __init__(self, value):
            self.value = value

        def chi2(self, orbit):
            calls.append((self, orbit))
            return self.value

    components = [Component(1.5), Component(2.25), Component(0.25)]
    orbit = object()
    data = JointData(components)

    assert data.datas is components
    assert data.chi2(orbit) == pytest.approx(4.0)
    assert calls == [(component, orbit) for component in components]


def test_joint_data_concatenates_available_astrometry():
    class AstrometryComponent:
        def __init__(self, x, y):
            self.x = np.asarray(x)
            self.y = np.asarray(y)

        def _astrometry(self, orbit):
            return self.x, self.y

    data = JointData(
        [
            AstrometryComponent([1.0, 2.0], [3.0, 4.0]),
            AstrometryComponent([5.0], [6.0]),
        ]
    )

    x, y = data._astrometry(object())

    np.testing.assert_allclose(x, [1.0, 2.0, 5.0])
    np.testing.assert_allclose(y, [3.0, 4.0, 6.0])


def test_joint_data_concatenates_available_radial_velocities():
    components = [
        SimpleNamespace(_radial_velocity=lambda orbit: np.array([1.0, 2.0])),
        SimpleNamespace(_radial_velocity=lambda orbit: np.array([3.0])),
    ]

    result = JointData(components)._radial_velocity(object())

    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_joint_data_ignores_modalities_a_component_does_not_provide():
    astrometry = make_astrometry()
    radial_velocity = make_rv()
    joint = JointData([astrometry, radial_velocity])

    x, y = joint._astrometry(DummyOrbit())
    rv = joint._radial_velocity(DummyOrbit())

    np.testing.assert_allclose(x, astrometry.x)
    np.testing.assert_allclose(y, astrometry.y)
    np.testing.assert_allclose(rv, radial_velocity.rv)


def test_joint_data_concatenates_component_time_arrays():
    components = [
        SimpleNamespace(t_series=lambda: np.array([1.0, 2.0])),
        SimpleNamespace(t_series=lambda: np.array([4.0])),
    ]

    result = JointData(components).t_series()

    np.testing.assert_allclose(result, [1.0, 2.0, 4.0])
