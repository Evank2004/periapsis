import numpy as np

from orbit_package.data.common import AstroRVData, AstrometryData, RadialVelocityData
from orbit_package.data.joint_data import JointData


class DummyOrbit:
    def __init__(self, x, y, rv):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._rv = np.asarray(rv)

    def astrometry(self, t):
        return self._x, self._y

    def radial_velocity(self, t):
        return self._rv


def test_astrometry_data_uses_explicit_reference_epoch():
    data = AstrometryData(
        t=np.array([1.0, 2.0, 3.0]),
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, 2.0, 3.0]),
        x_err=np.ones(3),
        y_err=np.ones(3),
        ref_epoch=2.5,
    )

    assert data.ref_epoch == 2.5


def test_astrometry_data_chi2_is_zero_for_exact_match():
    data = AstrometryData(
        t=np.array([1.0, 2.0]),
        x=np.array([3.0, 4.0]),
        y=np.array([5.0, 6.0]),
        x_err=np.ones(2),
        y_err=np.ones(2),
        ref_epoch=2.0,
    )
    orbit = DummyOrbit(x=[3.0, 4.0], y=[5.0, 6.0], rv=[0.0, 0.0])

    assert data.chi2(orbit) == 0.0


def test_radial_velocity_data_chi2_is_zero_for_exact_match():
    data = RadialVelocityData(
        t=np.array([1.0, 2.0]),
        rv=np.array([10.0, 11.0]),
        rv_err=np.ones(2),
    )
    orbit = DummyOrbit(x=[0.0, 0.0], y=[0.0, 0.0], rv=[10.0, 11.0])

    assert data.chi2(orbit) == 0.0


def test_joint_data_adds_component_chi2_values():
    astro = AstrometryData(
        t=np.array([0.0]),
        x=np.array([1.0]),
        y=np.array([2.0]),
        x_err=np.ones(1),
        y_err=np.ones(1),
        ref_epoch=0.0,
    )
    rv = RadialVelocityData(t=np.array([0.0]), rv=np.array([3.0]), rv_err=np.ones(1))
    orbit = DummyOrbit(x=[1.0], y=[2.0], rv=[3.0])

    assert JointData([astro, rv]).chi2(orbit) == 0.0


def test_astro_rv_data_chi2_is_zero_for_exact_match():
    data = AstroRVData(
        t=np.array([0.0]),
        x=np.array([1.0]),
        x_err=np.ones(1),
        y=np.array([2.0]),
        y_err=np.ones(1),
        rv=np.array([3.0]),
        rv_err=np.ones(1),
    )
    orbit = DummyOrbit(x=[1.0], y=[2.0], rv=[3.0])

    assert data.chi2(orbit) == 0.0