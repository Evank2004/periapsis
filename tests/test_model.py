import pytest

from periapsis.model.orbit import Orbit


def test_orbit_requires_complete_parameter_set():
    with pytest.raises(ValueError):
        Orbit(P=10.0, e=0.1)


def test_orbit_methods_raise_until_implemented():
    orbit = Orbit(P=10.0, e=0.1, t0=0.0, omega=0.0, Omega=0.0, cosi=1.0, a=1.0)

    with pytest.raises(NotImplementedError):
        orbit.astrometry(0.0)

    with pytest.raises(NotImplementedError):
        orbit.rv(0.0)

    with pytest.raises(NotImplementedError):
        orbit.xyz(0.0)

    with pytest.raises(NotImplementedError):
        orbit.vxyz(0.0)