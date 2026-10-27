"""
Tests end-to-end functionality of the periapsis package, including integration of multiple modules and components.

Integration test assumes that p.Orbit is working properly in order to generate synthetic data for tests. This module is tested independently in test_model.py.
"""

import pytest
import astropy.units as u
import numpy as np
import periapsis as p

model1 = p.Orbit(
    a1 = 1, # AU
    e = 0.3,
    cosi = 0.8, # rad
    P = 1, # yr
    omega1 = np.pi/4, # rad
    Omega1 = np.pi/3, # rad
    M0 = 0.0, # rad
    Tepoch = 0.0, # yr
    distance = 1000.0, # AU
    dalpha=0.0, # AU
    ddelta=0.0, # AU
    mu_alpha=0.0, # AU/yr
    mu_delta=0.0, # AU/yr
)

def test_astrometry_ang(capsys):
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
    }
    data = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)


    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=10,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )

    # Check that the proper warnings are emitted when prior units are not explicitly set for dimensionful parameters
    result = fitter.fit(data, rng=np.random.default_rng(42))
    stdout = capsys.readouterr().out
    warnings = [line for line in stdout.splitlines() if "ASSUMPTION:" in line]
    assert len(warnings) == 2
    assert warnings[0].startswith("ASSUMPTION: Assuming prior on 'P' is in units of yr")
    assert warnings[1].startswith("ASSUMPTION: Assuming prior on 'M0' is in units of rad")

    
    for name, param in sorted(result.median_params.items()):
        assert param == pytest.approx(model1[name], rel=0.05, abs=1e-3), f"Parameter {name} does not match expected value. Expected {model1[name]}, got {param}."


def test_astrometry_physical(capsys):
    t = np.linspace(0, 2, 101) # yr
    x_len, y_len = model1.astrometry(t, 0, 0, system=1)
    x_err = 0.01 * np.max(x_len) * np.ones_like(x_len)
    y_err = 0.01 * np.max(y_len) * np.ones_like(y_len)
    units = {
        "t": "yr",
        "x": "AU",
        "y": "AU",
        "x_err": "AU",
        "y_err": "AU",
    }
    data = p.AstrometryData(t=t, x=x_len, y=y_len, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    assert type(data) == p.data.PhysicalAstrometryData
    assert data.unit_system is not None

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=10,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )

    # Check that the proper warnings are emitted when prior units are not explicitly set for dimensionful parameters
    result = fitter.fit(data, rng=np.random.default_rng(42))
    stdout = capsys.readouterr().out
    warnings = [line for line in stdout.splitlines() if "ASSUMPTION:" in line]
    assert len(warnings) == 2
    assert warnings[0].startswith("ASSUMPTION: Assuming prior on 'P' is in units of yr")
    assert warnings[1].startswith("ASSUMPTION: Assuming prior on 'M0' is in units of rad")

    
    for name, param in sorted(result.median_params.items()):
        assert param == pytest.approx(model1[name], rel=0.05, abs=1e-3), f"Parameter {name} does not match expected value. Expected {model1[name]}, got {param}."


def test_astrometry_ang_concise_data_units(capsys):
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    data = p.AstrometryData(t=t*u.yr, x=x_ang*u.mas, y=y_ang*u.mas, x_err=x_err*u.mas, y_err=y_err*u.mas, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=10,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )

    # Check that the proper warnings are emitted when prior units are not explicitly set for dimensionful parameters
    result = fitter.fit(data, rng=np.random.default_rng(42))
    stdout = capsys.readouterr().out
    warnings = [line for line in stdout.splitlines() if "ASSUMPTION:" in line]
    assert len(warnings) == 2
    assert warnings[0].startswith("ASSUMPTION: Assuming prior on 'P' is in units of yr")
    assert warnings[1].startswith("ASSUMPTION: Assuming prior on 'M0' is in units of rad")

    
    for name, param in sorted(result.median_params.items()):
        assert param == pytest.approx(model1[name], rel=0.05, abs=1e-3), f"Parameter {name} does not match expected value. Expected {model1[name]}, got {param}."


def test_astrometry_ang_concise_data_prior_units():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    data = p.AstrometryData(t=t*u.yr, x=x_ang*u.mas, y=y_ang*u.mas, x_err=x_err*u.mas, y_err=y_err*u.mas, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=10,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5, u.yr),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi, "rad"),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    for name, param in sorted(result.median_params.items()):
        assert param == pytest.approx(model1[name], rel=0.05, abs=1e-3), f"Parameter {name} does not match expected value. Expected {model1[name]}, got {param}."