"""
Tests end-to-end functionality of the periapsis package, including integration of multiple modules and components.

Integration test assumes that p.Orbit is working properly in order to generate synthetic data for tests. This module is tested independently in test_model.py.
"""

import pytest
import astropy.units as u
import numpy as np
import periapsis as p

model1 = p.Orbit(
    a1 = 2.4, # AU
    e = 0.3,
    i = 0.8, # rad
    P = 0.7, # yr
    omega1 = np.pi/4, # rad
    Omega1 = np.pi/3, # rad
    M0 = 0.0, # rad
    Tepoch = 0.0, # yr
    distance = 1000.0, # AU
    dalpha=0.0, # AU
    ddelta=0.0, # AU
    mu_alpha=0.0, # AU/yr
    mu_delta=0.0, # AU/yr
    gamma=0.0, # AU/yr
)


def _ensure_parameters_match(model, result):
    bad_params = []
    for name in sorted(result.median_params.keys()):
        lower, median, upper = np.percentile(result[name, None], [2.28, 50, 97.72])
        if not (lower <= model[name] <= upper):
            if np.isclose(model[name], median, rtol=1e-10, atol=0):
                # The 2-sigma confidence interval is too precise to be meaningfully compared. The value is close enough.
                continue
            bad_params.append((name, model[name], (lower, upper)))


    assert not bad_params, f"Parameters do not match expected values:{'\n'.join(f'  {name}: {param} (expected {expected})' for name, param, expected in bad_params)}"


def test_astrometry_ang_mcmc_linear(capsys):
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
        nwalkers=25,
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

    _ensure_parameters_match(model1, result)


def test_astrometry_physical_mcmc_linear(capsys):
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
        nwalkers=25,
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

    _ensure_parameters_match(model1, result)


def test_astrometry_ang_concise_data_units_mcmc_linear(capsys):
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    data = p.AstrometryData(t=t*u.yr, x=x_ang*u.mas, y=y_ang*u.mas, x_err=x_err*u.mas, y_err=y_err*u.mas, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
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


    _ensure_parameters_match(model1, result)


def test_astrometry_ang_concise_data_prior_units_mcmc_linear():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    data = p.AstrometryData(t=t*u.yr, x=x_ang*u.mas, y=y_ang*u.mas, x_err=x_err*u.mas, y_err=y_err*u.mas, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5, u.yr),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi, "rad"),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_rv_mcmc_linear():
    t = np.linspace(0, 2, 101) # yr
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    data = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_gaia_astrometry_mcmc_linear():
    t = np.linspace(0, 2, 101) # yr
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x) * np.ones_like(x)
    units = {
        "t": "yr",
        "x": "mas",
        "err": "mas",
    }
    data = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=x, err=x_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_mcmc_linear():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata])

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_with_gaia_mcmc_linear():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    gaia_x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    gaia_x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    gaia_x_err = 0.01 * np.max(gaia_x) * np.ones_like(gaia_x)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    gaiadata = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=gaia_x, err=gaia_x_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata, gaiadata])

    np.random.seed(42)
    fitter = p.MCMCLinearFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )

    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_astrometry_ang_mcmc():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 100 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 100 * np.max(y_ang) * np.ones_like(y_ang)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
    }
    data = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega=p.UniformPrior(0.0, 2*np.pi),
        Omega=p.UniformPrior(0.0, 2*np.pi),
        cosi=p.UniformPrior(-1.0, 1.0),
        a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
        distance=p.FixedPrior(model1["distance"], "AU"),
        dalpha=p.FixedPrior(model1["dalpha"], "AU"),
        ddelta=p.FixedPrior(model1["ddelta"], "AU"),
        mu_alpha=p.FixedPrior(model1["mu_alpha"], "AU/yr"),
        mu_delta=p.FixedPrior(model1["mu_delta"], "AU/yr"),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_rv_mcmc():
    t = np.linspace(0, 2, 101) # yr
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    data = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.MCMCFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega=p.UniformPrior(0.0, 2*np.pi),
        K1=p.UniformPrior(model1["K1"]*0.5, model1["K1"]*1.5),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_mcmc():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata])

    np.random.seed(42)
    fitter = p.MCMCFitter(
        nwalkers=25,
        niter=15000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega1=p.UniformPrior(0.0, 2*np.pi),
        Omega1=p.UniformPrior(0.0, 2*np.pi),
        i=p.UniformPrior(0.0, np.pi),
        a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
        distance=p.FixedPrior(model1["distance"], "AU"),
        dalpha=p.FixedPrior(model1["dalpha"], "AU"),
        ddelta=p.FixedPrior(model1["ddelta"], "AU"),
        mu_alpha=p.FixedPrior(model1["mu_alpha"], "AU/yr"),
        mu_delta=p.FixedPrior(model1["mu_delta"], "AU/yr")
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_with_gaia_mcmc():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    gaia_x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    gaia_x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    gaia_x_err = 0.01 * np.max(gaia_x) * np.ones_like(gaia_x)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    gaiadata = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=gaia_x, err=gaia_x_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata, gaiadata])

    np.random.seed(42)
    fitter = p.MCMCFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega=p.UniformPrior(0.0, 2*np.pi),
        Omega=p.UniformPrior(0.0, 2*np.pi),
        i=p.UniformPrior(0.0, np.pi),
        a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
        parallax=p.UniformPrior(0.5*model1["parallax"], 1.5*model1["parallax"], "rad"),
        dalpha=p.FixedPrior(model1["dalpha"], "AU"),
        ddelta=p.FixedPrior(model1["ddelta"], "AU"),
        mu_alpha=p.FixedPrior(model1["mu_alpha"], "AU/yr"),
        mu_delta=p.FixedPrior(model1["mu_delta"], "AU/yr"),
    )
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_gaia_astrometry_mcmc():
    t = np.linspace(0, 2, 101) # yr
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x) * np.ones_like(x)
    units = {
        "t": "yr",
        "x": "mas",
        "err": "mas",
    }
    data = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=x, err=x_err, units=units, system=1, ref_epoch=0.0)
    np.random.seed(42)
    fitter = p.MCMCFitter(
        nwalkers=25,
        niter=5000,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega=p.UniformPrior(0.0, 2*np.pi),
        Omega=p.UniformPrior(0.0, 2*np.pi),
        cosi=p.UniformPrior(-1.0, 1.0),
        a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
        parallax=p.UniformPrior(0.5*model1["parallax"], 1.5*model1["parallax"], "rad"),
        dalpha_ang = p.FixedPrior(model1["dalpha_ang"], "rad"),
        ddelta_ang = p.FixedPrior(model1["ddelta_ang"], "rad"),
        mu_alpha=p.FixedPrior(model1["mu_alpha"], "AU/yr"),
        mu_delta=p.FixedPrior(model1["mu_delta"], "AU/yr"),
    )

    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)



def test_astrometry_ang_ultranest_linear():
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
    fitter = p.UltranestLinearFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_rv_ultranest_linear():
    t = np.linspace(0, 2, 101) # yr
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    data = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.UltranestLinearFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_gaia_astrometry_ultranest_linear():
    t = np.linspace(0, 2, 101) # yr
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x) * np.ones_like(x)
    units = {
        "t": "yr",
        "x": "mas",
        "err": "mas",
    }
    data = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=x, err=x_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.UltranestLinearFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_ultranest_linear():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata])

    np.random.seed(42)
    fitter = p.UltranestLinearFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


def test_joint_with_gaia_ultranest_linear():
    t = np.linspace(0, 2, 101) # yr
    x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
    x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
    y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    plx_fac = np.random.default_rng(42).normal(size=len(t))
    psi = np.random.default_rng(42).uniform(0, 2*np.pi, size=len(t))
    gaia_x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=plx_fac, system=1)
    gaia_x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
    gaia_x_err = 0.01 * np.max(gaia_x) * np.ones_like(gaia_x)
    units = {
        "t": "yr",
        "x": "mas",
        "y": "mas",
        "x_err": "mas",
        "y_err": "mas",
        "err": "mas",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
    rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
    gaiadata = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=plx_fac, x=gaia_x, err=gaia_x_err, units=units, system=1, ref_epoch=0.0)
    data = p.JointData([astrodata, rvdata, gaiadata])

    np.random.seed(42)
    fitter = p.UltranestLinearFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
    )

    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


# def test_astrometry_ang_ultranest():
#     t = np.linspace(0, 2, 101) # yr
#     x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
#     x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
#     y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
#     units = {
#         "t": "yr",
#         "x": "mas",
#         "y": "mas",
#         "x_err": "mas",
#         "y_err": "mas",
#     }
#     data = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)

#     np.random.seed(42)
#     fitter = p.UltranestFitter(
#         max_ncalls=100000,
#         quiet=True,
#         P=p.UniformPrior(0.5, 1.5),
#         e=p.UniformPrior(0.0, 0.5),
#         M0=p.UniformPrior(-np.pi, np.pi),
#         omega=p.UniformPrior(0.0, 2*np.pi),
#         Omega=p.UniformPrior(0.0, 2*np.pi),
#         i=p.UniformPrior(0.0, np.pi),
#         a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
#         distance=p.FixedPrior(model1["distance"]),
#         dalpha=p.FixedPrior(model1["dalpha"]),
#         ddelta=p.FixedPrior(model1["ddelta"]),
#         mu_alpha=p.FixedPrior(model1["mu_alpha"]),
#         mu_delta=p.FixedPrior(model1["mu_delta"]),
#     )
    
#     result = fitter.fit(data, rng=np.random.default_rng(42))

#     _ensure_parameters_match(model1, result)


def test_rv_ultranest():
    t = np.linspace(0, 2, 101) # yr
    rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
    rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
    units = {
        "t": "yr",
        "rv": "km/s",
        "rv_err": "km/s",
    }
    data = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)

    np.random.seed(42)
    fitter = p.UltranestFitter(
        max_ncalls=100000,
        quiet=True,
        P=p.UniformPrior(0.5, 1.5),
        e=p.UniformPrior(0.0, 0.5),
        M0=p.UniformPrior(-np.pi, np.pi),
        omega=p.UniformPrior(0.0, 2*np.pi),
        K1=p.UniformPrior(model1["K1"]*0.5, model1["K1"]*1.5, "AU/yr"),
    )
    
    result = fitter.fit(data, rng=np.random.default_rng(42))

    _ensure_parameters_match(model1, result)


# TODO re-enable when ultranest fitter is more efficient
# def test_gaia_astrometry_ultranest():
#     t = np.linspace(0, 2, 101) # yr
#     psi = np.random.default_rng(42).uniform(0, 2*np.pi)
#     x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1.0, system=1)
#     x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     x_err = 0.01 * np.max(x) * np.ones_like(x)
#     units = {
#         "t": "yr",
#         "x": "mas",
#         "err": "mas",
#     }
#     data = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=1.0, x=x, err=x_err, units=units, system=1, ref_epoch=0.0)

#     np.random.seed(42)
#     fitter = p.UltranestFitter(
#         max_ncalls=100000,
#         quiet=True,
#         P=p.UniformPrior(0.5, 1.5),
#         e=p.UniformPrior(0.0, 0.5),
#         M0=p.UniformPrior(-np.pi, np.pi),
#         omega=p.UniformPrior(0.0, 2*np.pi),
#         Omega=p.UniformPrior(0.0, 2*np.pi),
#         i=p.UniformPrior(0.0, np.pi),
#         a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
#         distance=p.FixedPrior(model1["distance"]),
#         dalpha=p.FixedPrior(model1["dalpha"]),
#         ddelta=p.FixedPrior(model1["ddelta"]),
#         mu_alpha=p.FixedPrior(model1["mu_alpha"]),
#         mu_delta=p.FixedPrior(model1["mu_delta"]),
#     )
    
#     result = fitter.fit(data, rng=np.random.default_rng(42))

#     _ensure_parameters_match(model1, result)


# TODO re-enable when ultranest fitter is more efficient
# def test_joint_ultranest():
#     t = np.linspace(0, 2, 101) # yr
#     x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
#     x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
#     y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
#     rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
#     rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
#     units = {
#         "t": "yr",
#         "x": "mas",
#         "y": "mas",
#         "x_err": "mas",
#         "y_err": "mas",
#         "rv": "km/s",
#         "rv_err": "km/s",
#     }
#     astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
#     rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
#     data = p.JointData([astrodata, rvdata])

#     np.random.seed(42)
#     fitter = p.UltranestFitter(
#         max_ncalls=100000,
#         quiet=True,
#         P=p.UniformPrior(0.5, 1.5),
#         e=p.UniformPrior(0.0, 0.5),
#         M0=p.UniformPrior(-np.pi, np.pi),
#         omega=p.UniformPrior(0.0, 2*np.pi),
#         Omega=p.UniformPrior(0.0, 2*np.pi),
#         i=p.UniformPrior(0.0, np.pi),
#         a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
#         distance=p.FixedPrior(model1["distance"]),
#         dalpha=p.FixedPrior(model1["dalpha"]),
#         ddelta=p.FixedPrior(model1["ddelta"]),
#         mu_alpha=p.FixedPrior(model1["mu_alpha"]),
#         mu_delta=p.FixedPrior(model1["mu_delta"]),
#     )
    
#     result = fitter.fit(data, rng=np.random.default_rng(42))

#     _ensure_parameters_match(model1, result)


# TODO re-enable when ultranest fitter is more efficient
# def test_joint_with_gaia_ultranest():
#     t = np.linspace(0, 2, 101) # yr
#     x_ang, y_ang = model1.astrometry_ang(t, 0, 0, system=1)
#     x_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     y_ang *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     x_err = 0.01 * np.max(x_ang) * np.ones_like(x_ang)
#     y_err = 0.01 * np.max(y_ang) * np.ones_like(y_ang)
#     rv = model1.rv(t, system=1) * (u.AU/u.yr).to(u.km/u.s) # convert AU/yr to km/s
#     rv_err = 0.01 * np.max(rv) * np.ones_like(rv)
#     psi = np.random.default_rng(42).uniform(0, 2*np.pi)
#     gaia_x = model1.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1.0, system=1)
#     gaia_x *= 1000 * 3600 * 180 / np.pi # convert rad to mas
#     gaia_x_err = 0.01 * np.max(gaia_x) * np.ones_like(gaia_x)
#     units = {
#         "t": "yr",
#         "x": "mas",
#         "y": "mas",
#         "x_err": "mas",
#         "y_err": "mas",
#         "err": "mas",
#         "rv": "km/s",
#         "rv_err": "km/s",
#     }
#     astrodata = p.AstrometryData(t=t, x=x_ang, y=y_ang, x_err=x_err, y_err=y_err, units=units, system=1, ref_epoch=0.0)
#     rvdata = p.RadialVelocityData(t=t, rv=rv, rv_err=rv_err, units=units, system=1, ref_epoch=0.0)
#     gaiadata = p.GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=1.0, x=gaia_x, err=gaia_x_err, units=units, system=1, ref_epoch=0.0)
#     data = p.JointData([astrodata, rvdata, gaiadata])

#     np.random.seed(42)
#     fitter = p.UltranestFitter(
#         max_ncalls=100000,
#         quiet=True,
#         P=p.UniformPrior(0.5, 1.5),
#         e=p.UniformPrior(0.0, 0.5),
#         M0=p.UniformPrior(-np.pi, np.pi),
#         omega=p.UniformPrior(0.0, 2*np.pi),
#         Omega=p.UniformPrior(0.0, 2*np.pi),
#         i=p.UniformPrior(0.0, np.pi),
#         a1_ang=p.LogUniformPrior(0.5*model1["a1_ang"], 1.5*model1["a1_ang"], "rad"),
#         distance=p.FixedPrior(model1["distance"]),
#         dalpha=p.FixedPrior(model1["dalpha"]),
#         ddelta=p.FixedPrior(model1["ddelta"]),
#         mu_alpha=p.FixedPrior(model1["mu_alpha"]),
#         mu_delta=p.FixedPrior(model1["mu_delta"]),
#     )

#     result = fitter.fit(data, rng=np.random.default_rng(42))

#     _ensure_parameters_match(model1, result)