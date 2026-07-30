import numpy as np
import pytest

import periapsis.fitting.gaia_mcmclinear as gaia_mcmc_module
import periapsis.fitting.gaia_ultranestlinear as gaia_ultranest_module
import periapsis.fitting.mcmc as mcmc_module
import periapsis.fitting.mcmclinear as mcmc_linear_module
import periapsis.fitting.ultranest as ultranest_module
import periapsis.fitting.ultranestlinear as ultranest_linear_module
from periapsis.data import AstrometryData, RadialVelocityData, GaiaData
from periapsis.data.joint_data import JointData
from periapsis.fitting import *
from periapsis.initial import InitialGuess
from periapsis.model import Orbit
from periapsis.prior import Bounds, UniformPrior, FixedPrior
from periapsis.utils.solvers import gaia_single_motion


def test_mcmc_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "a1": UniformPrior(0.1, 1.0)}
    fitter = MCMCFitter(nwalkers=10, niter=1000, sample_params=["P", "a1"], **priors)
    assert fitter.sample_params == {"P", "a1"}
    assert fitter.param_order == ("P", "a1")

def test_ultranest_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "a1": UniformPrior(0.1, 1.0)}
    fitter = UltranestFitter(output_params=["P", "a1"], **priors)
    assert fitter.output_params == {"P", "a1"}
    assert fitter.output_param_order == ("P", "a1")

def test_mcmc_linear_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "e": UniformPrior(0.0, 1.0), "Tp": UniformPrior(0.0, 10.0)}
    fitter = MCMCLinearFitter(nwalkers=10, niter=1000, sampled_params=["P", "e", "Tp"], **priors)
    assert fitter.sampled_params == {"P", "e", "Tp"}
    assert fitter.param_order == ("P", "e", "Tp")

def test_ultranest_linear_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "e": UniformPrior(0.0, 1.0), "Tp": UniformPrior(0.0, 10.0)}
    fitter = UltranestLinearFitter(**priors)
    assert fitter.output_params == {"P", "e", "Tp"}
    assert fitter.output_param_order == ("P", "e", "Tp")

def test_mcmc_gaia_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "e": UniformPrior(0.0, 1.0), "Tp": UniformPrior(0.0, 10.0)}
    fitter = MCMCGaiaFitter(nwalkers=10, niter=1000, sampled_params=["P", "e", "Tp"], **priors)
    assert fitter.sampled_params == {"P", "e", "Tp", "jitter"}
    assert fitter.param_order == ("P", "e", "Tp", "jitter")

def test_ultranest_gaia_fitter_initializes():
    priors = {"P": UniformPrior(1.0, 10.0), "e": UniformPrior(0.0, 1.0), "Tp": UniformPrior(0.0, 10.0)}
    fitter = UltranestGaiaFitter(**priors)
    assert fitter.output_params == {"P", "e", "Tp", "jitter"}
    assert fitter.output_param_order == ("P", "e", "Tp", "jitter")


test_priors = {
    "P": UniformPrior(1.0, 10.0),
    "a1": UniformPrior(0.1, 10.0),
    "K1": UniformPrior(0.0, 10.0),
    "e": UniformPrior(0.0, 0.99),
    "M0": UniformPrior(0.0, 2 * np.pi),
    "omega": UniformPrior(0.0, 2 * np.pi),
    "cosi": UniformPrior(-1.0, 1.0),
    "Omega": UniformPrior(0.0, 2 * np.pi),
    "dx": FixedPrior(0.0),
    "dy": FixedPrior(0.0),
    "dpmra": FixedPrior(0.0),
    "dpmdec": FixedPrior(0.0),
    "Tepoch": FixedPrior(0.0),
    "systemic_velocity": FixedPrior(0.0),
}

test_astrometry_priors = {
    "P": UniformPrior(1.0, 10.0),
    "a1": UniformPrior(0.1, 10.0),
    "e": UniformPrior(0.0, 0.99),
    "M0": UniformPrior(0.0, 2 * np.pi),
    "omega": UniformPrior(0.0, 2 * np.pi),
    "cosi": UniformPrior(-1.0, 1.0),
    "Omega": UniformPrior(0.0, 2 * np.pi),
    "dx": FixedPrior(0.0),
    "dy": FixedPrior(0.0),
    "dpmra": FixedPrior(0.0),
    "dpmdec": FixedPrior(0.0),
    "Tepoch": FixedPrior(0.0),
}

test_gaia_astrometry_priors = {
    "P": UniformPrior(1.0, 10.0),
    "a1": UniformPrior(0.1, 10.0),
    "e": UniformPrior(0.0, 0.99),
    "M0": UniformPrior(0.0, 2 * np.pi),
    "omega": UniformPrior(0.0, 2 * np.pi),
    "cosi": UniformPrior(-1.0, 1.0),
    "Omega": UniformPrior(0.0, 2 * np.pi),
    "dalpha": FixedPrior(0.0),
    "ddelta": FixedPrior(0.0),
    "mu_alpha": FixedPrior(0.0),
    "mu_delta": FixedPrior(0.0),
    "distance": FixedPrior(10.0),
    "Tepoch": FixedPrior(0.0),
}

test_rv_priors = {
    "P": UniformPrior(1.0, 10.0),
    "K1": UniformPrior(0.0, 10.0),
    "e": UniformPrior(0.0, 0.99),
    "M0": UniformPrior(0.0, 2 * np.pi),
    "omega": UniformPrior(0.0, 2 * np.pi),
    "systemic_velocity": FixedPrior(0.0),
    "Tepoch": FixedPrior(0.0),
}


def test_mcmc_fitter_runs_with_astrometry_data():
    fitter = MCMCFitter(nwalkers=20, niter=1000, sample_params=["P", "a1", "e", "M0", "omega", "cosi", "Omega"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0)
    t = np.linspace(0, 10, 100)
    x, y = model.astrometry(t, system=1)
    data=AstrometryData(t, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    results = fitter.fit(data, np.random.default_rng(0))
    assert isinstance(results, FitResults)


def test_mcmc_fitter_runs_with_rv_data():
    fitter = MCMCFitter(nwalkers=20, niter=1000, sample_params=["P", "e", "M0", "omega", "K1"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t = np.linspace(0, 10, 100)
    v = model.rv(t, system=1)
    data=RadialVelocityData(t, v, 0.01, system=1)
    results = fitter.fit(data, np.random.default_rng(0))
    assert isinstance(results, FitResults)


def test_mcmc_fitter_does_not_run_with_gaia_data():
    fitter = MCMCFitter(nwalkers=20, niter=1000, sample_params=["P", "a1", "e", "M0", "omega", "cosi", "Omega"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.ones(len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    with pytest.raises(ValueError):
        fitter.fit(data, rng=np.random.default_rng(0))


def test_mcmc_fitter_runs_with_joint_data():
    fitter = MCMCFitter(nwalkers=20, niter=1000, sample_params=["P", "a1", "e", "M0", "omega", "i", "Omega"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t_astrometry = np.linspace(0, 10, 100)
    x, y = model.astrometry(t_astrometry, system=1)
    data_astrometry = AstrometryData(t_astrometry, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    t_rv = np.linspace(5.1, 15.1, 10)
    rv = model.rv(t_rv, system=1)
    data_rv = RadialVelocityData(t_rv, rv, 0.01, system=1)
    data = JointData([data_astrometry, data_rv])
    results = fitter.fit(data, rng=np.random.default_rng(0))
    assert isinstance(results, FitResults)

def test_mcmc_linear_fitter_runs_with_astrometry_data():
    fitter = MCMCLinearFitter(nwalkers=20, niter=1000, sampled_params=["P", "e", "Tp"], **test_astrometry_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t = np.linspace(0, 10, 100)
    x, y = model.astrometry(t, system=1)
    data = AstrometryData(t, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    results = fitter.fit(data, rng=np.random.default_rng(0))
    assert isinstance(results, FitResults)

def test_mcmc_linear_fitter_does_not_run_with_rv_data():
    fitter = MCMCLinearFitter(nwalkers=20, niter=1000, sampled_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t = np.linspace(0, 10, 100)
    v = model.rv(t, system=1)
    data = RadialVelocityData(t, v, 0.01, system=1)
    with pytest.raises(ValueError):
        fitter.fit(data, rng=np.random.default_rng(0))

def test_mcmc_linear_fitter_does_not_run_with_joint_data():
    fitter = MCMCLinearFitter(nwalkers=20, niter=1000, sampled_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t_astrometry = np.linspace(0, 10, 100)
    x, y = model.astrometry(t_astrometry, system=1)
    data_astrometry = AstrometryData(t_astrometry, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    t_rv = np.linspace(5.1, 15.1, 10)
    rv = model.rv(t_rv, system=1)
    data_rv = RadialVelocityData(t_rv, rv, 0.01, system=1)
    data = JointData([data_astrometry, data_rv])
    with pytest.raises(ValueError):
        fitter.fit(data, rng=np.random.default_rng(0))

def test_mcmc_linear_fitter_does_not_run_with_gaia_data():
    fitter = MCMCLinearFitter(nwalkers=20, niter=1000, sampled_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.ones(len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    with pytest.raises(ValueError):
        fitter.fit(data, rng=np.random.default_rng(0))

def test_ultranest_fitter_runs_with_astrometry_data():
    fitter = UltranestFitter(max_ncalls=1000, output_params=["P", "a1", "e", "M0", "omega", "cosi", "Omega"], **test_astrometry_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0)
    t = np.linspace(0, 10, 100)
    x, y = model.astrometry(t, system=1)
    data = AstrometryData(t, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    results = fitter.fit(data)
    assert isinstance(results, FitResults)

def test_ultranest_fitter_runs_with_rv_data():
    fitter = UltranestFitter(max_ncalls=1000, output_params=["P", "e", "M0", "omega", "K1"], **test_rv_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t = np.linspace(0, 10, 100)
    v = model.rv(t, system=1)
    data = RadialVelocityData(t, v, 0.01, system=1)
    results = fitter.fit(data)
    assert isinstance(results, FitResults)

def test_ultranest_fitter_runs_with_gaia_data():
    fitter = UltranestFitter(max_ncalls=1000, output_params=["P", "a1", "e", "M0", "omega", "cosi", "Omega",], **test_gaia_astrometry_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.ones(len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    results = fitter.fit(data)
    assert isinstance(results, FitResults)

def test_ultranest_fitter_runs_with_joint_data():
    fitter = UltranestFitter(max_ncalls=1000, output_params=["P", "a1", "e", "M0", "omega", "i", "Omega"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t_astrometry = np.linspace(0, 10, 100)
    x, y = model.astrometry(t_astrometry, system=1)
    data_astrometry = AstrometryData(t_astrometry, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    t_rv = np.linspace(5.1, 15.1, 10)
    rv = model.rv(t_rv, system=1)
    data_rv = RadialVelocityData(t_rv, rv, 0.01, system=1)
    data = JointData([data_astrometry, data_rv])
    results = fitter.fit(data)
    assert isinstance(results, FitResults)

def test_ultranest_linear_fitter_runs_with_astrometry_data():
    fitter = UltranestLinearFitter(max_ncalls=1000, output_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0)
    t = np.linspace(0, 10, 100)
    x, y = model.astrometry(t, system=1)
    data = AstrometryData(t, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    results = fitter.fit(data)
    assert isinstance(results, FitResults)

def test_ultranest_linear_fitter_does_not_run_with_rv_data():
    fitter = UltranestLinearFitter(max_ncalls=1000, output_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t = np.linspace(0, 10, 100)
    v = model.rv(t, system=1)
    data = RadialVelocityData(t, v, 0.01, system=1)
    with pytest.raises(ValueError):
        fitter.fit(data)

def test_ultranest_linear_fitter_does_not_run_with_gaia_data():
    fitter = UltranestLinearFitter(max_ncalls=1000, output_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.ones(len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    with pytest.raises(ValueError):
        fitter.fit(data)

def test_ultranest_linear_fitter_does_not_run_with_joint_data():
    fitter = UltranestLinearFitter(max_ncalls=1000, output_params=["P", "e", "Tp"], **test_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0, systemic_velocity=0.0)
    t_astrometry = np.linspace(0, 10, 100)
    x, y = model.astrometry(t_astrometry, system=1)
    data_astrometry = AstrometryData(t_astrometry, x, y, 0.01, 0.01, ref_epoch=0.0, system=1)
    t_rv = np.linspace(5.1, 15.1, 10)
    rv = model.rv(t_rv, system=1)
    data_rv = RadialVelocityData(t_rv, rv, 0.01, system=1)
    data = JointData([data_astrometry, data_rv])
    with pytest.raises(ValueError):
        fitter.fit(data)

def test_gaia_mcmclinear_fitter_runs_with_gaia_data():
    fitter = MCMCGaiaFitter(nwalkers=20, niter=1000, sampled_params=["P", "e", "Tp"], **test_gaia_astrometry_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.random.uniform(0.5, 1.5, size=len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    results = fitter.fit(data, np.random.default_rng(0))
    assert isinstance(results, FitResults)

def test_gaia_ultranestlinear_fitter_runs_with_gaia_data():
    fitter = UltranestGaiaFitter(max_ncalls=1000, output_params=["P", "e", "Tp"], **test_gaia_astrometry_priors)
    model = Orbit(P=5.0, a1=1.0, e=0.5, M0=np.pi/2, omega=np.pi/4, i=np.pi/4, Omega=np.pi/3, mu_alpha=0.0, mu_delta=0.0, dalpha=0.0, ddelta=0.0, distance=10.0)
    t = np.linspace(0, 10, 100)
    psi = np.random.uniform(0, 2 * np.pi, size=len(t))
    a = model.gaia_astrometry(t, spsi=np.sin(psi), cpsi=np.cos(psi), par_factor=1, system=1)
    data = GaiaData(spsi=np.sin(psi), cpsi=np.cos(psi), t=t, plx_fac=np.random.uniform(0.5, 1.5, size=len(t)), x=a, err=0.01*np.ones(len(t)), system=1)
    results = fitter.fit(data)
    assert isinstance(results, FitResults)


class DummyData:
    def __init__(self, t, x, y, x_err, y_err, ref_epoch, mu_x, mu_y):
        self.t = np.asarray(t)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.x_err = np.asarray(x_err)
        self.y_err = np.asarray(y_err)
        self.ref_epoch = ref_epoch
        self.mu_x = mu_x
        self.mu_y = mu_y


class DummyFitter(Fitter):
    def fit(self, data):
        return data


def test_proper_motion_fit_uses_provided_mu_values():
    fitter = DummyFitter()
    data = DummyData(
        t=[0.0, 1.0, 2.0],
        x=[5.0, 7.0, 9.0],
        y=[-3.0, -4.0, -5.0],
        x_err=[1.0, 1.0, 1.0],
        y_err=[1.0, 1.0, 1.0],
        ref_epoch=1.0,
        mu_x=2.0,
        mu_y=-1.0,
    )

    result = fitter._proper_motion_fit(data)

    assert result["params"]["x0"] == 7.0
    assert result["params"]["y0"] == -4.0
    assert result["params"]["mu_x"] == 2.0
    assert result["params"]["mu_y"] == -1.0


# The sampler doubles below replace only the expensive external execution
# engines. They still call the fitters' real prior transforms, likelihoods,
# Orbit models, Data.chi2 methods, posterior assembly, and FitResults code.
class ProbeComplete(RuntimeError):
    pass


class FastEmceeSampler:
    instances = []

    def __init__(
        self,
        nwalkers,
        ndim,
        log_prob_fn,
        args=(),
        pool=None,
    ):
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.args = args
        self.pool = pool
        self.acceptance_fraction = np.full(nwalkers, 0.5)
        self.run_arguments = None
        type(self).instances.append(self)

    def run_mcmc(self, initial, niter, progress):
        initial = np.asarray(initial, dtype=float)
        assert initial.shape == (self.nwalkers, self.ndim)
        self.run_arguments = (initial.copy(), niter, progress)
        self.chain = np.repeat(initial[np.newaxis, :, :], niter, axis=0)
        initial_log_prob = np.array(
            [
                self.log_prob_fn(position, *self.args)
                for position in initial
            ]
        )
        self.log_prob = np.repeat(
            initial_log_prob[np.newaxis, :],
            niter,
            axis=0,
        )

    def get_chain(self, discard=0, thin=1, flat=False):
        if thin < 1:
            raise ValueError("thin must be at least one")
        selected = self.chain[discard::thin]
        if flat:
            return selected.reshape(-1, self.ndim)
        return selected

    def get_log_prob(self, discard=0, thin=1, flat=False):
        if thin < 1:
            raise ValueError("thin must be at least one")
        selected = self.log_prob[discard::thin]
        if flat:
            return selected.reshape(-1)
        return selected


class ProbeEmceeSampler(FastEmceeSampler):
    value = None

    def run_mcmc(self, initial, niter, progress):
        initial = np.asarray(initial, dtype=float)
        type(self).value = self.log_prob_fn(initial[0], *self.args)
        raise ProbeComplete


class FastNestedSampler:
    instances = []

    def __init__(
        self,
        param_names,
        loglike,
        transform,
        derived_param_names=(),
        wrapped_params=None,
    ):
        self.param_names = tuple(param_names)
        self.loglike = loglike
        self.transform = transform
        self.derived_param_names = tuple(derived_param_names)
        self.wrapped_params = wrapped_params
        self.run_kwargs = None
        self.cubes = None
        self.transformed_samples = None
        self.logl = None
        type(self).instances.append(self)

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        ndim = len(self.param_names)
        levels = (0.25, 0.5, 0.75)
        self.cubes = np.array(
            [np.full(ndim, level, dtype=float) for level in levels]
        )
        self.transformed_samples = np.array(
            [self.transform(cube) for cube in self.cubes]
        )
        self.logl = np.array(
            [self.loglike(sample) for sample in self.transformed_samples]
        )
        return {
            "samples": self.transformed_samples,
            "weighted_samples": {"logl": self.logl},
            "ess": 23,
            "logz": -4.5,
            "logzerr": 0.2,
        }


class ProbeNestedSampler(FastNestedSampler):
    transformed = None
    value = None

    def run(self, **kwargs):
        cube = np.full(len(self.param_names), 0.5)
        type(self).transformed = self.transform(cube)
        type(self).value = self.loglike(type(self).transformed)
        raise ProbeComplete


def deterministic_initial(values):
    calls = []

    class DeterministicInitial(InitialGuess):
        def get_initial_guess(self, param_order, nwalkers):
            calls.append((tuple(param_order), nwalkers))
            row = [values[name] for name in param_order]
            return np.tile(row, (nwalkers, 1))

    return DeterministicInitial, calls


def make_exact_rv_problem():
    truth = {
        "P": 4.0,
        "e": 0.2,
        "Tp": 0.7,
        "K1": 3.0,
        "omega1": 0.4,
        "systemic_velocity": 1.25,
        "Tepoch": 0.0,
    }
    orbit = Orbit(**truth)
    times = np.linspace(0.0, 12.0, 31)
    data = RadialVelocityData(
        times,
        orbit.rv(times, system="1"),
        rv_err=0.2,
        system="1",
    )
    priors = {
        "P": UniformPrior(3.0, 5.0),
        "e": UniformPrior(0.1, 0.3),
        "Tp": UniformPrior(0.2, 1.2),
        "K1": UniformPrior(2.0, 4.0),
        "omega1": UniformPrior(-0.1, 0.9),
        "systemic_velocity": FixedPrior(truth["systemic_velocity"]),
        "Tepoch": FixedPrior(truth["Tepoch"]),
    }
    sampled = ("P", "e", "Tp", "K1", "omega1")
    return data, truth, priors, sampled


def make_exact_astrometry_problem(ref_epoch=0.0, periastron_time=1.3):
    truth = {
        "P": 4.0,
        "e": 0.35,
        "Tp": periastron_time,
        "Tepoch": ref_epoch,
        "A1": 1.2,
        "B1": -0.7,
        "F1": 0.3,
        "G1": 0.8,
        "dx": 2.0,
        "dy": -1.0,
        "dpmra": 0.05,
        "dpmdec": -0.03,
    }
    times = ref_epoch + np.linspace(0.0, 13.0, 37)
    orbit = Orbit(**truth)
    x, y = orbit.astrometry(times, system="1")
    data = AstrometryData(
        times,
        x,
        y,
        x_err=0.05,
        y_err=0.08,
        ref_epoch=ref_epoch,
        system="1",
    )
    priors = {
        "P": UniformPrior(3.0, 5.0),
        "e": UniformPrior(0.25, 0.45),
        "Tp": UniformPrior(periastron_time - 0.5, periastron_time + 0.5),
    }
    return data, truth, priors


def make_exact_gaia_problem():
    truth = {
        "P": 4.0,
        "e": 0.2,
        "Tp": 1.0,
        "a1": 1.3,
        "cosi": 0.6,
        "omega1": 0.5,
        "Omega": 0.8,
        "dalpha": 0.4,
        "ddelta": -0.2,
        "mu_alpha": 0.03,
        "mu_delta": -0.02,
        "parallax": 0.1,
    }
    times = np.linspace(0.0, 10.0, 41)
    psi = np.mod(np.arange(times.size) * 2.399963229728653, 2.0 * np.pi)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    plx_fac = np.sin(0.73 * times) + 0.2 * np.cos(1.31 * times)
    orbit = Orbit(**truth)
    values = orbit.gaia_astrometry(
        times,
        spsi,
        cpsi,
        plx_fac,
        system="1",
    )
    data = GaiaData(
        spsi=spsi,
        cpsi=cpsi,
        t=times,
        plx_fac=plx_fac,
        x=values,
        err=0.1,
        system="1",
    )
    priors = {
        "P": UniformPrior(3.0, 5.0),
        "e": UniformPrior(0.1, 0.3),
        "Tp": UniformPrior(0.5, 1.5),
    }
    return data, truth, priors


def test_proper_motion_fit_recovers_weighted_linear_motion():
    """Highlight regressions in the weighted free-slope proper-motion baseline."""
    fitter = DummyFitter()
    times = np.array([3.0, 4.0, 7.0, 9.0, 12.0])
    ref_epoch = 7.0
    dt = times - ref_epoch
    data = DummyData(
        t=times,
        x=2.5 + 0.4 * dt,
        y=-1.25 - 0.2 * dt,
        x_err=[0.2, 0.4, 0.3, 0.5, 0.25],
        y_err=[0.3, 0.2, 0.5, 0.4, 0.25],
        ref_epoch=ref_epoch,
        mu_x=None,
        mu_y=None,
    )

    result = fitter._proper_motion_fit(data)

    assert result["params"] == pytest.approx(
        {"x0": 2.5, "mu_x": 0.4, "y0": -1.25, "mu_y": -0.2}
    )
    assert result["chi2"] == pytest.approx(0.0, abs=1e-24)
    assert result["dof"] == 2 * len(times) - 4


def test_astrometric_offset_seeds_match_proper_motion_fit():
    """Ensure initializer offset seeds remain consistent with the baseline fit."""
    fitter = DummyFitter()
    data, _truth, _priors = make_exact_astrometry_problem()

    seeds = fitter._astrometric_offset_seeds(data)
    fit = fitter._proper_motion_fit(data)["params"]

    assert seeds == {
        "dx": fit["x0"],
        "dy": fit["y0"],
        "dpmra": fit["mu_x"],
        "dpmdec": fit["mu_y"],
    }


@pytest.mark.parametrize(
    ("nwalkers", "niter"),
    [(0, 10), (-1, 10), (4, 0), (4, -1)],
)
def test_mcmc_rejects_nonpositive_sampler_sizes(nwalkers, niter):
    """Highlight missing validation for unusable walker and iteration counts.

    Bug location: ``src/periapsis/fitting/mcmc.py:17-18`` stores both values
    without first requiring them to be positive.
    """
    with pytest.raises(ValueError):
        MCMCFitter(
            nwalkers=nwalkers,
            niter=niter,
            sample_params=["P"],
            P=UniformPrior(1.0, 2.0),
        )


def test_mcmc_rejects_duplicate_sampled_parameters():
    """Ensure duplicate names cannot silently corrupt MCMC column mappings."""
    with pytest.raises(ValueError, match="unique"):
        MCMCFitter(
            nwalkers=4,
            niter=10,
            sample_params=["P", "P"],
            P=UniformPrior(1.0, 2.0),
        )


def test_mcmc_rejects_empty_sampled_parameters():
    """Ensure MCMC cannot be configured with a zero-dimensional state."""
    with pytest.raises(ValueError, match="At least one"):
        MCMCFitter(
            nwalkers=4,
            niter=10,
            sample_params=[],
        )


def test_mcmc_rejects_sampled_parameter_without_prior():
    """Highlight sampled dimensions that are unconstrained by any prior."""
    with pytest.raises(ValueError, match="missing priors"):
        MCMCFitter(
            nwalkers=4,
            niter=10,
            sample_params=["P", "e"],
            P=UniformPrior(1.0, 2.0),
        )


def test_mcmc_rejects_fixed_sampled_parameter():
    """Ensure a fixed parameter is not also exposed as a sampled dimension."""
    with pytest.raises(ValueError, match="cannot be fixed"):
        MCMCFitter(
            nwalkers=4,
            niter=10,
            sample_params=["P"],
            P=FixedPrior(2.0),
        )


def test_mcmc_rejects_nonfixed_prior_that_cannot_enter_posterior():
    """Highlight nonfixed priors that would otherwise be silently ignored.

    Bug location: ``src/periapsis/fitting/mcmc.py:163-171`` only records
    reachable priors and silently drops every other nonfixed prior.
    """
    with pytest.warns(UserWarning, match="e"):
        MCMCFitter(
            nwalkers=4,
            niter=10,
            sample_params=["P"],
            P=UniformPrior(1.0, 2.0),
            e=UniformPrior(0.0, 0.9),
        )


def test_mcmc_linear_rejects_duplicate_sampled_parameters():
    """Highlight duplicate linear-MCMC dimensions and ambiguous output columns.

    Bug location: ``src/periapsis/fitting/mcmclinear.py:27-28`` collapses the
    names into a frozenset without checking whether the input had duplicates.
    """
    with pytest.raises(ValueError, match="Duplicate"):
        MCMCLinearFitter(
            nwalkers=8,
            niter=10,
            sampled_params=("P", "e", "Tp", "P"),
            P=UniformPrior(3.0, 5.0),
            e=UniformPrior(0.1, 0.3),
            Tp=UniformPrior(0.0, 1.0),
        )


def test_ultranest_rejects_duplicate_output_parameters():
    """Ensure duplicate nested outputs cannot be hidden by set conversion.

    Bug location: ``src/periapsis/fitting/ultranest.py:18-19`` stores both a
    deduplicated set and the original tuple without rejecting duplicates.
    """
    with pytest.raises(ValueError, match="Duplicate"):
        UltranestFitter(
            output_params=("P", "P"),
            P=UniformPrior(1.0, 2.0),
        )


def test_gaia_mcmc_accepts_list_sample_order_when_adding_jitter():
    """Highlight list/tuple concatenation failures when jitter is auto-added.

    Bug location: ``src/periapsis/fitting/gaia_mcmclinear.py:36`` concatenates
    a tuple onto the caller's potentially list-valued ``sampled_params``.
    """
    fitter = MCMCGaiaFitter(
        nwalkers=8,
        niter=10,
        sampled_params=["P", "e", "Tp"],
        P=UniformPrior(3.0, 5.0),
        e=UniformPrior(0.1, 0.3),
        Tp=UniformPrior(0.5, 1.5),
    )

    assert fitter.param_order == ("P", "e", "Tp", "jitter")


def test_gaia_ultranest_accepts_list_output_order_when_adding_jitter():
    """Highlight list/tuple concatenation failures in Gaia nested setup.

    Bug location: ``src/periapsis/fitting/gaia_ultranestlinear.py:39``
    concatenates a tuple onto the caller's potentially list-valued outputs.
    """
    fitter = UltranestGaiaFitter(
        output_params=["P", "e", "Tp"],
        P=UniformPrior(3.0, 5.0),
        e=UniformPrior(0.1, 0.3),
        Tp=UniformPrior(0.5, 1.5),
    )

    assert fitter.output_param_order == ("P", "e", "Tp", "jitter")


def test_gaia_mcmc_does_not_duplicate_explicit_jitter_sample():
    """Ensure an explicitly sampled jitter parameter is not appended twice.

    Bug location: ``src/periapsis/fitting/gaia_mcmclinear.py:35-36`` appends
    ``jitter`` whenever it is nonfixed, even when it is already requested.
    """
    fitter = MCMCGaiaFitter(
        nwalkers=8,
        niter=10,
        sampled_params=("P", "e", "Tp", "jitter"),
        P=UniformPrior(3.0, 5.0),
        e=UniformPrior(0.1, 0.3),
        Tp=UniformPrior(0.5, 1.5),
        jitter=UniformPrior(0.01, 0.05),
    )

    assert fitter.param_order.count("jitter") == 1


@pytest.mark.parametrize(
    "fitter_class",
    [MCMCGaiaFitter, UltranestGaiaFitter],
)
def test_gaia_fitters_preserve_user_supplied_jitter_prior(fitter_class):
    """Highlight user jitter priors being overwritten by the default prior.

    Bug locations: ``src/periapsis/fitting/gaia_mcmclinear.py:32-37`` and
    ``src/periapsis/fitting/gaia_ultranestlinear.py:34-40`` first retain the
    supplied prior and then replace it with the default log-uniform prior.
    """
    jitter_prior = UniformPrior(0.01, 0.05)
    common = {
        "P": UniformPrior(3.0, 5.0),
        "e": UniformPrior(0.1, 0.3),
        "Tp": UniformPrior(0.5, 1.5),
    }
    if fitter_class is MCMCGaiaFitter:
        fitter = fitter_class(
            nwalkers=8,
            niter=10,
            sampled_params=("P", "e", "Tp"),
            jitter=jitter_prior,
            **common,
        )
    else:
        fitter = fitter_class(
            output_params=("P", "e", "Tp"),
            jitter=jitter_prior,
            **common,
        )

    assert fitter.priors["jitter"] is jitter_prior


def test_gaia_mcmc_rejects_non_gaia_data_with_clear_error():
    """Ensure unsupported data fail at the API boundary with a clear error.

    Bug location: ``src/periapsis/fitting/gaia_mcmclinear.py:51-58`` uses Gaia
    fields before performing any explicit ``GaiaData`` type validation.
    """
    data, _truth, priors, _sampled = make_exact_rv_problem()
    fitter = MCMCGaiaFitter(
        nwalkers=8,
        niter=10,
        sampled_params=("P", "e", "Tp"),
        jitter=0.1,
        P=priors["P"],
        e=priors["e"],
        Tp=priors["Tp"],
    )

    with pytest.raises(ValueError, match="GaiaData"):
        fitter.fit(data, rng=np.random.default_rng(0))


def test_mcmc_prior_sampling_resamples_derived_bound_violations():
    """Verify rejection sampling reapplies bounds on transformed parameters."""
    class SequencedPrior:
        def __init__(self):
            self.calls = []

        def sample(self, rng, size=1):
            self.calls.append(size)
            if len(self.calls) == 1:
                return np.array([1.0, 2.0, 4.0])
            return np.full(size, 2.0)

    period_prior = SequencedPrior()
    fitter = MCMCFitter(
        nwalkers=3,
        niter=2,
        sample_params=("P",),
        P=period_prior,
        n=Bounds(lower=2.0, upper=4.0),
    )

    samples = fitter._sample_priors(
        fitter.param_order,
        size=3,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(samples[:, 0], [2.0, 2.0, 2.0])
    assert period_prior.calls == [3, 2]


def test_mcmc_log_posterior_uses_real_orbit_likelihood_and_all_direct_priors():
    """Ensure the real RV posterior sums every direct prior and likelihood."""
    data, truth, priors, sampled = make_exact_rv_problem()
    fitter = MCMCFitter(
        nwalkers=10,
        niter=4,
        sample_params=sampled,
        **priors,
    )
    context = fitter._posterior_context(data)
    parameters = np.array([truth[name] for name in sampled])

    value = fitter._log_posterior(parameters, context)
    expected_log_prior = sum(
        priors[name].logpdf(truth[name])
        for name in sampled
    )

    assert data.chi2(Orbit(**truth)) == pytest.approx(0.0, abs=1e-24)
    assert value == pytest.approx(expected_log_prior, abs=1e-10)


def test_mcmc_log_posterior_rejects_direct_prior_violation_before_likelihood():
    """Highlight invalid direct-prior values that must short-circuit to -inf."""
    data, truth, priors, sampled = make_exact_rv_problem()
    fitter = MCMCFitter(
        nwalkers=10,
        niter=4,
        sample_params=sampled,
        **priors,
    )
    context = fitter._posterior_context(data)
    parameters = np.array([truth[name] for name in sampled])
    parameters[sampled.index("e")] = 0.95

    assert fitter._log_posterior(parameters, context) == -np.inf


def test_mcmc_log_posterior_evaluates_prior_on_derived_parameter():
    """Ensure priors on transformed parameters contribute to the posterior."""
    data, truth, _priors, _sampled = make_exact_rv_problem()
    mean_motion_prior = UniformPrior(1.0, 2.0)
    fitter = MCMCFitter(
        nwalkers=4,
        niter=2,
        sample_params=("P",),
        n=mean_motion_prior,
        e=FixedPrior(truth["e"]),
        Tp=FixedPrior(truth["Tp"]),
        K1=FixedPrior(truth["K1"]),
        omega1=FixedPrior(truth["omega1"]),
        systemic_velocity=FixedPrior(truth["systemic_velocity"]),
        Tepoch=FixedPrior(truth["Tepoch"]),
    )
    context = fitter._posterior_context(data)

    assert [name for name, _prior in context.derived_prior_items] == ["n"]
    value = fitter._log_posterior(np.array([truth["P"]]), context)
    assert value == pytest.approx(
        mean_motion_prior.logpdf(2.0 * np.pi / truth["P"]),
        abs=1e-10,
    )


def test_mcmc_fit_integrates_initialization_sampler_and_results(
    monkeypatch,
):
    """Exercise the full MCMC flow and expose dropped sampler/result metadata.

    Bug location: ``src/periapsis/fitting/mcmc.py:258`` explicitly stores
    ``None`` as ``raw_sampler`` instead of preserving the completed sampler.
    """
    data, truth, priors, sampled = make_exact_rv_problem()
    initial_class, initial_calls = deterministic_initial(truth)
    FastEmceeSampler.instances = []
    monkeypatch.setattr(
        mcmc_module.emcee,
        "EnsembleSampler",
        FastEmceeSampler,
    )
    monkeypatch.setattr(
        mcmc_module.emcee.autocorr,
        "integrated_time",
        lambda chain, quiet: np.full(chain.shape[-1], 0.5),
    )
    pool = object()
    fitter = MCMCFitter(
        nwalkers=10,
        niter=4,
        sample_params=sampled,
        pool=pool,
        **priors,
    )

    results = fitter.fit(
        data,
        rng=np.random.default_rng(5),
        initial=initial_class,
    )

    sampler = FastEmceeSampler.instances[-1]
    assert isinstance(results, FitResults)
    assert initial_calls == [(sampled, 10)]
    assert sampler.pool is pool
    assert sampler.run_arguments[1:] == (4, True)
    assert np.isfinite(sampler.log_prob).all()
    assert results.backend == "emcee"
    assert results.param_names == sampled
    assert results.priors is fitter.priors
    assert results.PM_fit is None
    assert results.mean_acceptance_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(results.tau, np.full(len(sampled), 0.5))
    for name in sampled:
        np.testing.assert_allclose(results[name], truth[name])
        assert results.MAP_params[name] == pytest.approx(truth[name])
        assert results.median_params[name] == pytest.approx(truth[name])
    assert results.raw_samples is sampler


def test_mcmc_fit_recovers_when_autocorrelation_time_is_not_finite(
    monkeypatch,
):
    """Highlight crashes when autocorrelation estimation returns only NaNs.

    Bug location: ``src/periapsis/fitting/mcmc.py:227-232`` converts the
    extrema of an all-NaN autocorrelation estimate directly to integers.
    """
    data, truth, priors, sampled = make_exact_rv_problem()
    initial_class, _calls = deterministic_initial(truth)
    monkeypatch.setattr(
        mcmc_module.emcee,
        "EnsembleSampler",
        FastEmceeSampler,
    )
    monkeypatch.setattr(
        mcmc_module.emcee.autocorr,
        "integrated_time",
        lambda chain, quiet: np.full(chain.shape[-1], np.nan),
    )
    fitter = MCMCFitter(
        nwalkers=10,
        niter=4,
        sample_params=sampled,
        **priors,
    )

    with pytest.warns(UserWarning, match="Autocorrelation"):
        with pytest.warns(RuntimeWarning, match="All-NaN"):
            results = fitter.fit(
                data,
                rng=np.random.default_rng(5),
                initial=initial_class,
            )

    assert isinstance(results, FitResults)
    assert results["P"].size == 40


def test_mcmc_astrometry_fit_carries_proper_motion_baseline(
    monkeypatch,
):
    """Ensure astrometric MCMC results retain the proper-motion comparison fit."""
    data, truth, orbital_priors = make_exact_astrometry_problem()
    sampled = ("P", "e", "Tp", "A1", "B1", "F1", "G1", "dx", "dy", "dpmra", "dpmdec")
    priors = {
        **orbital_priors,
        "A1": UniformPrior(0.0, 2.0),
        "B1": UniformPrior(-1.5, 0.0),
        "F1": UniformPrior(0.0, 1.0),
        "G1": UniformPrior(0.0, 1.5),
        "dx": UniformPrior(1.0, 3.0),
        "dy": UniformPrior(-2.0, 0.0),
        "dpmra": UniformPrior(0.0, 0.1),
        "dpmdec": UniformPrior(-0.08, 0.02),
        "Tepoch": FixedPrior(truth["Tepoch"]),
    }
    initial_class, _calls = deterministic_initial(truth)
    monkeypatch.setattr(
        mcmc_module.emcee,
        "EnsembleSampler",
        FastEmceeSampler,
    )
    monkeypatch.setattr(
        mcmc_module.emcee.autocorr,
        "integrated_time",
        lambda chain, quiet: np.full(chain.shape[-1], 0.5),
    )
    fitter = MCMCFitter(
        nwalkers=24,
        niter=3,
        sample_params=sampled,
        **priors,
    )

    results = fitter.fit(
        data,
        rng=np.random.default_rng(1),
        initial=initial_class,
    )

    expected_pm = fitter._proper_motion_fit(data)
    assert results.PM_fit["chi2"] == pytest.approx(expected_pm["chi2"])
    assert results.PM_fit["dof"] == expected_pm["dof"]
    assert results.PM_fit["params"] == pytest.approx(expected_pm["params"])


def test_mcmc_linear_likelihood_uses_absolute_periastron_time(
    monkeypatch,
):
    """Highlight erroneous scaling of absolute Tp by the orbital period.

    Bug locations: ``src/periapsis/fitting/mcmclinear.py:99`` and
    ``src/periapsis/fitting/mcmclinear.py:187`` use ``Tp * P`` although
    ``Tp`` is already an absolute time.
    """
    data, truth, priors = make_exact_astrometry_problem(
        ref_epoch=0.0,
        periastron_time=1.3,
    )
    initial_class, _calls = deterministic_initial(truth)
    ProbeEmceeSampler.value = None
    monkeypatch.setattr(
        mcmc_linear_module.emcee,
        "EnsembleSampler",
        ProbeEmceeSampler,
    )
    fitter = MCMCLinearFitter(
        nwalkers=8,
        niter=2,
        sampled_params=("P", "e", "Tp"),
        **priors,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(
            data,
            rng=np.random.default_rng(0),
            initial=initial_class,
        )

    expected_log_prior = sum(
        prior.logpdf(truth[name])
        for name, prior in priors.items()
    )
    assert ProbeEmceeSampler.value == pytest.approx(
        expected_log_prior,
        abs=1e-8,
    )


def test_mcmc_linear_passes_caller_parameter_order_to_initialization(
    monkeypatch,
):
    """Ensure set iteration cannot reorder initializer and sampler columns.

    Bug locations: ``src/periapsis/fitting/mcmclinear.py:41`` reconstructs
    column order from a frozenset, and line 114 repeats that unordered mapping.
    """
    data, truth, priors = make_exact_astrometry_problem()
    requested_order = ("Tp", "q", "P", "e")
    truth_with_extra = {**truth, "q": 0.5}
    priors_with_extra = {
        **priors,
        "q": UniformPrior(0.0, 1.0),
    }
    initial_class, calls = deterministic_initial(truth_with_extra)
    monkeypatch.setattr(
        mcmc_linear_module.emcee,
        "EnsembleSampler",
        ProbeEmceeSampler,
    )
    fitter = MCMCLinearFitter(
        nwalkers=8,
        niter=2,
        sampled_params=requested_order,
        **priors_with_extra,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(
            data,
            rng=np.random.default_rng(0),
            initial=initial_class,
        )

    assert calls == [(requested_order, 8)]


def test_ultranest_requires_priors_to_cover_every_output():
    """Ensure every requested nested output is constrained by supplied priors."""
    with pytest.raises(ValueError, match="Missing priors"):
        UltranestFitter(
            output_params=("P", "e"),
            P=UniformPrior(1.0, 2.0),
        )


def test_ultranest_rejects_overconstrained_prior_parameterizations():
    """Highlight contradictory priors on mutually derivable parameters."""
    with pytest.raises(ValueError, match="Overconstrained"):
        UltranestFitter(
            output_params=("P",),
            P=UniformPrior(3.0, 5.0),
            n=UniformPrior(1.0, 2.0),
        )


def test_ultranest_sample_order_follows_prior_insertion_order():
    """Ensure nested sample columns preserve deterministic prior ordering.

    Bug location: ``src/periapsis/fitting/ultranest.py:34-35`` builds
    ``sample_order`` from a set, discarding the caller's insertion order.
    """
    _data, _truth, priors, sampled = make_exact_rv_problem()

    fitter = UltranestFitter(
        output_params=sampled,
        **priors,
    )

    assert fitter.sample_order == sampled


@pytest.mark.parametrize(
    "bound",
    [Bounds(lower=1.0), Bounds(upper=2.0)],
)
def test_ultranest_likelihood_supports_one_sided_derived_bounds(
    monkeypatch,
    bound,
):
    """Highlight comparisons against None for valid one-sided Bounds priors.

    Bug location: ``src/periapsis/fitting/ultranest.py:71`` performs one
    chained comparison even when either bound endpoint is ``None``.
    """
    data, _truth, priors, sampled = make_exact_rv_problem()
    ProbeNestedSampler.value = None
    monkeypatch.setattr(
        ultranest_module.ultranest,
        "ReactiveNestedSampler",
        ProbeNestedSampler,
    )
    fitter = UltranestFitter(
        output_params=sampled,
        n=bound,
        **priors,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(data, quiet=True)

    assert np.isfinite(ProbeNestedSampler.value)


def test_ultranest_fit_integrates_transform_likelihood_and_results(
    monkeypatch,
):
    """Exercise nested transforms through results and expose lost diagnostics.

    Bug locations: ``src/periapsis/fitting/ultranest.py:133`` writes ``ESS``,
    while ``src/periapsis/fitting/results.py:22`` only consumes ``Ess``.
    """
    data, _truth, priors, sampled = make_exact_rv_problem()
    FastNestedSampler.instances = []
    monkeypatch.setattr(
        ultranest_module.ultranest,
        "ReactiveNestedSampler",
        FastNestedSampler,
    )
    fitter = UltranestFitter(
        output_params=sampled,
        min_num_live_points=20,
        min_ess=15,
        dlogz=0.2,
        dKL=0.3,
        frac_remain=0.05,
        Lepsilon=0.01,
        max_iters=50,
        max_ncalls=100,
        **priors,
    )

    results = fitter.fit(data, quiet=True)

    sampler = FastNestedSampler.instances[-1]
    assert isinstance(results, FitResults)
    assert np.all(np.isfinite(sampler.logl))
    assert results.raw_samples is sampler
    assert results.backend == "ultranest"
    assert results.fit_method == "Campbell"
    assert results.Ess == 23
    assert sampler.run_kwargs == {
        "min_num_live_points": 20,
        "min_ess": 15,
        "dlogz": 0.2,
        "dKL": 0.3,
        "frac_remain": 0.05,
        "Lepsilon": 0.01,
        "max_iters": 50,
        "max_ncalls": 100,
        "show_status": False,
        "viz_callback": False,
    }
    best_index = int(np.argmax(sampler.logl))
    for index, name in enumerate(sampler.param_names):
        np.testing.assert_allclose(
            results[name],
            sampler.transformed_samples[:, index],
        )
        assert results.MAP_params[name] == pytest.approx(
            sampler.transformed_samples[best_index, index]
        )


def test_ultranest_results_persist_requested_derived_outputs(
    monkeypatch,
):
    """Ensure requested derived outputs are stored as named result columns.

    Bug location: ``src/periapsis/fitting/ultranest.py:123-136`` assembles
    result mappings from sampled names only, omitting derived output columns.
    """
    data, truth, _priors, _sampled = make_exact_rv_problem()
    priors = {
        "n": UniformPrior(1.0, 2.0),
        "e": FixedPrior(truth["e"]),
        "Tp": FixedPrior(truth["Tp"]),
        "K1": FixedPrior(truth["K1"]),
        "omega1": FixedPrior(truth["omega1"]),
        "systemic_velocity": FixedPrior(truth["systemic_velocity"]),
        "Tepoch": FixedPrior(truth["Tepoch"]),
    }
    monkeypatch.setattr(
        ultranest_module.ultranest,
        "ReactiveNestedSampler",
        FastNestedSampler,
    )
    fitter = UltranestFitter(
        output_params=("P",),
        **priors,
    )

    results = fitter.fit(data, quiet=True)

    assert "P" in results.samples
    np.testing.assert_allclose(
        results.samples["P"],
        2.0 * np.pi / results.samples["n"],
    )


def test_ultranest_linear_likelihood_respects_nonzero_reference_epoch(
    monkeypatch,
):
    """Highlight double subtraction of reference epoch in linear likelihoods.

    Bug location: ``src/periapsis/fitting/ultranestlinear.py:112-115``
    subtracts ``ref_epoch`` from data times before subtracting absolute ``Tp``.
    """
    data, truth, priors = make_exact_astrometry_problem(
        ref_epoch=10.0,
        periastron_time=11.3,
    )
    ProbeNestedSampler.value = None
    monkeypatch.setattr(
        ultranest_linear_module.ultranest,
        "ReactiveNestedSampler",
        ProbeNestedSampler,
    )
    fitter = UltranestLinearFitter(
        output_params=("P", "e", "Tp"),
        **priors,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(data, quiet=True)

    assert ProbeNestedSampler.value == pytest.approx(0.0, abs=1e-8)


def test_ultranest_linear_fit_reports_sampler_ess_and_valid_counts(
    monkeypatch,
):
    """Ensure linear nested results retain ESS and posterior filtering counts.

    Bug locations: ``src/periapsis/fitting/ultranestlinear.py:220`` writes
    ``ESS``, while ``src/periapsis/fitting/results.py:22`` expects ``Ess``.
    """
    data, _truth, priors = make_exact_astrometry_problem()
    monkeypatch.setattr(
        ultranest_linear_module.ultranest,
        "ReactiveNestedSampler",
        FastNestedSampler,
    )
    fitter = UltranestLinearFitter(
        output_params=("P", "e", "Tp"),
        **priors,
    )

    results = fitter.fit(data, quiet=True)

    assert results.Ess == 23
    assert results.samples["n_samples_raw"] == 3
    assert results.samples["n_samples_valid"] == 3
    assert results.samples["samples"].shape == (3, 11)


def test_gaia_mcmc_objective_accumulates_every_prior_term(
    monkeypatch,
):
    """Highlight Gaia MCMC overwriting, rather than summing, prior terms.

    Bug location: ``src/periapsis/fitting/gaia_mcmclinear.py:129`` assigns
    each prior log-density to ``lp`` instead of accumulating it.
    """
    data, truth, priors = make_exact_gaia_problem()
    ProbeEmceeSampler.value = None

    def exact_initial(_self, param_order, nwalkers):
        row = [truth[name] for name in param_order]
        return np.tile(row, (nwalkers, 1))

    monkeypatch.setattr(
        gaia_mcmc_module.GaiaInitialGuess,
        "get_initial_guess",
        exact_initial,
    )
    monkeypatch.setattr(
        gaia_mcmc_module.emcee,
        "EnsembleSampler",
        ProbeEmceeSampler,
    )
    fitter = MCMCGaiaFitter(
        nwalkers=10,
        niter=2,
        sampled_params=("P", "e", "Tp"),
        jitter=0.1,
        **priors,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(data, rng=np.random.default_rng(0))

    expected = sum(
        prior.logpdf(truth[name])
        for name, prior in priors.items()
    )
    assert ProbeEmceeSampler.value == pytest.approx(expected, abs=1e-8)


def test_gaia_ultranest_can_fit_linear_nuisance_terms_without_fixed_copies(
    monkeypatch,
):
    """Expose mismatched Gaia nuisance names passed from the solve to Orbit.

    Bug locations: ``src/periapsis/fitting/gaia_ultranestlinear.py:72`` and
    lines 161-164 use ``delta_alpha``/``delta_delta``, while
    ``src/periapsis/model/orbit.py:28-30`` requires ``dalpha``/``ddelta``.
    """
    data, _truth, priors = make_exact_gaia_problem()
    ProbeNestedSampler.value = None
    monkeypatch.setattr(
        gaia_ultranest_module.ultranest,
        "ReactiveNestedSampler",
        ProbeNestedSampler,
    )
    fitter = UltranestGaiaFitter(
        output_params=("P", "e", "Tp"),
        jitter=0.1,
        **priors,
    )

    with pytest.raises(ProbeComplete):
        fitter.fit(data, quiet=True)

    assert np.isfinite(ProbeNestedSampler.value)


def test_gaia_ultranest_single_motion_uses_time_before_parallax_factor(
    monkeypatch,
):
    """Highlight swapped time and parallax-factor arguments in the baseline fit.

    Bug location: ``src/periapsis/fitting/gaia_ultranestlinear.py:175`` passes
    the arguments opposite to ``src/periapsis/utils/solvers.py:208``.
    """
    data, truth, priors = make_exact_gaia_problem()
    full_priors = {
        **priors,
        "a1": FixedPrior(truth["a1"]),
        "cosi": FixedPrior(truth["cosi"]),
        "omega1": FixedPrior(truth["omega1"]),
        "Omega": FixedPrior(truth["Omega"]),
        "dalpha": FixedPrior(truth["dalpha"]),
        "ddelta": FixedPrior(truth["ddelta"]),
        "mu_alpha": FixedPrior(truth["mu_alpha"]),
        "mu_delta": FixedPrior(truth["mu_delta"]),
        "parallax": FixedPrior(truth["parallax"]),
    }
    monkeypatch.setattr(
        gaia_ultranest_module.ultranest,
        "ReactiveNestedSampler",
        FastNestedSampler,
    )
    fitter = UltranestGaiaFitter(
        output_params=("P", "e", "Tp"),
        jitter=0.1,
        **full_priors,
    )

    results = fitter.fit(data, quiet=True)
    expected = gaia_single_motion(
        data.spsi,
        data.cpsi,
        data.t,
        data.plx_fac,
        data.x,
        data.err,
    )

    np.testing.assert_allclose(
        results.Single_motion_params["mu"],
        expected["mu"],
    )
    np.testing.assert_allclose(
        results.Single_motion_params["mu_err"],
        expected["mu_err"],
    )
    assert results.Single_motion_params["chi2"] == pytest.approx(
        expected["chi2"]
    )
    assert results.Single_motion_params["dof"] == expected["dof"]
    assert results.samples["n_samples_raw"] == 3
    assert results.samples["n_samples_valid"] == 3


def test_gaia_ultranest_rejects_all_samples_outside_linear_bounds(
    monkeypatch,
):
    """Ensure finite rejection sentinels cannot enter the posterior as samples."""
    data, _truth, priors = make_exact_gaia_problem()
    # The exact data have parallax 0.1; this bound rejects every linear solve.
    constrained_priors = {
        **priors,
        "parallax": Bounds(lower=10.0, upper=20.0),
    }
    monkeypatch.setattr(
        gaia_ultranest_module.ultranest,
        "ReactiveNestedSampler",
        FastNestedSampler,
    )
    fitter = UltranestGaiaFitter(
        output_params=("P", "e", "Tp"),
        jitter=0.1,
        **constrained_priors,
    )

    with pytest.raises(RuntimeError, match="No valid posterior samples"):
        fitter.fit(data, quiet=True)
