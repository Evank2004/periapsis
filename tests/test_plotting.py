import numpy as np
from astropy import units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from periapsis.data.common import AstrometryData, RadialVelocityData
from periapsis.data.gaia import GaiaData
from periapsis.data.joint_data import JointData
from periapsis.fitting.results import FitResults
from periapsis.plotting.plots import orbit_plot, phase_fold_rv_plot, rv_fit_plot

ASTRO_UNITS = {"t": u.yr, "x": u.rad, "y": u.rad, "x_err": u.rad, "y_err": u.rad}
RV_UNITS = {"t": u.yr, "rv": u.AU / u.yr, "rv_err": u.AU / u.yr}
GAIA_UNITS = {"t": u.yr, "x": u.rad, "err": u.rad}


def test_phase_fold_rv_plot_uses_each_dataset_system_once():
    results = FitResults(
        MAP_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
            "K1": 4.0,
            "omega1": 0.3,
        },
        median_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
            "K1": 4.0,
            "omega1": 0.3,
        },
        priors={},
        param_names={'P', 'e', 'Tp', 'K', 'omega', 'gamma', 'K1', 'omega1'}
    )

    t = np.linspace(0.0, 20.0, 5)
    data = JointData([
        RadialVelocityData(t, np.zeros_like(t), np.ones_like(t), system="relative", units=RV_UNITS),
        RadialVelocityData(t + 1.0, np.zeros_like(t), np.ones_like(t), system="1", units=RV_UNITS),
    ])

    fig = phase_fold_rv_plot(results, data)
    assert len(fig.axes[0].lines) == 4
    plt.close(fig)


def test_phase_fold_rv_plot_accepts_regular_rv_data():
    results = FitResults(
        MAP_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
        },
        median_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
        },
        priors={},
        param_names={'P', 'e', 'Tp', 'K', 'omega', 'gamma'}
    )

    t = np.linspace(0.0, 20.0, 6)
    data = RadialVelocityData(t, np.zeros_like(t), np.ones_like(t), system="relative", units=RV_UNITS)

    fig = phase_fold_rv_plot(results, data)
    assert len(fig.axes[0].lines) >= 2
    plt.close(fig)


def test_rv_fit_plot_accepts_regular_rv_data():
    results = FitResults(
        MAP_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
        },
        median_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "K": 5.0,
            "omega": 0.2,
            "gamma": 0.0,
        },
        priors={},
        param_names={'P', 'e', 'Tp', 'K', 'omega', 'gamma'},
    )

    t = np.linspace(0.0, 20.0, 6)
    data = RadialVelocityData(t, np.zeros_like(t), np.ones_like(t), system="relative", units=RV_UNITS)

    fig = rv_fit_plot(results, data)
    assert fig is not None
    assert len(fig.axes[0].lines) >= 2
    plt.close(fig)


def test_rv_fit_plot_uses_declared_time_and_velocity_units():
    results = FitResults(
        MAP_params={
            "P": 1.0,
            "e": 0.1,
            "Tp": 0.0,
            "K": 0.01,
            "omega": 0.2,
            "gamma": 0.0,
        },
        median_params={
            "P": 1.0,
            "e": 0.1,
            "Tp": 0.0,
            "K": 0.01,
            "omega": 0.2,
            "gamma": 0.0,
        },
        priors={},
        param_names={"P", "e", "Tp", "K", "omega", "gamma"},
    )
    data = RadialVelocityData(
        t=[0.0, 100.0, 200.0],
        rv=[0.0, 1.0, 0.0],
        rv_err=[0.1, 0.1, 0.1],
        system="relative",
        units={"t": u.day, "rv": u.km / u.s, "rv_err": u.km / u.s},
    )

    fig = rv_fit_plot(results, data)

    assert "d" in fig.axes[0].get_xlabel()
    assert "km / s" in fig.axes[0].get_ylabel()
    plt.close(fig)



def test_orbit_plot_accepts_single_gaia_data():
    results = FitResults(
        MAP_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "A1": 0.5,
            "B1": 0.3,
            "F1": 0.2,
            "G1": 0.1,
            "dalpha": 0.0,
            "ddelta": 0.0,
            "mu_alpha": 0.0,
            "mu_delta": 0.0,
            "parallax": 1.0,
        },
        median_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "A1": 0.5,
            "B1": 0.3,
            "F1": 0.2,
            "G1": 0.1,
            "dalpha": 0.0,
            "ddelta": 0.0,
            "mu_alpha": 0.0,
            "mu_delta": 0.0,
            "parallax": 1.0,
        },
        priors={},
        param_names={'P', 'e', 'Tp', 'A1', 'B1', 'F1', 'G1', 'dalpha', 'ddelta', 'mu_alpha', 'mu_delta', 'parallax'},
    )

    t = np.linspace(0.0, 20.0, 6)
    spsi = np.sin(np.linspace(0.0, 2 * np.pi, len(t)))
    cpsi = np.cos(np.linspace(0.0, 2 * np.pi, len(t)))
    data = GaiaData(spsi, cpsi, t, np.ones_like(t), np.zeros_like(t), np.ones_like(t), system="1", units=GAIA_UNITS)

    fig = orbit_plot(results, data)
    assert fig is not None
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_orbit_plot_accepts_joint_astrometry_data():
    results = FitResults(
        MAP_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "A1": 0.4,
            "B1": 0.25,
            "F1": 0.15,
            "G1": 0.08,
            "dalpha": 0.0,
            "ddelta": 0.0,
            "mu_alpha": 0.0,
            "mu_delta": 0.0,
        },
        median_params={
            "P": 10.0,
            "e": 0.1,
            "Tp": 1.0,
            "A1": 0.4,
            "B1": 0.25,
            "F1": 0.15,
            "G1": 0.08,
            "dalpha": 0.0,
            "ddelta": 0.0,
            "mu_alpha": 0.0,
            "mu_delta": 0.0,
        },
        priors={},
        param_names={'P', 'e', 'Tp', 'A1', 'B1', 'F1', 'G1', 'dalpha', 'ddelta', 'mu_alpha', 'mu_delta'},
    )

    t = np.linspace(0.0, 20.0, 6)
    data = JointData([
        AstrometryData(
            t,
            np.zeros_like(t),
            np.zeros_like(t),
            np.ones_like(t),
            np.ones_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            system="1",
            units=ASTRO_UNITS,
        ),
        AstrometryData(
            t + 0.5,
            np.zeros_like(t),
            np.zeros_like(t),
            np.ones_like(t),
            np.ones_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            system="1",
            units=ASTRO_UNITS,
        ),
    ])

    fig = orbit_plot(results, data)
    assert fig is not None
    assert len(fig.axes) >= 3
    plt.close(fig)
