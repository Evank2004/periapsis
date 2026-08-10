"""
periapsis - Orbit modeling and fitting with flexible parameters
"""

from .data import AstrometryData, GaiaData, RadialVelocityData, JointData
from .fitting import MCMCFitter, UltranestFitter, MCMCLinearFitter, UltranestLinearFitter, MCMCGaiaFitter, UltranestGaiaFitter, FitResults
from .initial import InitialGuess, AstrometryInitialGuess, AstrometryLinearInitialGuess, RVInitialGuess, GaiaInitialGuess, JointInitialGuess
from .model import Orbit
from .plotting import all_plots, mcmc_autocorrelation_plot, corner_plot, ess_distribution_plot, prior_dist_plot, prior_histogram_2d, prior_conditional_histogram_2d, posterior_over_prior, orbit_plot, sky_motion_plot, multi_orbit_plot, distribution, mass_distribution
from .prior import Prior, FixedPrior, Bounds, LogUniformPrior, LogNormalPrior, NormalPrior, UniformPrior
from .stats import red_chi2, delta_chi2, credible_intervals, all_stats
from .params import a, a1, a2, b, b1, b2, p, p1, p2, r_a, r_a1, r_a2, r_p, r_p1, r_p2, e, i, cosi, sini, omega, omega1, omega2, Omega, Omega1, Omega2
from .params import piomega, piomega1, piomega2, P, n, A, A1, A2, B, B1, B2, C, C1, C2, F, F1, F2, G, G1, G2, H, H1, H2, c, c1, c2, h, h1, h2
from .params import Mtot, M1, M2, minM1, minM2, Msini, mu, Tepoch, Tp, t0, M0, L0, E0, nu0, l0, l01, l02, uM0, uM01, uM02, u0, u01, u02, K, K1, K2, q
from .params import dalpha, ddelta, mu_alpha, mu_delta, gamma, f1, f2, parallax, distance
from .params import log, ang

__all__ = [
    "data", "fitting", "model", "prior", "stats", "plotting", "params"

    # Data
    "AstrometryData", "GaiaData", "RadialVelocityData", "JointData",

    # Fitters
    "MCMCFitter", "UltranestFitter", "MCMCLinearFitter", "UltranestLinearFitter", "MCMCGaiaFitter", "UltranestGaiaFitter", "FitResults",

    # Initial Guesses
    "InitialGuess", "AstrometryInitialGuess", "AstrometryLinearInitialGuess", "RVInitialGuess", "GaiaInitialGuess", "JointInitialGuess",

    # Modeling
    "Orbit",

    # Plotting
    "all_plots", "mcmc_autocorrelation_plot", "corner_plot", "ess_distribution_plot", "prior_dist_plot", "prior_histogram_2d", "prior_conditional_histogram_2d", "posterior_over_prior", "orbit_plot", "sky_motion_plot", "multi_orbit_plot", "distribution", "mass_distribution",

    # Priors
    "Prior", "FixedPrior", "Bounds", "LogUniformPrior", "LogNormalPrior", "NormalPrior", "UniformPrior",

    # Stats
    "red_chi2", "delta_chi2", "credible_intervals", "all_stats",

    # Parameters
    "a", "a1", "a2", "b", "b1", "b2", "p", "p1", "p2", "r_a", "r_a1", "r_a2", "r_p", "r_p1", "r_p2", "e", "i", "cosi", "sini", "omega", "omega1", "omega2", "Omega", "Omega1", "Omega2",
    "piomega", "piomega1", "piomega2", "P", "n", "A", "A1", "A2", "B", "B1", "B2", "C", "C1", "C2", "F", "F1", "F2", "G", "G1", "G2", "H", "H1", "H2", "c", "c1", "c2", "h", "h1", "h2",
    "Mtot", "M1", "M2", "minM1", "minM2", "Msini", "mu", "Tepoch", "Tp", "t0", "M0", "L0", "E0", "nu0", "l0", "l01", "l02", "uM0", "uM01", "uM02", "u0", "u01", "u02", "K", "K1", "K2", "q",
    "dalpha", "ddelta", "mu_alpha", "mu_delta", "gamma", "f1", "f2", "parallax", "distance",
    "log", "ang"
]