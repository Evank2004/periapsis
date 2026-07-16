from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.model.campbell import CampbellOrbit
from periapsis.initial.initial import InitialFit
from periapsis.utils.helpers import _match_param_keys
from periapsis.utils.solvers import solve_mass
import numpy as np
import emcee
from typing import cast


def _log_prior(params, param_order,prior_kwargs,m1=None,m2_max=None):
    lp = 0.0
    for name, val in zip(param_order, params):
        prior = prior_kwargs.get(name)
        if prior is not None:
            lp += prior.logpdf(val)
            if not np.isfinite(lp):
                return -np.inf
        else:
            print(f"Warning:Missing prior for {name}.")

    if m1 is not None and m2_max is not None:
        params_dict = _match_param_keys(dict(zip(param_order, params)))
        m2 = solve_mass(params_dict['a'], params_dict['P'], m1)
        if not np.isfinite(m2) or m2 > m2_max:
            return -np.inf
    return lp


def _campbell_log_like(params, data, param_order):
    params_dict = _match_param_keys(dict(zip(param_order, params)))
    model = CampbellOrbit(ref_epoch=getattr(data, 'ref_epoch', None), **params_dict)
    chi2 = data.chi2(model)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def _campbell_log_posterior(params, data, param_order,prior_kwargs,m1=None,m2_max=None):
    lp = _log_prior(params, param_order, prior_kwargs,m1=m1,m2_max=m2_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _campbell_log_like(params, data, param_order)

class MCMCCampbell(Fitter):
    def __init__(self, nwalkers, niter, m1=None,m2_max=None, pool=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.m2_max = m2_max
        

    def fit(self, data: Data) -> FitResults:

        pm_fit = self._proper_motion_fit(data)

        param_order = list(self.prior_kwargs.keys())
        ndim = len(param_order)
        
        initial_dict = InitialFit(data,method='Campbell', **self.prior_kwargs).get_intial()
        initial_guess = np.array([initial_dict[name] for name in param_order])
        bounds = np.array(
            [[self.prior_kwargs[name].min, self.prior_kwargs[name].max] for name in param_order],
            dtype=float,
        )
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        initial_guess = np.clip(initial_guess, lower, upper)

        pos = np.clip(initial_guess + 1e-4 * np.random.randn(self.nwalkers, ndim), lower, upper)

        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            _campbell_log_posterior,
            args=(data, param_order, self.prior_kwargs, self.m1, self.m2_max),
            pool=self.pool,
        )
        sampler.run_mcmc(pos, self.niter, progress=True)

        chain = cast(np.ndarray, sampler.get_chain())
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        Ess = (self.niter*self.nwalkers)/tau
        mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

        burn = int(np.nanmax(tau) * 2)
        thin = int(np.nanmin(tau) * 2)

        samples = cast(np.ndarray, sampler.get_chain(discard=burn, thin=thin, flat=True))
        lnprobs = cast(np.ndarray, sampler.get_log_prob(discard=burn, thin=thin, flat=True))

        
        param_means = chain.mean(axis=1)

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(param_order, samples[best_i]))
        median_params = dict(zip(param_order, np.median(samples, axis=0)))
        
        results_dict = {}
        for i, name in enumerate(param_order):
            results_dict[name] = samples[:, i]

        results_dict['lnprob'] = lnprobs
        results_dict['Ess'] = Ess
        results_dict['mean_acceptance_fraction'] = mean_acceptance_fraction
        results_dict['tau'] = tau
        results_dict['param_means'] = param_means
        results_dict['param_names'] = param_order
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['PM_fit'] = pm_fit
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        results_dict['raw_sampler'] = None
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'Campbell'
        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results