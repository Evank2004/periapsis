from emcee import model

from .fitter import Fitter
from .results import FitResults
from periapsis.data import Data, data
from periapsis.prior import Prior
from periapsis.initial import InitialFit
from periapsis.model import Orbit
from periapsis.params import covered_parameters, build_transform_function
import numpy as np
import emcee
from typing import cast, Iterable

def _log_prior(params: np.ndarray, param_order, prior_kwargs: dict[str, Prior]):
    lp = 0.0
    for name, prior in prior_kwargs.items():
        try:
            val = build_transform_function(param_order, name) # TODO include fixed priors as known data for the transform.
        except KeyError:
            continue # This prior does not apply to this sampled parameter set.
            
        lp += prior.logpdf(val(**{p: params[i] for i, p in enumerate(param_order)}))
    return lp
    
def _log_likelihood(params: np.ndarray, data: Data, param_order: list):
    model = Orbit(**{p: params[i] for i, p in enumerate(param_order)})
    chi2 = data.chi2(model)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2

def _log_posterior(params, data, param_order, prior_kwargs):
    lp = _log_prior(params, param_order, prior_kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(params, data, param_order)

class MCMCFitter(Fitter):
    def __init__(self, nwalkers: int, niter: int, sample_params: Iterable, pool=None, **priors):
        # super().__init__(**priors)
        self.prior_kwargs = priors
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.sample_params = set(sample_params)
        self.prior_params = set(priors.keys())
        self.sample_covered_params = covered_parameters(self.sample_params)
        self.prior_covered_params = covered_parameters(self.prior_params)
        self.posterior_covered_params = covered_parameters(self.sample_params.union(self.prior_params))

        assert set(self.sample_params).issubset(self.prior_covered_params), f"Sampled parameters must have an explicit or implicit prior defined. {set(self.sample_params) - self.prior_covered_params} are missing priors."

        # TODO ensure that no sample_params are overconstrained by the priors.


    def fit(self, data: Data, rng: np.random.RandomState) -> FitResults:

        pm_fit = self._proper_motion_fit(data)

        param_order = list(self.sample_params)
        
        # initial_dict = InitialFit(data,method='Campbell', **self.prior_kwargs).get_intial()
        # initial_guess = np.array([initial_dict[name] for name in param_order])
        # bounds = np.array(
            # [[self.prior_kwargs[name].min, self.prior_kwargs[name].max] for name in param_order],
            # dtype=float,
        # )
        # lower = bounds[:, 0]
        # upper = bounds[:, 1]
        # initial_guess = np.clip(initial_guess, lower, upper)

        ndim = len(self.sample_params)
        # pos = np.clip(initial_guess + 1e-4 * np.random.randn(self.nwalkers, ndim), lower, upper)
        
        # Sample initial prior distributions
        values = {}
        for name, prior in self.prior_kwargs.items():
            values[name] = prior.sample(rng, size=self.nwalkers)
        
        # Transform prior distributions to sampled parameters
        poss = []
        for name in param_order:
            transform = build_transform_function(values.keys(), name)
            poss.append(transform(**values))
        pos = np.array(poss).T
        
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            _log_posterior,
            args=(data, param_order, self.prior_kwargs),
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
        # results_dict['fit_method'] = 'Campbell'
        fit_results = FitResults(**results_dict)
        return fit_results