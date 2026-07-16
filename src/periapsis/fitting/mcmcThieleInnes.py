from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.model import thieleinnes
from periapsis.initial.initial import InitialFit
from periapsis.utils.solvers import transform_theile
from periapsis.utils.helpers import _match_param_keys
import numpy as np
import emcee

class MCMCThieleInnes(Fitter):
    def __init__(self, nwalkers, niter, m1=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nwalkers = nwalkers
        self.niter = niter

    def fit(self, data: Data) -> FitResults:
        pm_fit = self._proper_motion_fit(data)

        param_order = list(self.prior_kwargs.keys())
        ndim = len(param_order)

        def ln_prior(params):
            #this is a placeholder, we will need to figure out transformations
            lp = 0 
            for name,val in zip(param_order, params):
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    lp += prior.logpdf(val)
                    if not np.isfinite(lp):
                        return -np.inf
                else:
                    print(f"Warning:Missing prior for {name}.")
            return lp

        def ln_like(params, data):
            params_dict = _match_param_keys(dict(zip(param_order, params)))
            model = thieleinnes.ThieleInnesOrbit(ref_epoch=getattr(data, 'ref_epoch', None), **params_dict)
            chi2 = data.chi2(model)
            if not np.isfinite(chi2):
                return -np.inf
            return -0.5 * chi2

        def ln_prob(params, data):
            lp = ln_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + ln_like(params, data)
            
        initial_dict = InitialFit(data,method='ThieleInnes', **self.prior_kwargs).get_intial()
        initial_guess = np.array([initial_dict[name] for name in param_order])
        bounds = np.array(
            [[self.prior_kwargs[name].min, self.prior_kwargs[name].max] for name in param_order],
            dtype=float,
        )
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        initial_guess = np.clip(initial_guess, lower, upper)
        
        pos = np.clip(initial_guess + 1e-4 * np.random.randn(self.nwalkers, ndim), lower, upper)

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, ln_prob, args=(data,))
        sampler.run_mcmc(pos, self.niter, progress=True)

        chain = sampler.get_chain()
        param_means = chain.mean(axis=1)

        tau = emcee.autocorr.integrated_time(chain,quiet=True)

        Ess = (self.niter*self.nwalkers)/tau

        maf = np.mean(sampler.acceptance_fraction)

        burn = int(np.nanmax(tau) * 2)
        thin = int(np.nanmin(tau) * 2)

        samples = sampler.get_chain(discard=burn,thin=thin,flat=True)
        lnprobs = sampler.get_log_prob(discard=burn,thin=thin,flat=True)

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(param_order, samples[best_i]))

        median_params = dict(zip(param_order, np.median(samples, axis=0)))

        results_dict = {}
        for i, name in enumerate(param_order):
            results_dict[name] = samples[:, i]
        results_dict['lnprob'] = lnprobs
        results_dict['Ess'] = Ess
        results_dict['mean_acceptance_fraction'] = maf
        results_dict['tau'] = tau
        results_dict['param_means'] = param_means
        results_dict['param_names'] = param_order
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['PM_fit'] = pm_fit
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)

        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'ThieleInnes'
        results_dict['raw_sampler'] = None

        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results
            
            