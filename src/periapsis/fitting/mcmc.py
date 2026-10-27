import warnings
from copy import deepcopy
from .fitter import Fitter
from .results import FitResults
from periapsis.data import Data, AstrometryData, PhysicalAstrometryData, RadialVelocityData, JointData,GaiaData
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.initial import InitialGuess, AstrometryInitialGuess, RVInitialGuess, JointInitialGuess, GaiaInitialGuess
from periapsis.model import Orbit
from periapsis.params import covered_parameters, build_transform_functions, overconstrained_parameters
import numpy as np
import emcee
from typing import Type, cast, Iterable, Optional
from dataclasses import dataclass
from functools import lru_cache
from periapsis.params.units import CanonicalUnits


@dataclass(frozen=True)
class _PosteriorContext:
    data: Data
    param_order: tuple[str, ...]
    fixed_items: tuple[tuple[str, object], ...]
    direct_prior_items: tuple[tuple[int, Prior], ...]
    derived_prior_items: tuple[tuple[str, Prior], ...]
    known_names: tuple[str, ...]


class MCMCFitter(Fitter):
    def __init__(self, nwalkers: int, niter: int, sample_params: Iterable,ref_epoch=None, pool=None, initial: Optional[Type[InitialGuess]] = None, **priors):
        super().__init__(ref_epoch,**priors)
        if nwalkers <= 0 or niter <= 0:
            raise ValueError("nwalkers and niter must be positive integers.")
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.param_order = tuple(sample_params)
        self.sample_params = frozenset(sample_params)
        self.initial = initial
        if len(self.param_order) != len(self.sample_params):
            raise ValueError("Sampled parameters must be unique.")
        if not self.param_order:
            raise ValueError("At least one parameter must be sampled.")
        self.param_indexes = {
            name: index for index, name in enumerate(self.param_order)
        }
        self.prior_params = set(self.priors.keys())
        self.fixed_prior_params = {p for p in self.prior_params if isinstance(self.priors[p], FixedPrior)}
        self.non_bound_prior_params = {p for p in self.prior_params if not isinstance(self.priors[p], Bounds)}
        self.sample_covered_params = covered_parameters(set(self.sample_params) | self.fixed_prior_params)
        self.prior_covered_params = covered_parameters(self.non_bound_prior_params)
        self.posterior_covered_params = covered_parameters(set(self.sample_params) | self.non_bound_prior_params)
        self.overconstrained_priors = overconstrained_parameters(self.non_bound_prior_params)

        missing_priors = self.sample_params.difference(self.prior_covered_params)
        if missing_priors:
            raise ValueError(f"Sampled parameters must have an explicit or implicit prior defined. {missing_priors} are missing priors.")

        if self.overconstrained_priors:
            raise ValueError(f"Some priors are contradictory and must be removed or replaced with a Bounds: {sorted(self.overconstrained_priors)}.")

        if any(prior_param not in self.sample_covered_params and not isinstance(self.priors[prior_param], FixedPrior) for prior_param in self.prior_params):
            unreachable_priors = [prior_param for prior_param in self.prior_params if prior_param not in self.sample_covered_params and not isinstance(self.priors[prior_param], FixedPrior)]
            warnings.warn(f"Some priors are not reachable from the sampled parameters and will be ignored: {sorted(unreachable_priors)}.")
        
        fixed_names = {
            name for name, prior in self.priors.items() if isinstance(prior, FixedPrior)
        }
        sampled_and_fixed = self.sample_params.intersection(fixed_names)
        if sampled_and_fixed:
            raise ValueError(f"Sampled parameters cannot be fixed. {sorted(sampled_and_fixed)} are fixed.")

    
    def _log_posterior(self, params: np.ndarray, context: _PosteriorContext):
        lp = 0.0

        for index, prior in context.direct_prior_items:
            if isinstance(prior, FixedPrior):
                continue  # Skip fixed priors
            contribution = cast(float, prior.logpdf(params[index]))
            if not np.isfinite(contribution):
                return -np.inf
            lp += contribution
        
        values = dict(context.fixed_items)
        values.update(zip(context.param_order, params))

        if context.derived_prior_items:
            target_names = tuple(name for name, _ in context.derived_prior_items)
            transform = _build_prior_transform(context.known_names, target_names)
            transformed = transform(**values)
            for name, prior in context.derived_prior_items:
                if isinstance(prior, FixedPrior):
                    continue  # Skip fixed priors
                contribution = cast(float, prior.logpdf(transformed[name]))
                if not np.isfinite(contribution):
                    return -np.inf
                lp += contribution
            # Cache transformed values for likelihood evaluation
            values.update(transformed)
        
        model = Orbit(**values)
        log_likelihood = context.data.log_likelihood(model)
        if not np.isfinite(log_likelihood):
            return -np.inf
        return lp + log_likelihood
    

    def _posterior_context(self, data: Data, priors=None) -> _PosteriorContext:
        if priors is None:
            priors = self.priors

        param_order = self.param_order
        param_indexes = {
            name: index for index, name in enumerate(param_order)
        }
        fixed_items = tuple(
            (name, prior.value)
            for name, prior in priors.items()
            if isinstance(prior, FixedPrior)
        )
        known_names = tuple(
            dict.fromkeys((*param_order, *(name for name, _ in fixed_items)))
        )
        reachable = covered_parameters(set(known_names))

        direct_prior_items = []
        derived_prior_items = []
        for name, prior in priors.items():
            if isinstance(prior, FixedPrior):
                continue
            if name in param_indexes:
                direct_prior_items.append((param_indexes[name], prior))
            elif name in reachable:
                derived_prior_items.append((name, prior))
            else:
                print(f"WARNING: Prior for parameter {name} is not reachable from sampled parameters or fixed parameters. It will be ignored in the posterior evaluation.")
        # Bounds are the cheapest rejection checks, so retain stable prior order
        # within each group while evaluating Bounds first.
        direct_prior_items.sort(
            key=lambda item: not isinstance(item[1], Bounds)
        )

        return _PosteriorContext(
            data=data,
            param_order=param_order,
            fixed_items=fixed_items,
            direct_prior_items=tuple(direct_prior_items),
            derived_prior_items=tuple(derived_prior_items),
            known_names=known_names,
        )

    def _fit(self, data: Data, canonical_priors: dict[str, Prior], canonical_ref_epoch: Optional[float], rng: np.random.RandomState) -> dict:
        context = self._posterior_context(data, canonical_priors)
        ndim = len(self.sample_params)

        initial = self.initial
        if initial is None:
            if isinstance(data, AstrometryData):
                initial = AstrometryInitialGuess
            elif isinstance(data, PhysicalAstrometryData):
                initial = AstrometryInitialGuess
            elif isinstance(data, RadialVelocityData):
                initial = RVInitialGuess
            elif isinstance(data, JointData):
                initial = JointInitialGuess
            elif isinstance(data, GaiaData):
                initial = GaiaInitialGuess
            else:
                raise ValueError("No initial guess class provided and data type is not recognized for MCMC initial guess generation.")
        initial_instance = initial(data, rng, canonical_ref_epoch, **canonical_priors)
        pos = initial_instance.get_initial_guess(self.param_order, self.nwalkers)

        
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            self._log_posterior,
            args=(context,),
            pool=self.pool,
        )
        sampler.run_mcmc(pos, self.niter, progress=True)

        chain = cast(np.ndarray, sampler.get_chain())
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        Ess = (self.niter*self.nwalkers)/tau
        mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

        nanmaxtau = np.nanmax(tau)
        nanmintau = np.nanmin(tau)
        if not np.isnan(nanmaxtau):
            burn = int(np.nanmax(tau) * 2)
        else:
            warnings.warn("Autocorrelation time could not be estimated. Setting burn-in to 0.")
            burn = 0
        if not np.isnan(nanmintau):
            thin = max(1, int(np.nanmin(tau) * 2))
        else:
            warnings.warn("Autocorrelation time could not be estimated. Setting thinning to 1.")
            thin = 1

        samples = cast(np.ndarray, sampler.get_chain(discard=burn, thin=thin, flat=True))
        lnprobs = cast(np.ndarray, sampler.get_log_prob(discard=burn, thin=thin, flat=True))

        
        param_means = chain.mean(axis=1)

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(self.param_order, samples[best_i]))
        median_params = dict(zip(self.param_order, np.median(samples, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = canonical_priors[prior].value
            median_params[prior] = canonical_priors[prior].value

        reported_samps = np.empty_like(samples)
        for i, name in enumerate(self.param_order):
            reported_samps[:, i] = samples[:, i]

        reported_param_means = np.empty_like(param_means)
        for i, name in enumerate(self.param_order):
            reported_param_means[:, i] = param_means[:, i]

        reported_best_params = {
            name: value
            for name, value in best_params.items()
        }
        reported_median_params = {
            name: value
            for name, value in median_params.items()
        }
        
        results_dict = {}
        for i, name in enumerate(self.param_order):
            results_dict[name] = reported_samps[:, i]

        results_dict['lnprob'] = lnprobs
        results_dict['Ess'] = Ess
        results_dict['mean_acceptance_fraction'] = mean_acceptance_fraction
        results_dict['tau'] = tau
        results_dict['param_means'] = reported_param_means
        results_dict['param_names'] = self.param_order
        results_dict['sampled_param_names'] = self.param_order
        results_dict['MAP_params'] = reported_best_params
        results_dict['median_params'] = reported_median_params
        results_dict['raw_sampler'] = sampler
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'full'
        return results_dict

@lru_cache(maxsize=128)
def _build_prior_transform(known_names: tuple[str, ...], target_names: tuple[str, ...]):
    return build_transform_functions(known_names, target_names)