from .fitter import Fitter
from .results import FitResults
from periapsis.data import Data, data
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.initial import InitialFit
from periapsis.model import Orbit
from periapsis.params import covered_parameters, build_transform_functions, overconstrained_parameters
import numpy as np
import emcee
from typing import cast, Iterable
from dataclasses import dataclass
from functools import lru_cache

class MCMCFitter(Fitter):
    def __init__(self, nwalkers: int, niter: int, sample_params: Iterable, pool=None, **priors):
        self.prior_kwargs = priors
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.param_order = tuple(sample_params)
        self.sample_params = frozenset(sample_params)
        if len(self.param_order) != len(self.sample_params):
            raise ValueError("Sampled parameters must be unique.")
        if not self.param_order:
            raise ValueError("At least one parameter must be sampled.")
        self.param_indexes = {
            name: index for index, name in enumerate(self.param_order)
        }
        self.prior_params = set(priors.keys())
        self.non_bound_prior_params = {p for p in self.prior_params if not isinstance(self.prior_kwargs[p], Bounds)}
        self.sample_covered_params = covered_parameters(self.sample_params)
        self.prior_covered_params = covered_parameters(self.non_bound_prior_params)
        self.posterior_covered_params = covered_parameters(self.sample_params.union(self.non_bound_prior_params))
        self.overconstrained_priors = overconstrained_parameters(self.non_bound_prior_params)

        missing_priors = self.sample_params.difference(self.prior_covered_params)
        if missing_priors:
            raise ValueError(f"Sampled parameters must have an explicit or implicit prior defined. {missing_priors} are missing priors.")

        if self.overconstrained_priors:
            raise ValueError(f"Some priors are contradictory and must be removed or replaced with a Bounds: {sorted(self.overconstrained_priors)}.")
        
        fixed_names = {
            name for name, prior in self.prior_kwargs.items() if isinstance(prior, FixedPrior)
        }
        sampled_and_fixed = self.sample_params.intersection(fixed_names)
        if sampled_and_fixed:
            raise ValueError(f"Sampled parameters cannot be fixed. {sorted(sampled_and_fixed)} are fixed.")

    def _sample_priors(self, param_order, size: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
        """
        Samples the prior distributions, and then uses rejection sampling to ensure that the sampled parameters are consistent with any provided bounds.
        """
        non_bound_names = tuple(
            name 
            for name in self.prior_kwargs 
            if not isinstance(self.prior_kwargs[name], Bounds)
        )
        sample_transform = build_transform_functions(non_bound_names, tuple(param_order))
        param_indexes = {name: i for i, name in enumerate(param_order)}
        derived_bound_names = tuple(
            name
            for name, prior in self.prior_kwargs.items()
            if isinstance(prior, Bounds)
            and name not in param_indexes
            and name in self.prior_covered_params
        )
        bound_known_names = tuple(dict.fromkeys((*param_order, *non_bound_names)))
        bound_transform = build_transform_functions(bound_known_names, derived_bound_names)

        bad = np.ones(size, dtype=bool)
        pos = np.empty((size, len(param_order)))
        warned = False
        while np.any(bad):
            # Sample initial prior distributions
            values = {}
            for name, prior in self.prior_kwargs.items():
                if not isinstance(prior, Bounds):
                    values[name] = prior.sample(rng, size=np.sum(bad))
            
            # Transform prior distributions to sampled parameters
            transformed_samples = sample_transform(**values)
            pos[bad] = np.column_stack([transformed_samples[name] for name in param_order])

            # Check bounds
            known_param_values = {}
            for name, i in param_indexes.items():
                known_param_values[name] = pos[bad, i]
            known_param_values.update({p: values[p] for p in self.non_bound_prior_params})
            derived_bound_values = bound_transform(**known_param_values)
            
            new_bad = np.zeros(size, dtype=bool)
            for name, prior in self.prior_kwargs.items():
                if isinstance(prior, Bounds):
                    if name in param_indexes:
                        if prior.lower is not None:
                            new_bad[bad] |= pos[bad, param_indexes[name]] < prior.lower
                        if prior.upper is not None:
                            new_bad[bad] |= pos[bad, param_indexes[name]] > prior.upper
                    elif name in derived_bound_values:
                        val = derived_bound_values[name]
                        if prior.lower is not None:
                            new_bad[bad] |= val < prior.lower
                        if prior.upper is not None:
                            new_bad[bad] |= val > prior.upper
                    else:
                        if not warned:
                            print(f"Warning: Bounds prior for parameter {name} cannot be applied to any sampled parameter.")
                            warned = True
            bad = new_bad
        return pos
    
    def _log_posterior(self, params: np.ndarray, context: _PosteriorContext):
        lp = 0.0

        for index, prior in context.direct_prior_items:
            contribution = prior.logpdf(params[index])
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
                contribution = prior.logpdf(transformed[name])
                if not np.isfinite(contribution):
                    return -np.inf
                lp += contribution
            # Cache transformed values for likelihood evaluation
            values.update(transformed)
        
        model = Orbit(**values)
        chi2 = context.data.chi2(model)
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2
    

    def _posterior_context(self, data: Data) -> _PosteriorContext:
        param_order = self.param_order
        param_indexes = {
            name: index for index, name in enumerate(param_order)
        }
        fixed_items = tuple(
            (name, prior.value)
            for name, prior in self.prior_kwargs.items()
            if isinstance(prior, FixedPrior)
        )
        known_names = tuple(
            dict.fromkeys((*param_order, *(name for name, _ in fixed_items)))
        )
        reachable = covered_parameters(known_names)

        direct_prior_items = []
        derived_prior_items = []
        for name, prior in self.prior_kwargs.items():
            if isinstance(prior, FixedPrior):
                continue
            if name in param_indexes:
                direct_prior_items.append((param_indexes[name], prior))
            elif name in reachable:
                derived_prior_items.append((name, prior))
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
        


    def fit(self, data: Data, rng: np.random.RandomState) -> FitResults:

        pm_fit = self._proper_motion_fit(data)

        # param_order = list(self.sample_params)
        param_order = self.param_order
        context = self._posterior_context(data)
        
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
        
        pos = self._sample_priors(param_order, self.nwalkers, rng)
        
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            self._log_posterior,
            # args=(data, param_order, self.prior_kwargs),
            args=(context,),
            pool=self.pool,
        )
        sampler.run_mcmc(pos, self.niter, progress=True)

        chain = cast(np.ndarray, sampler.get_chain())
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        Ess = (self.niter*self.nwalkers)/tau
        mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

        burn = int(np.nanmax(tau) * 2)
        thin = max(1, int(np.nanmin(tau) * 2))

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
        results_dict['priors'] = self.prior_kwargs
        # results_dict['fit_method'] = 'Campbell'
        fit_results = FitResults(**results_dict)
        return fit_results
    

@dataclass(frozen=True)
class _PosteriorContext:
    data: Data
    param_order: tuple[str, ...]
    fixed_items: tuple[tuple[str, object], ...]
    direct_prior_items: tuple[tuple[int, Prior], ...]
    derived_prior_items: tuple[tuple[str, Prior], ...]
    known_names: tuple[str, ...]


@lru_cache(maxsize=128)
def _build_prior_transform(known_names: tuple[str, ...], target_names: tuple[str, ...]):
    return build_transform_functions(known_names, target_names)