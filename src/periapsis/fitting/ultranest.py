from periapsis.data.common import AstrometryData
from periapsis.prior import Bounds, FixedPrior

from .fitter import Fitter
from periapsis.model.orbit import Orbit
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.params.transforms import covered_parameters, build_transform_functions, overconstrained_parameters

import numpy as np
import ultranest

reject_logl = -1e300

class UltranestFitter(Fitter):
    def __init__(self, output_params, min_num_live_points=400, min_ess=400, dlogz=0.5, dKL=0.5, frac_remain=0.01, Lepsilon=0.001, max_iters=None, max_ncalls=None, **priors):
        super().__init__(**priors)
        self.output_params = frozenset(output_params)
        self.output_param_order = tuple(output_params)
        if len(self.output_params) != len(self.output_param_order):
            raise ValueError("Duplicate parameters found in output_params.")
        self.min_num_live_points = min_num_live_points
        self.min_ess = min_ess
        self.dlogz = dlogz
        self.dKL = dKL
        self.frac_remain = frac_remain
        self.Lepsilon = Lepsilon
        self.max_iters = max_iters
        self.max_ncalls = max_ncalls

        self.prior_params = set(priors.keys())
        self.prior_order = tuple(priors.keys())
        self.fixed_prior_params = {p for p in self.prior_order if isinstance(self.priors[p], FixedPrior)}
        self.non_fixed_prior_params = {p for p in self.prior_order if not isinstance(self.priors[p], FixedPrior)}
        self.non_bound_prior_params = {p for p in self.prior_order if not isinstance(self.priors[p], Bounds)}
        self.bound_params = {p for p in self.prior_order if isinstance(self.priors[p], Bounds)}
        self.non_bound_fixed_prior_params = {p for p in self.prior_order if not isinstance(self.priors[p], (Bounds, FixedPrior))}
        self.sample_order = tuple([p for p in self.prior_order if not isinstance(self.priors[p], (Bounds, FixedPrior))])
        self.output_covered_params = covered_parameters(self.output_params)
        self.prior_covered_params = covered_parameters(self.non_bound_prior_params)
        self.posterior_covered_params = covered_parameters(self.output_params.union(self.non_bound_prior_params))
        self.overconstrained_priors = overconstrained_parameters(self.non_bound_prior_params)

        if any(param not in self.prior_covered_params for param in output_params):
            missing = [param for param in output_params if param not in self.prior_covered_params]
            raise ValueError(f"Missing priors to constrain sampled parameters: {missing}")

        if self.overconstrained_priors:
            raise ValueError(f"Overconstrained priors: {self.overconstrained_priors}. Please remove one of the priors for these parameters. Optionally replace with a Bounds prior if you want to constrain the parameter without sampling it.")
        
        self.prior_to_sampled_transform = build_transform_functions(self.non_bound_prior_params, output_params)

    def fit(self, data: Data, quiet=False) -> FitResults:
        param_order = self.sample_order
        full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        likelihood_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params], self.bound_params)

        def prior_transform(cube):
            cube = np.array(cube, copy=True)
            sampled_priors = {}
            i = 0
            for name in param_order:
                sampled_priors[name] = self.priors[name].unp(cube[i])
                i += 1
            params = self.prior_to_sampled_transform(**sampled_priors, **{name: self.priors[name].value for name in self.fixed_prior_params})
            return np.array([*[sampled_priors[name] for name in param_order], *[params[name] for name in self.output_param_order if name not in param_order]])

        def log_likelihood(params):
            full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
            param_dict = dict(zip(full_param_order, params))
            param_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            bound_param_dict = likelihood_transform(**param_dict)
            for name in self.bound_params:
                if self.priors[name].lower is not None and bound_param_dict[name] < self.priors[name].lower:
                    return reject_logl # TODO possibly slope inwards towards the bounds instead of hard cutoff
                if self.priors[name].upper is not None and bound_param_dict[name] > self.priors[name].upper:
                    return reject_logl # TODO possibly slope inwards towards the bounds
            model = Orbit(**param_dict)
            chi2 = data.chi2(model)
            return -0.5 * chi2

                


        # def log_likelihood(params):
        #     prior_params = self.sampled_to_prior_transform(**dict(zip(param_order, params)), **{name: self.priors[name].value for name in self.fixed_prior_params})
        #     for name, prior in self.priors.items():
        #         if isinstance(prior, Bounds):
        #             if not (prior.lower <= prior_params[name] <= prior.upper):
        #                 return -np.inf
        #     params_dict = dict(zip(self.output_params, params))
        #     fixed_prior_dict = {name: prior.value for name, prior in self.priors.items() if isinstance(prior, FixedPrior)}
        #     model = Orbit(**params_dict, **fixed_prior_dict)
        #     chi2 = data.chi2(model)
        #     if chi2 is None:
        #         return -np.inf
        #     return -0.5 * chi2

        if isinstance(data, AstrometryData):
            pm_fit = self._proper_motion_fit(data)
        else:
            pm_fit = dict()

        sampler = ultranest.ReactiveNestedSampler(
            param_names=tuple(param_order),
            loglike=log_likelihood, 
            transform=prior_transform,
            derived_param_names=tuple([name for name in self.output_param_order if name not in param_order]),
            wrapped_params=None, # TODO
        )
        results = sampler.run(
            min_num_live_points=self.min_num_live_points,
            min_ess=self.min_ess,
            dlogz=self.dlogz,
            dKL=self.dKL,
            frac_remain=self.frac_remain,
            Lepsilon=self.Lepsilon,
            max_iters=self.max_iters,
            max_ncalls=self.max_ncalls,
            show_status=not quiet,
            viz_callback=False if quiet else 'auto',
        )
        
        samples = np.array(results['samples'])
        logl = np.array(results['weighted_samples']['logl'])
        # logl = np.array(results['logl'])
        
        best_i = np.argmax(logl)
        best_params = dict(zip(param_order, samples[best_i]))
        median_params = dict(zip(param_order, np.median(samples, axis=0)))
               
        
        results_dict = {}
        for i, name in enumerate(param_order):
            results_dict[name] = samples[:, i]

        # Add derived parameters to results_dict
        for i, name in enumerate([name for name in self.output_param_order if name not in param_order]):
            results_dict[name] = samples[:, len(param_order) + i]
                
                
        results_dict['Ess'] = results['ess']
        results_dict['logZ'] = results['logz']
        results_dict['logZerr'] = results['logzerr']
        results_dict['param_names'] = param_order
        results_dict['raw_sampler'] = sampler
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['PM_fit'] = pm_fit
        results_dict['logl'] = logl
        results_dict['samples'] = samples
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'Campbell'
        results_dict['priors'] = self.priors

        # TODO: normalize ref_epoch
        if results_dict['ref_epoch'] is not None:
            results_dict['priors']['Tepoch'] = FixedPrior(results_dict['ref_epoch'])
        
        fit_results = FitResults(**results_dict)
        return fit_results

