from periapsis.data import AstrometryData, RadialVelocityData, GaiaData, JointData
from periapsis.prior import Bounds, FixedPrior

from .fitter import Fitter
from periapsis.model.orbit import Orbit
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.params.transforms import covered_parameters, build_transform_functions, overconstrained_parameters, wrapped_parameters
from periapsis.params.units import CanonicalUnits

import numpy as np
import ultranest

reject_logl = -1e300

class UltranestFitter(Fitter):
    def __init__(self, output_params, ref_epoch=None, min_num_live_points=400, min_ess=400, dlogz=0.5, dKL=0.5, frac_remain=0.01, Lepsilon=0.001, max_iters=None, max_ncalls=None, **priors):
        super().__init__(ref_epoch,**priors)
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
        canonical_priors = self._canonical_priors(data)
        canonical_ref_epoch = canonical_priors['Tepoch'].value
        param_order = self.sample_order
        full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        likelihood_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params], self.bound_params)

        def prior_transform(cube):
            cube = np.array(cube, copy=True)
            sampled_priors = {}
            i = 0
            for name in param_order:
                sampled_priors[name] = canonical_priors[name].unp(cube[i])
                i += 1
            params = self.prior_to_sampled_transform(
                **sampled_priors,
                **{name: canonical_priors[name].value for name in self.fixed_prior_params},
            )
            return np.array([*[sampled_priors[name] for name in param_order], *[params[name] for name in self.output_param_order if name not in param_order]])

        def log_likelihood(params):
            full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
            param_dict = dict(zip(full_param_order, params))
            param_dict.update({name: canonical_priors[name].value for name in self.fixed_prior_params})
            bound_param_dict = likelihood_transform(**param_dict)
            for name in self.bound_params:
                if canonical_priors[name].lower is not None and bound_param_dict[name] < canonical_priors[name].lower:
                    return reject_logl # TODO possibly slope inwards towards the bounds instead of hard cutoff
                if canonical_priors[name].upper is not None and bound_param_dict[name] > canonical_priors[name].upper:
                    return reject_logl # TODO possibly slope inwards towards the bounds
            model = Orbit(**param_dict)
            log_likelihood = data.log_likelihood(model)
            if not np.isfinite(log_likelihood):
                return reject_logl
            return log_likelihood

                

        null_hypothesis_canonical = self._null_hypothesis_fit(
            data, ref_epoch=canonical_ref_epoch
        )

        sampler = ultranest.ReactiveNestedSampler(
            param_names=tuple(param_order),
            loglike=log_likelihood, 
            transform=prior_transform,
            derived_param_names=tuple([name for name in self.output_param_order if name not in param_order]),
            wrapped_params=[name in wrapped_parameters for name in param_order],
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

        best_i = np.argmax(logl)
        best_params = dict(zip(full_param_order, samples[best_i]))
        median_params = dict(zip(full_param_order, np.median(samples, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = canonical_priors[prior].value
            median_params[prior] = canonical_priors[prior].value

        if null_hypothesis_canonical is None:
            raise RuntimeError("Null-hypothesis fitting returned no result.")

        def from_canonical(name, value):
            unit = data.parameter_unit(name)
            dimension = data.parameter_dimension(name)
            if unit is None or dimension is None:
                raise ValueError(f"Could not resolve units for parameter '{name}'.")
            return np.asarray(value) / unit.to(CanonicalUnits[dimension])

        reported_samples = np.column_stack([
            from_canonical(name, samples[:, i])
            for i, name in enumerate(full_param_order)
        ])
        reported_best_params = {
            name: from_canonical(name, value)
            for name, value in best_params.items()
        }
        reported_median_params = {
            name: from_canonical(name, value)
            for name, value in median_params.items()
        }
        null_hypothesis = dict(null_hypothesis_canonical)
        null_hypothesis['params'] = {
            name: from_canonical(name, value)
            for name, value in null_hypothesis_canonical['params'].items()
        }
        known_output_names = set(full_param_order) | set(canonical_priors)
        parameter_factors = {
            name: data.parameter_unit(name).to(
                CanonicalUnits[data.parameter_dimension(name)]
            )
            for name in known_output_names
        }

        results_dict = {}
        for i, name in enumerate(param_order):
            results_dict[name] = reported_samples[:, i]

        for i, name in enumerate([name for name in self.output_param_order if name not in param_order]):
            results_dict[name] = reported_samples[:, len(param_order) + i]

        results_dict['Ess'] = results['ess']
        results_dict['logZ'] = results['logz']
        results_dict['logZerr'] = results['logzerr']
        results_dict['param_names'] = param_order
        results_dict['raw_sampler'] = sampler
        results_dict['MAP_params'] = reported_best_params
        results_dict['median_params'] = reported_median_params
        results_dict['null_hypothesis'] = null_hypothesis
        results_dict['logl'] = logl
        results_dict['samples'] = reported_samples
        results_dict['ref_epoch'] = self._reported_ref_epoch(data)
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'Campbell'
        results_dict['priors'] = self.priors
        results_dict['canonical_priors'] = canonical_priors
        results_dict['parameter_factors'] = parameter_factors

        
        fit_results = FitResults(**results_dict)
        return fit_results

