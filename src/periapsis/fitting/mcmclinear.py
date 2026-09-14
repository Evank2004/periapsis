import warnings

from periapsis.data import Data, AstrometryData, RadialVelocityData, GaiaData, JointData

from .fitter import Fitter
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.utils.helpers import _matrix_builder, _matrix_filler, _lsq_helper, _sigma,_jitter_check
from periapsis.initial import InitialGuess, AstrometryLinearInitialGuess, RVInitialGuess, GaiaInitialGuess, JointInitialGuess
from periapsis.prior import Bounds, FixedPrior
from periapsis.params.transforms import covered_parameters, build_transform_functions
from periapsis.params.units import CanonicalUnits
import periapsis.params as par
import numpy as np
import emcee
from typing import Type

class MCMCLinearFitter(Fitter):
    def __init__(self, nwalkers, niter, sampled_params=('P', 'e', 'Tp'), ref_epoch=None, **priors):
        super().__init__(ref_epoch,**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.fixed_prior_params = {p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)}
        sampled_params = tuple(sampled_params)
        sampled_jitter_params = tuple(
            name
            for name, prior in self.priors.items()
            if name.endswith("_jitter")
            and not isinstance(prior, (Bounds, FixedPrior))
            and name not in sampled_params
        )
        sampled_flux_params = tuple(
            name
            for name, prior in self.priors.items()
            if name.startswith('f_')
            and not isinstance(prior, (Bounds, FixedPrior))
            and name not in sampled_params
        )
        sampled_params = (*sampled_params, *sampled_jitter_params, *sampled_flux_params)
        self.covered_params = covered_parameters({*sampled_params, *self.fixed_prior_params})
        if any(param not in self.covered_params for param in (par.P, par.e, par.Tp)):
            raise ValueError("MCMCLinearFitter requires sampled_params to define 'P', 'e', and 'Tp'.")
        # TODO - Raise a warning if user is sampling more params than necessary
        self.sampled_params = frozenset(sampled_params)
        self.param_order = tuple(sampled_params)
        if len(self.sampled_params) != len(self.param_order):
            raise ValueError("Duplicate parameters found in sampled_params.")
        self.prior_covered_params = covered_parameters(self.priors.keys())
        if any(param not in self.prior_covered_params for param in sampled_params):
            missing = [param for param in sampled_params if param not in self.prior_covered_params]
            raise ValueError(f"Missing priors for sampled parameters: {missing}")
        self.early_prior_params = {p for p in self.priors.keys() if not isinstance(self.priors[p], FixedPrior) and p in self.covered_params}
        self.late_prior_params = {p for p in self.priors.keys() if not isinstance(self.priors[p], FixedPrior) and p not in self.covered_params}
        

    def fit(self, data: Data, rng: np.random.RandomState, initial: Type[InitialGuess] = None) -> FitResults:
        if not isinstance(data, (AstrometryData, RadialVelocityData, JointData, GaiaData)):
            raise ValueError("MCMCLinearFitter supports AstrometryData, RadialVelocityData, JointData, and GaiaData.")

        canonical_priors = self._canonical_priors(data)
        canonical_ref_epoch = canonical_priors['Tepoch'].value
        param_order = self.param_order
        transform_args = [par.P, par.e, par.Tp]
        known_flux_params = (*param_order, *self.fixed_prior_params)
        if any(name.startswith('f_') for name in known_flux_params):
            transform_args.append(par.q)

        param_transforms = build_transform_functions({*param_order, *self.fixed_prior_params}, transform_args)
        ndim = len(param_order)

        self.M, self.cols = _matrix_builder(data, canonical_ref_epoch)


        # Matrix method cached variables
        if isinstance(data, AstrometryData):
            mm_eta = np.concatenate((data.x, data.y))
            mm_sigma = np.concatenate((data.x_err, data.y_err))
            
            
        
        
        elif isinstance(data, RadialVelocityData):
            mm_eta = data.rv
            mm_sigma = data.rv_err
            
           

        elif isinstance(data,JointData):
            mm_eta = data._concat_obs()
            mm_sigma = data._err()
            
        elif isinstance(data,GaiaData):
            mm_eta = data.x
            mm_sigma = data.err
            

        def matrix_method(params_dict):
            
            _matrix_filler(self.M,self.cols,params_dict,data)

            sigma = _sigma(data, params_dict, mm_sigma)
            mu,chi2 = _lsq_helper(self.M,mm_eta,sigma)

             
            return mu, chi2, sigma
    

        early_prior_transforms = build_transform_functions([*self.sampled_params, *self.fixed_prior_params], self.early_prior_params)
        late_prior_transforms = build_transform_functions([*self.sampled_params, *self.cols.keys(),*self.fixed_prior_params], self.late_prior_params)
        def lnprob(params, data):
            # Evaulate priors for parameters that don't need full orbit solution, short circuiting if any are invalid
            ln_prior = 0.0
            early_transformed = early_prior_transforms(
                **dict(zip(param_order, params)),
                **{name: canonical_priors[name].value for name in self.fixed_prior_params}
            )
            for name in self.early_prior_params:
                val = early_transformed[name]
                if not np.isfinite(val):
                    return -np.inf
                ln_prior += canonical_priors[name].logpdf(val)
            
            # Calculate full orbit solution and chi2
            params_dict = dict(zip(param_order, params))
            params_dict.update({
                name: canonical_priors[name].value
                for name in self.fixed_prior_params
            })
            params_dict.update(param_transforms(**params_dict))

            mu, chi2, sigma = matrix_method(params_dict)

            # Evaluate priors for parameters that require full orbit solution, short circuiting if any are invalid
            
            late_transformed = late_prior_transforms(
                **dict(zip(param_order, params)),
                **{**{name: canonical_priors[name].value for name in self.fixed_prior_params},
                    **{name:mu[self.cols[name]] for name in self.cols }})
                
            for name in self.late_prior_params:
                val = late_transformed[name]
                if not np.isfinite(val):
                    return -np.inf
                ln_prior += canonical_priors[name].logpdf(val)

            if not np.isfinite(chi2):
                return -np.inf
            
            
            jitter_check = _jitter_check(data, params_dict)

            if np.any(jitter_check):
                ln_likelihood = -0.5 * (chi2
                    + np.sum(np.log(2 * np.pi * sigma ** 2))
                )
            else:
                ln_likelihood = -0.5 * chi2

            return ln_prior + ln_likelihood
        
        null_hypothesis_canonical = self._null_hypothesis_fit(
            data, ref_epoch=canonical_ref_epoch
        )

        if initial is None:
            if isinstance(data, AstrometryData):
                initial = AstrometryLinearInitialGuess
            elif isinstance(data, RadialVelocityData):
                initial = RVInitialGuess
            elif isinstance(data, GaiaData):
                initial = GaiaInitialGuess
            elif isinstance(data, JointData):
                initial = JointInitialGuess
            else:
                raise ValueError("No initial guess class provided and data type is not recognized for linearized MCMC initial guess generation.")
        initial_instance = initial(data, rng, canonical_ref_epoch, **canonical_priors)
        pos = initial_instance.get_initial_guess(param_order, self.nwalkers)
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, lnprob, args=(data,))
        sampler.run_mcmc(pos, self.niter,progress=True)

        chain = sampler.get_chain()
        param_means = chain.mean(axis=1)

        tau = emcee.autocorr.integrated_time(chain,quiet=True)

        Ess = (self.niter*self.nwalkers)/tau

        maf = np.mean(sampler.acceptance_fraction)

        nanmaxtau = np.nanmax(tau)
        nanmintau = np.nanmin(tau)

        if not np.isnan(nanmaxtau):
            burn = int(np.nanmax(tau) * 2)
        else:
            warnings.warn("Autocorrelation time could not be estimated. Setting burn-in to 0.")
            burn = 0

        if not np.isnan(nanmintau):
            thin = int(np.nanmin(tau) * 2)
        else:
            warnings.warn("Autocorrelation time could not be estimated. Setting thinning to 1.")
            thin = 1

        samples = sampler.get_chain(discard=burn,thin=thin,flat=True)
        lnprobs = sampler.get_log_prob(discard=burn,thin=thin,flat=True)

       
        
        full_posterior = [] 
        

        for param in samples:
            transformed_param = dict(zip(param_order, param))
            transformed_param.update({
                name: canonical_priors[name].value
                for name in self.fixed_prior_params
            })
            transformed_param.update(param_transforms(**transformed_param))

            mu, _, _ = matrix_method(transformed_param)
            full_posterior.append((*param,*(mu[self.cols[name]] for name in self.cols)))

        post_labels = [*param_order, *self.cols.keys()]

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(post_labels, full_posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(full_posterior, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = canonical_priors[prior].value
            median_params[prior] = canonical_priors[prior].value

        if null_hypothesis_canonical is None:
            raise RuntimeError("Null-hypothesis fitting returned no result.")

        def from_canonical(name, value):
            unit = data.parameter_unit(name)
            dimension = data.parameter_dimension(name)
            factor = unit.to(CanonicalUnits[dimension])
            return np.asarray(value) / factor

        full_posterior_array = np.asarray(full_posterior, dtype=float)
        reported_posterior = np.column_stack([
            from_canonical(name, full_posterior_array[:, i])
            for i, name in enumerate(post_labels)
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
        parameter_factors = {
            name: data.parameter_unit(name).to(
                CanonicalUnits[data.parameter_dimension(name)]
            )
            for name in set(post_labels) | set(canonical_priors)
        }

        columns = {label: [] for label in post_labels}
        for sample in reported_posterior:
            for label, value in zip(post_labels, sample):
                columns[label].append(value)

        results_dict: dict[str, object] = {label: np.array(columns[label]) for label in post_labels}

        results_dict['lnprob'] = lnprobs
        results_dict['Ess'] = Ess
        results_dict['mean_acceptance_fraction'] = maf
        results_dict['tau'] = tau
        results_dict['param_means'] = np.column_stack([
            from_canonical(name, param_means[:, i])
            for i, name in enumerate(param_order)
        ])
        results_dict['param_names'] = post_labels
        results_dict['MAP_params'] = reported_best_params
        results_dict['median_params'] = reported_median_params
        results_dict['null_hypothesis'] = null_hypothesis
        results_dict['ref_epoch'] = self._reported_ref_epoch(data)
        
        results_dict['raw_sampler'] = None
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'linear'
        results_dict['priors'] = self.priors
        results_dict['canonical_priors'] = canonical_priors
        results_dict['parameter_factors'] = parameter_factors
        fit_results = FitResults(**results_dict)
        return fit_results
        
        
       

        

                
