import warnings

from periapsis.data import Data, AstrometryData, RadialVelocityData, GaiaData, JointData

from .fitter import Fitter
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.utils.helpers import _matrix_builder,_matrix_filler,_lsq_helper
from periapsis.initial import InitialGuess, AstrometryLinearInitialGuess, RVInitialGuess, GaiaInitialGuess, JointInitialGuess
from periapsis.prior import FixedPrior
from periapsis.params.transforms import covered_parameters, build_transform_functions
import periapsis.params as par
import numpy as np
import emcee
from typing import Type

class MCMCLinearFitter(Fitter):
    def __init__(self, nwalkers, niter, sampled_params=('P', 'e', 'Tp'), ref_epoch=0, **priors):
        super().__init__(ref_epoch,**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.fixed_prior_params = {p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)}
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

        param_order = self.param_order
        param_transforms = build_transform_functions({*param_order, *self.fixed_prior_params}, (par.P, par.e, par.Tp,))
        ndim = len(param_order)

        self.M, self.cols = _matrix_builder(data, self.ref_epoch)


        # Matrix method cached variables
        if isinstance(data, AstrometryData):
            mm_eta = np.concatenate((data.x, data.y))
            mm_sigma = np.concatenate((data.x_err, data.y_err))
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            
        
        
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

            mu,chi2 = _lsq_helper(self.M,mm_eta,mm_sigma)

             
            return mu, chi2
    

        early_prior_transforms = build_transform_functions([*self.sampled_params, *self.fixed_prior_params], self.early_prior_params)
        late_prior_transforms = build_transform_functions([*self.sampled_params, *self.cols.keys(),*self.fixed_prior_params], self.late_prior_params)
        def lnprob(params, data):
            # Evaulate priors for parameters that don't need full orbit solution, short circuiting if any are invalid
            ln_prior = 0.0
            early_transformed = early_prior_transforms(
                **dict(zip(param_order, params)),
                **{name: self.priors[name].value for name in self.fixed_prior_params}
            )
            for name in self.early_prior_params:
                val = early_transformed[name]
                if not np.isfinite(val):
                    return -np.inf
                ln_prior += self.priors[name].logpdf(val)
            
            # Calculate full orbit solution and chi2
            params_dict = param_transforms(**dict(zip(param_order,params)), **{name: self.priors[name].value for name in self.fixed_prior_params})

            mu, chi2 = matrix_method(params_dict)

            # Evaluate priors for parameters that require full orbit solution, short circuiting if any are invalid
            
            late_transformed = late_prior_transforms(
                **dict(zip(param_order, params)),
                **{**{name: self.priors[name].value for name in self.fixed_prior_params},
                    **{name:mu[self.cols[name]] for name in self.cols }})
                
            for name in self.late_prior_params:
                val = late_transformed[name]
                if not np.isfinite(val):
                    return -np.inf
                ln_prior += self.priors[name].logpdf(val)

            if not np.isfinite(chi2):
                return -np.inf
            
            ln_likelihood = -0.5 * chi2

            return ln_prior + ln_likelihood
        
        null_hypothesis = self._null_hypothesis_fit(data)

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
        initial_instance = initial(data, rng, self.ref_epoch, **self.priors)
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
            transformed_param = param_transforms(**dict(zip(param_order, param)), **{name: self.priors[name].value for name in self.fixed_prior_params})
            
            mu, _ = matrix_method(transformed_param)
            full_posterior.append((*param,*(mu[self.cols[name]] for name in self.cols)))

        post_labels = [*param_order, *self.cols.keys()]

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(post_labels, full_posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(full_posterior, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = self.priors[prior].value
            median_params[prior] = self.priors[prior].value

        columns = {label: [] for label in post_labels}
        for sample in full_posterior:
            for label, value in zip(post_labels, sample):
                columns[label].append(value)

        results_dict: dict[str, object] = {label: np.array(columns[label]) for label in post_labels}

        results_dict['lnprob'] = lnprobs
        results_dict['Ess'] = Ess
        results_dict['mean_acceptance_fraction'] = maf
        results_dict['tau'] = tau
        results_dict['param_means'] = param_means
        results_dict['param_names'] = post_labels
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['null_hypothesis'] = null_hypothesis
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        
        results_dict['raw_sampler'] = None
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'linear'
        results_dict['priors'] = self.priors
        # TODO: normalize ref_epoch
        if results_dict['ref_epoch'] is not None:
            results_dict['priors']['Tepoch'] = FixedPrior(results_dict['ref_epoch'])
        fit_results = FitResults(**results_dict)
        return fit_results
        
        
       

        

                
