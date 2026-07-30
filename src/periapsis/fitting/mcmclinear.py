import warnings

from periapsis.data import Data, AstrometryData, RadialVelocityData, GaiaData, JointData

from .fitter import Fitter
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.initial import InitialGuess, AstrometryInitialGuess, RVInitialGuess, GaiaInitialGuess, JointInitialGuess
from periapsis.prior import FixedPrior, Bounds
from periapsis.utils.solvers import transform_theile
from periapsis.utils.solvers import solve_mass
from periapsis.params.transforms import covered_parameters, build_transform_functions
import numpy as np
import matplotlib.pyplot as plt
import emcee
from typing import Type

class MCMCLinearFitter(Fitter):
    def __init__(self, nwalkers, niter, sampled_params=('P', 'e', 'Tp'), **priors):
        super().__init__(**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.covered_params = covered_parameters(sampled_params)
        if any(param not in self.covered_params for param in ('P', 'e', 'Tp')):
            raise ValueError("MCMCGaia requires sampled_params to define 'P', 'e', and 'Tp'.")
        # TODO - Raise a warning if user is sampling more params than necessary
        self.sampled_params = frozenset(sampled_params)
        self.param_order = tuple(sampled_params)
        if len(self.sampled_params) != len(self.param_order):
            raise ValueError("Duplicate parameters found in sampled_params.")
        self.fixed_prior_params = {p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)}

        self.prior_covered_params = covered_parameters(self.priors.keys())
        if any(param not in self.prior_covered_params for param in sampled_params):
            missing = [param for param in sampled_params if param not in self.prior_covered_params]
            raise ValueError(f"Missing priors for sampled parameters: {missing}")


    def fit(self, data: Data, rng: np.random.RandomState, initial: Type[InitialGuess] = None) -> FitResults:
        if not isinstance(data, AstrometryData):
            raise ValueError("MCMCLinearFitter currently supports AstrometryData.")

        param_order = self.param_order
        param_transforms = build_transform_functions(param_order, ('P', 'e', 'Tp',))
        ndim = len(param_order)

        # TODO normalize ref_epoch
        ref_epoch = getattr(data, 'ref_epoch', 0.0)
        
        def matrix_method(params_dict,data,E):
            
            nobs = len(data.t)
            dt = data.t - ref_epoch

            M = np.zeros((2*nobs,8))

            eta =np.concatenate((data.x,data.y))
            sigma = np.concatenate((data.x_err,data.y_err))
    

            X = np.cos(E) - params_dict['e']
            Y = np.sqrt(1-params_dict['e']**2)*np.sin(E)

            M[:nobs,0] = 1 #dx
            M[:nobs,1] = dt #pmra
            M[:nobs,2] = X # A
            M[:nobs,3] = Y # F

            # now bottom half y obs
            M[nobs:,4] = 1 #dy
            M[nobs:,5] = dt #pmdec
            M[nobs:,6] = X # B
            M[nobs:,7] = Y # G

            #now we need to get covariance matrix
            # which diagnol matrix, with err_x^2 on top and err_y^2 on bottom
            # so we can just say C^-1 is equivalent to (A*w) ....
            w = 1/sigma # just do 1/sigma to keep track of where the weights have been applied
            # now we can calculate M^T C^-1 M and M^T C^-1 eta
            eta_w = eta * w 
            M_w = M * w[:, None] # multiply each row of M by corresponding weight

            MTM = M_w.T @ M_w
            MT_eta = M_w.T @ eta_w # matching equation
            # now we can solve for mu using np.linalg.solve
            mu = np.linalg.solve(MTM, MT_eta) # dx,pmra,B,G,dy,pmdec,A,F

            model_werr = M_w @ mu # this is the model prediction with the error already over
            # this is (obs - model)/err
            resids = eta_w - model_werr
            chi2 = np.sum(resids**2)
            
            return mu, chi2
    
        

        def lnprob(params, data):
            params_dict = param_transforms(**dict(zip(param_order,params)))

            dt = data.t - ref_epoch
            ti = dt - params_dict['Tp']

            M = 2 * np.pi * ti / params_dict['P']
            E = solve_kepler(M, params_dict['e'])

            mu, chi2 = matrix_method(params_dict, data, E)

            ln_prior = 0.0

            for name, prior in self.priors.items():
                if isinstance(prior, FixedPrior):
                    continue  # Skip fixed priors
                try:
                    transform = build_transform_functions([*self.sampled_params, "dx", "dpmra", f"A{data.system}", f"F{data.system}", "dy", "dpmdec", f"B{data.system}", f"G{data.system}", *self.fixed_prior_params], [name])
                    val = transform(
                        **{name: params[i] for i, name in enumerate(self.param_order)},
                        **{**{name: self.priors[name].value for name in self.fixed_prior_params}, "dx": mu[0], "dpmra": mu[1], f"A{data.system}": mu[2], f"F{data.system}": mu[3], "dy": mu[4], "dpmdec": mu[5], f"B{data.system}": mu[6], f"G{data.system}": mu[7]},
                    )[name]
                    if not np.isfinite(val):
                        return -np.inf
                    ln_prior += prior.logpdf(val)
                except KeyError:
                    # If the parameter is not reachable from the sampled_params or mu, skip it
                    warnings.warn(f"Parameter {name} is not reachable from sampled_params or mu. Skipping prior evaluation for this parameter.")
                    continue

            if not np.isfinite(chi2):
                return -np.inf
            
            ln_likelihood = -0.5 * chi2

            return ln_prior + ln_likelihood
        
        if isinstance(data, AstrometryData):
            pm_fit = self._proper_motion_fit(data)
        else:
            pm_fit = dict()

        if initial is None:
            if isinstance(data, AstrometryData):
                initial = AstrometryInitialGuess
            elif isinstance(data, RadialVelocityData):
                initial = RVInitialGuess
            elif isinstance(data, GaiaData):
                initial = GaiaInitialGuess
            elif isinstance(data, JointData):
                initial = JointInitialGuess
            else:
                raise ValueError("No initial guess class provided and data type is not recognized for linearized MCMC initial guess generation.")
        initial_instance = initial(data, rng, **self.priors)
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
            transformed_param = param_transforms(**dict(zip(param_order, param)))
            P,e,Tp = transformed_param["P"], transformed_param["e"], transformed_param["Tp"]
            M = 2*np.pi * (data.t - ref_epoch - Tp) / P
            E = solve_kepler(M,e)
            mu, _ = matrix_method({'e': e}, data, E)
            dx = mu[0]
            dpmra = mu[1]
            A = mu[2]
            F = mu[3]
            dy = mu[4]
            dpmdec = mu[5]
            B = mu[6]
            G = mu[7]
            full_posterior.append((*param,A,B,F,G,dx,dy,dpmra,dpmdec))

        post_labels = [*param_order,f'A{data.system}',f'B{data.system}',f'F{data.system}',f'G{data.system}','dx','dy','dpmra','dpmdec']

        best_i = np.argmax(lnprobs)
        best_params = dict(zip(post_labels, full_posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(full_posterior, axis=0)))

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
        results_dict['PM_fit'] = pm_fit
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
        
        
       

        

                
