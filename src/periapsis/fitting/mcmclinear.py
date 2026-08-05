import warnings

from periapsis.data import Data, AstrometryData, RadialVelocityData, GaiaData, JointData

from .fitter import Fitter
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.initial import InitialGuess, AstrometryLinearInitialGuess, RVInitialGuess, GaiaInitialGuess, JointInitialGuess
from periapsis.prior import FixedPrior
from periapsis.params.transforms import covered_parameters, build_transform_functions
import numpy as np
import emcee
from typing import Type

class MCMCLinearFitter(Fitter):
    def __init__(self, nwalkers, niter, sampled_params=('P', 'e', 'Tp'), **priors):
        super().__init__(**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.fixed_prior_params = {p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)}
        self.covered_params = covered_parameters({*sampled_params, *self.fixed_prior_params})
        if any(param not in self.covered_params for param in ('P', 'e', 'Tp')):
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
        if isinstance(data,GaiaData):
            raise ValueError("MCMCLinearFitter does not support GaiaData. Use MCMCGaiaFitter instead.")   

        param_order = self.param_order
        param_transforms = build_transform_functions({*param_order, *self.fixed_prior_params}, ('P', 'e', 'Tp',))
        ndim = len(param_order)

        # TODO normalize ref_epoch
        ref_epoch = getattr(data, 'ref_epoch', 0.0)

        # Matrix method cached variables
        if isinstance(data, AstrometryData):
            mm_nobs = len(data.t)
            mm_dt = data.t - ref_epoch
            mm_M = np.zeros((2*mm_nobs, 8))
            mm_M[:mm_nobs,0] = 1 # dx
            mm_M[:mm_nobs,1] = mm_dt # pmra
            mm_M[mm_nobs:,4] = 1 # dy
            mm_M[mm_nobs:,5] = mm_dt # pmdec

            mm_eta = np.concatenate((data.x, data.y))
            mm_sigma = np.concatenate((data.x_err, data.y_err))
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            data_type = 'astrometry'
        
        
        elif isinstance(data, RadialVelocityData):
            mm_nobs = len(data.t)
            mm_dt = data.t - ref_epoch
            mm_M = np.zeros((mm_nobs, 3))
            mm_M[:,0] = 1 # gamma

            mm_eta = data.rv
            mm_sigma = data.rv_err
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            data_type = 'rv'

        elif isinstance(data,JointData):
            astro_data = data.as_astrometry_data()
            rv_data = data.as_radial_velocity_data()
            astro_nobs = len(astro_data.t)
            rv_nobs = len(rv_data.t)
            dt_astro = astro_data.t - ref_epoch
            dt_rv = rv_data.t - ref_epoch

            n_rows = 2*astro_nobs + rv_nobs
            ncols = 11

            mm_M = np.zeros((n_rows, ncols))
            mm_M[:astro_nobs,0] = 1 # dx
            mm_M[:astro_nobs,1] = dt_astro # pmra
            mm_M[astro_nobs:2*astro_nobs,4] = 1 # dy
            mm_M[astro_nobs:2*astro_nobs,5] = dt_astro # pmdec
            mm_M[2*astro_nobs:,8] = 1 # gamma

            mm_eta = np.concatenate((astro_data.x, astro_data.y, rv_data.rv))
            mm_sigma = np.concatenate((astro_data.x_err, astro_data.y_err, rv_data.rv_err))
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            data_type = 'joint'



        def _orbit_coord_func(params_dict,data):
            dt = data.t - ref_epoch
            ti = dt - params_dict['Tp']
            M = 2 * np.pi * ti / params_dict['P']
            E = solve_kepler(M, params_dict['e'])
            X = np.cos(E) - params_dict['e']
            Y = np.sqrt(1-params_dict['e']**2)*np.sin(E)
            nu = 2 * np.arctan2(np.sqrt(1+params_dict['e'])*np.sin(E/2), np.sqrt(1-params_dict['e'])*np.cos(E/2))
            return X,Y,nu

        def matrix_method(params_dict):
            
            if data_type == 'astrometry':
                X,Y,_ = _orbit_coord_func(params_dict,data)
                # M[:nobs,0] = 1 #dx
                # M[:nobs,1] = dt #pmra
                mm_M[:mm_nobs,2] = X # A
                mm_M[:mm_nobs,3] = Y # F

                # now bottom half y obs
                # M[nobs:,4] = 1 #dy
                # M[nobs:,5] = dt #pmdec
                mm_M[mm_nobs:,6] = X # B
                mm_M[mm_nobs:,7] = Y # G

            elif data_type == 'rv':
                _,_,nu = _orbit_coord_func(params_dict,data)
                mm_M[mm_nobs,1] = np.cos(nu) + params_dict['e'] # h
                mm_M[mm_nobs,2] = np.sin(nu) # c

            elif data_type == 'joint':
                X_a,Y_a,_ = _orbit_coord_func(params_dict,data._astrometry)
                _,_,nu_rv = _orbit_coord_func(params_dict,data._radial_velocity)
                # astrometry part
                mm_M[:astro_nobs,2] = X_a # A
                mm_M[:astro_nobs,3] = Y_a # F
                mm_M[astro_nobs:2*astro_nobs,6] = X_a # B
                mm_M[astro_nobs:2*astro_nobs,7] = Y_a # G
                # rv part
                mm_M[2*astro_nobs:,9] = np.cos(nu_rv) + params_dict['e'] # h
                mm_M[2*astro_nobs:,10] = np.sin(nu_rv) # c



            #now we need to get covariance matrix
            # which diagnol matrix, with err_x^2 on top and err_y^2 on bottom
            # so we can just say C^-1 is equivalent to (A*w) ....
            # w = 1/sigma # just do 1/sigma to keep track of where the weights have been applied
            # now we can calculate M^T C^-1 M and M^T C^-1 eta
            # eta_w = eta * w 
            M_w = mm_M * mm_w[:, None] # multiply each row of M by corresponding weight

            MTM = M_w.T @ M_w
            MT_eta = M_w.T @ mm_eta_w # matching equation
            # now we can solve for mu using np.linalg.solve
            mu = np.linalg.solve(MTM, MT_eta) # dx,pmra,B,G,dy,pmdec,A,F

            model_werr = M_w @ mu # this is the model prediction with the error already over
            # this is (obs - model)/err
            resids = mm_eta_w - model_werr
            chi2 = np.sum(resids**2)
            
    
            
            return mu, chi2
    

        early_prior_transforms = build_transform_functions([*self.sampled_params, *self.fixed_prior_params], self.early_prior_params)
        late_prior_transforms = build_transform_functions([*self.sampled_params, "dx", "dpmra", f"A{data.system}", f"F{data.system}", "dy", "dpmdec", f"B{data.system}", f"G{data.system}", *self.fixed_prior_params], self.late_prior_params)
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
            if data_type == 'astrometry':
                late_transformed = late_prior_transforms(
                    **dict(zip(param_order, params)),
                    **{**{name: self.priors[name].value for name in self.fixed_prior_params},
                       "dx": mu[0], "dpmra": mu[1], f"A{data.system}": mu[2], f"F{data.system}": mu[3], "dy": mu[4], "dpmdec": mu[5], f"B{data.system}": mu[6], f"G{data.system}": mu[7]}
                )
            elif data_type == 'rv':
                late_transformed = late_prior_transforms(
                    **dict(zip(param_order, params)),
                    **{**{name: self.priors[name].value for name in self.fixed_prior_params},
                       "gamma": mu[0], f"h{data.system}": mu[1], f"c{data.system}": mu[2]}
                )
            elif data_type == 'joint':
                late_transformed = late_prior_transforms(
                    **dict(zip(param_order, params)),
                    **{**{name: self.priors[name].value for name in self.fixed_prior_params},
                       "dx": mu[0], "dpmra": mu[1], f"A{data.system}": mu[2], f"F{data.system}": mu[3], "dy": mu[4], "dpmdec": mu[5], f"B{data.system}": mu[6], f"G{data.system}": mu[7], "gamma": mu[8], f"h{data.system}": mu[9], f"c{data.system}": mu[10]}
                )
            
            for name in self.late_prior_params:
                val = late_transformed[name]
                if not np.isfinite(val):
                    return -np.inf
                ln_prior += self.priors[name].logpdf(val)

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
                initial = AstrometryLinearInitialGuess
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
        if data_type == 'astrometry':

            for param in samples:
                transformed_param = param_transforms(**dict(zip(param_order, param)), **{name: self.priors[name].value for name in self.fixed_prior_params})
                
                mu, _ = matrix_method(transformed_param)
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

        elif data_type == 'rv':
            for param in samples:
                transformed_param = param_transforms(**dict(zip(param_order, param)), **{name: self.priors[name].value for name in self.fixed_prior_params})
                
                mu, _ = matrix_method(transformed_param)
                gamma = mu[0]
                h = mu[1]
                c = mu[2]
                full_posterior.append((*param,h,c,gamma))

            post_labels = [*param_order,f'h{data.system}',f'c{data.system}','gamma']

        elif data_type == 'joint':
            for param in samples:
                transformed_param = param_transforms(**dict(zip(param_order, param)), **{name: self.priors[name].value for name in self.fixed_prior_params})
                
                mu, _ = matrix_method(transformed_param)
                dx = mu[0]
                dpmra = mu[1]
                A = mu[2]
                F = mu[3]
                dy = mu[4]
                dpmdec = mu[5]
                B = mu[6]
                G = mu[7]
                gamma = mu[8]
                h = mu[9]
                c = mu[10]
                full_posterior.append((*param,A,B,F,G,dx,dy,dpmra,dpmdec,h,c,gamma))

            post_labels = [*param_order,f'A{data.system}',f'B{data.system}',f'F{data.system}',f'G{data.system}','dx','dy','dpmra','dpmdec',f'h{data.system}',f'c{data.system}','gamma']

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
        
        
       

        

                
