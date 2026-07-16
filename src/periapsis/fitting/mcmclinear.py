from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.initial.initial import InitialFit
from periapsis.utils.solvers import transform_theile
from periapsis.utils.helpers import _match_param_keys
from periapsis.utils.solvers import solve_mass
import numpy as np
import matplotlib.pyplot as plt
import emcee

class MCMCLinearFitter(Fitter):
    def __init__(self,nwalkers,niter, m1=None,m2_max=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.sampled_params = ('P', 'e', 't0')
        self.m2_max = m2_max

    def fit(self, data: Data) -> FitResults:
        param_order = [name for name in self.sampled_params if name in self.prior_kwargs]
        ndim = len(param_order)

        if ndim != len(self.sampled_params):
            missing = [name for name in self.sampled_params if name not in self.prior_kwargs]
            raise ValueError(f"MCMCLinearFitter requires priors for {self.sampled_params}; missing {missing}")

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
        
            
        def _check_offsets_m2(params,mu):
            dx = mu[0]
            dpmra = mu[1]
            A = mu[2]
            F = mu[3]
            dy = mu[4]
            dpmdec = mu[5]
            B = mu[6]
            G = mu[7]

            params_dict = _match_param_keys(dict(zip(param_order, params)))
            
            for name,value in [('dx', dx), ('dpmra', dpmra), ('dy', dy), ('dpmdec', dpmdec)] :
                prior = self.prior_kwargs.get(name)
                if prior is not None and hasattr(prior, 'min') and hasattr(prior, 'max'):
                    if not (prior.min <= value <= prior.max):
                       return True
            if self.m1 is not None and self.m2_max is not None:
                a1, _, _, _ = transform_theile(A, B, F, G)
                m2 = solve_mass(a1, params_dict['P'], self.m1)
                if (m2 > self.m2_max or m2 < 0):
                    return True
            return False

        def objective(params, data):
            try:          
                params_dict = _match_param_keys(dict(zip(param_order, params)))

                dt = data.t - ref_epoch
                ti = dt - params_dict['t0'] * params_dict['P']

                M = 2 * np.pi * ti / params_dict['P']
                E = solve_kepler(M, params_dict['e'])

                mu, chi2 = matrix_method(params_dict, data, E)

                dx = mu[0]
                dpmra = mu[1]
                A = mu[2]
                F = mu[3]
                dy = mu[4]
                dpmdec = mu[5]
                B = mu[6]
                G = mu[7]
            except ZeroDivisionError:
                return -np.inf

            if _check_offsets_m2(params, mu):
                return -np.inf

            return chi2 
        

        def lnprob(params, data):

            ln_prior = 0
            for name,val in zip(param_order, params):
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    ln_prior += prior.logpdf(val)
                    if not np.isfinite(ln_prior):
                        return -np.inf
                else:
                    print(f"Warning:Missing prior for {name}.")

            


            chi2 = objective(params, data)
            if not np.isfinite(chi2):
                return -np.inf
            
            ln_likelihood = -0.5 * chi2

            return ln_prior + ln_likelihood
        
        pm_fit = self._proper_motion_fit(data)

        initial_fit = InitialFit(
                data,
                method='Campbell',
                **self.prior_kwargs,
            ).get_intial()

        P0 = initial_fit['P']
        e0 = initial_fit['e']
        T00 = initial_fit['t0']

        initial_params = [P0, e0, T00]
        
        bounds = np.array(
            [[self.prior_kwargs[name].min, self.prior_kwargs[name].max] for name in param_order],
            dtype=float,
        )
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        initial_params = np.clip(np.asarray(initial_params, dtype=float), lower, upper)
        
        initial = np.clip(initial_params + np.random.randn(self.nwalkers,ndim) * 1e-2, lower, upper)

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, lnprob, args=(data,))
        sampler.run_mcmc(initial, self.niter,progress=True)

        chain = sampler.get_chain()
        param_means = chain.mean(axis=1)

        tau = emcee.autocorr.integrated_time(chain,quiet=True)

        Ess = (self.niter*self.nwalkers)/tau

        maf = np.mean(sampler.acceptance_fraction)

        burn = int(np.nanmax(tau) * 2)
        thin = int(np.nanmin(tau) * 2)

        samples = sampler.get_chain(discard=burn,thin=thin,flat=True)
        lnprobs = sampler.get_log_prob(discard=burn,thin=thin,flat=True)

       
        
        full_posterior = [] 
        for param in samples:
            P,e,t0 = param
            M = 2*np.pi * (data.t - ref_epoch - t0*P) / P
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
            full_posterior.append((P,e,t0,A,B,F,G,dx,dy,dpmra,dpmdec))

        post_labels = ['P','e','t0','A','B','F','G','dx','dy','dpmra','dpmdec']

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
        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results
        
        
       

        

                
