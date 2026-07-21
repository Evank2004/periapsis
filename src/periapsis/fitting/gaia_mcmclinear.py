from .fitter import Fitter
from periapsis.data.gaia import GaiaData
from periapsis.fitting.results import FitResults
from periapsis.initial.gaia_initial import GaiaInitialFit
from periapsis.utils.helpers import _match_param_keys
from periapsis.utils.solvers import gaia_single_motion
from periapsis.utils.solvers import solve_kepler
import numpy as np
import emcee


class MCMCGaia(Fitter):
    def __init__(self,nwalkers,niter,pool=None,**priors):
        super().__init__(**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.sampled_params = ('P', 'e', 't0')

    def fit(self,data: GaiaData) -> FitResults:
        """Fit the Gaia data using MCMC"""
        param_order = [name for name in self.sampled_params if name in self.prior_kwargs]
        ndim = len(param_order)
        mu_single = gaia_single_motion(data.spsi,data.cpsi,data.t,data.plx_fac,data.x,data.err)
        



        def matrix_method(params_dict,data):
            P,e,t0 = params_dict['P'],params_dict['e'],params_dict['t0']

            nobs = len(data.t)

            ti = data.t - t0*P

            M = 2*np.pi * ti/P
            E = solve_kepler(M,e)

            X = np.cos(E) - e
            Y = np.sqrt(1-e**2)*np.sin(E)

            A = np.column_stack([
                data.spsi, #delta_alpha
                data.cpsi, #delta_delta
                data.plx_fac, #parallax
                data.spsi*data.t, #mu_alpha
                data.cpsi*data.t, #mu_delta
                X*data.spsi, #B
                Y*data.spsi, #G
                X*data.cpsi, #A
                Y*data.cpsi  #F
            ])

            w = 1.0 / data.err
            x_w = data.x * w
            A_w = A * w[:, None]

            ATA = A_w.T @ A_w
            ATx = A_w.T @ x_w

            mu = np.linalg.solve(ATA, ATx) # [delta alpha,delta delta, parallax,mu_alpha,mu_delta,dx,dpmra,dy,dpmdec,B,G,A,F]

            model_werr = A_w @ mu

            cov_mu = np.linalg.inv(ATA)
            mu_err = np.sqrt(np.diag(cov_mu))

            residuals = x_w - model_werr
            chi2 = np.sum(residuals**2)

            return mu, mu_err, chi2
   
        def objective(theta,data):
            params_dict = dict(zip(param_order, theta))
            _, _, chi2 = matrix_method(params_dict,data)

            for name, val in params_dict.items():
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    lp = prior.logpdf(val)
                    if not np.isfinite(lp):
                        return -np.inf
                else:
                    print(f"Warning:Missing prior for {name}.")

            if not np.isfinite(chi2):
                return -np.inf
            
            ln_like = -0.5 * chi2

            return ln_like
        

        initial_fit = GaiaInitialFit(data, **self.prior_kwargs).initial_guess()
            
        P0 = initial_fit['P']
        e0 = initial_fit['e']
        T00 = initial_fit['t0']

        initial_set = [P0, e0, T00]

        bounds = np.array(
            [[self.prior_kwargs[name].min, self.prior_kwargs[name].max] for name in param_order],
            dtype=float,
        )
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        initial_params = np.clip(np.asarray(initial_set, dtype=float), lower, upper)
        
        initial = np.clip(initial_params + np.random.randn(self.nwalkers,ndim) * 1e-2, lower, upper)


        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, objective, args=(data), pool=self.pool)
        sampler.run_mcmc(initial, self.niter, progress=True)

        chain = sampler.get_chain()
        param_means = chain.mean(axis=1)

        tau = emcee.autocorr.integrated_time(chain,quiet=True)

        Ess = (self.niter*self.nwalkers)/tau

        maf = np.mean(sampler.acceptance_fraction)

        burn = int(np.nanmax(tau) * 2)
        thin = int(np.nanmin(tau) * 2)

        samples = sampler.get_chain(discard=burn,thin=thin,flat=True)
        lnprobs = sampler.get_log_prob(discard=burn,thin=thin,flat=True)

       
        posterior = []
        valid_logp = []
        for sample, l_prob in zip(samples, lnprobs):
            params_dict = dict(zip(param_order, sample))
            mu, _, _ = matrix_method(params_dict, data)
            
            if mu is None:
                continue
            
            P, e, t0 = sample
            delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, B, G, A, F = mu
            
            posterior.append([P, e, t0, delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, A, B, F, G])
            valid_logp.append(l_prob)
       
        post_labels = ['P','e','t0','dalpha','ddelta','parallax','mu_alpha','mu_delta','A','B','F','G']

        best_i = np.argmax(valid_logp)
        best_params = dict(zip(post_labels, posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(posterior, axis=0)))

        columns = {label: [] for label in post_labels}
        for sample in posterior:
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
        results_dict['Single_motion_params'] = mu_single
        # results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        
        results_dict['raw_sampler'] = None
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'linear'
        fit_results = FitResults(**results_dict)
       
        return fit_results