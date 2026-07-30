import warnings

from .fitter import Fitter
from periapsis.data.gaia import GaiaData
from periapsis.fitting.results import FitResults
from periapsis.initial.gaia_initial import GaiaInitialGuess
from periapsis.utils.solvers import gaia_single_motion
from periapsis.utils.solvers import solve_kepler
from periapsis.prior.fixed_prior import FixedPrior
from periapsis.prior.log_uniform_prior import LogUniformPrior
from periapsis.params.transforms import covered_parameters, build_transform_functions
import numpy as np
import emcee

class MCMCGaiaFitter(Fitter):
    def __init__(self, nwalkers, niter, pool=None, sampled_params=('P', 'e', 'Tp'), jitter=None, **priors):
        super().__init__(**priors)
        self.nwalkers = nwalkers
        self.niter = niter
        self.pool = pool
        self.covered_params = covered_parameters(sampled_params)
        if any(param not in self.covered_params for param in ('P', 'e', 'Tp')):
            raise ValueError("MCMCGaia requires sampled_params to define 'P', 'e', and 'Tp'.")

        # FIXME - don't always sample jitter.
        if isinstance(jitter, FixedPrior):
            self.jitter = float(jitter.value)
        elif isinstance(jitter, (int, float)):
            self.jitter = float(jitter)
        else:
            self.jitter = None
            if jitter is not None:
                self.priors['jitter'] = jitter
                if 'jitter' not in sampled_params:
                    sampled_params += ('jitter',)

        if self.jitter is None and 'jitter' not in self.priors:
            if 'jitter' not in sampled_params:
                sampled_params += ('jitter',)
            self.priors['jitter'] = LogUniformPrior(0.001, 0.5) #mas 
        
        # TODO - Raise a warning if user is sampling more params than necessary
        self.param_order = tuple(sampled_params)
        self.sampled_params = frozenset(sampled_params)
        self.fixed_prior_params = {p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)}
        self.non_fixed_prior_params = {p for p in self.priors.keys() if not isinstance(self.priors[p], FixedPrior)}

        self.prior_covered_params = covered_parameters(self.priors.keys())
        if any(param not in self.prior_covered_params for param in sampled_params):
            missing = [param for param in sampled_params if param not in self.prior_covered_params]
            raise ValueError(f"Missing priors for sampled parameters: {missing}")


    def fit(self, data: GaiaData, rng: np.random.RandomState = np.random.default_rng()) -> FitResults:
        """Fit the Gaia data using MCMC"""
        if not isinstance(data, GaiaData):
            raise ValueError("MCMCGaiaFitter currently supports GaiaData.")
        param_order = self.param_order
        param_transforms = build_transform_functions(param_order, ('P', 'e', 'Tp',))
        _matrix_method_params = ("dalpha", "ddelta", "parallax", "mu_alpha", "mu_delta", f"B{data.system}", f"G{data.system}", f"A{data.system}", f"F{data.system}")
        param_to_prior_transforms = build_transform_functions([*param_order, *self.fixed_prior_params, *_matrix_method_params], self.non_fixed_prior_params)
        ndim = len(param_order)
        mu_single = gaia_single_motion(data.spsi,data.cpsi,data.t,data.plx_fac,data.x,data.err)
        system = getattr(data, 'system', None)



        def matrix_method(params_dict,data):
            P,e,Tp = params_dict['P'],params_dict['e'],params_dict['Tp']

            ti = data.t - Tp

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

            if 'jitter' in params_dict:
                jitter = params_dict['jitter']
                err = np.sqrt(data.err**2 + jitter**2)
            else:
                err = np.sqrt(data.err**2 + self.jitter**2) 

            w = 1.0 / err
            x_w = data.x * w
            A_w = A * w[:, None]

            ATA = A_w.T @ A_w
            ATx = A_w.T @ x_w

            mu = np.linalg.solve(ATA, ATx) # [delta alpha,delta delta, parallax,mu_alpha,mu_delta,dx,dpmra,dy,dpmdec,B,G,A,F]

            model_werr = A_w @ mu

            cov_mu = np.linalg.inv(ATA)
            mu_err = np.sqrt(np.diag(cov_mu))

            residuals = x_w - model_werr
            if "jitter" in params_dict:
                chi2 = np.sum(residuals**2 + np.log(2 * np.pi * err**2))  # Include the log term for jitter (same as joker's)
            else:
                chi2 = np.sum(residuals**2)

            return mu, mu_err, chi2
   
        def objective(params,data):
            params_dict = param_transforms(**dict(zip(param_order,params)))
            if 'jitter' in param_order:
                params_dict['jitter'] = params[param_order.index('jitter')]

            mu, _, chi2 = matrix_method(params_dict,data)


            prior_params_dict = param_to_prior_transforms(**params_dict, **{**{_matrix_method_params[i]: mu[i] for i in range(len(_matrix_method_params))}, **{p: self.priors[p].value for p in self.fixed_prior_params}})
            lp = 0
            for name, val in prior_params_dict.items():
                prior = self.priors.get(name)
                if prior is not None:
                    if isinstance(prior, FixedPrior):
                        continue  # Skip fixed priors
                    lp += prior.logpdf(val)
                    if not np.isfinite(lp):
                        return -np.inf
                else:
                    print(f"Warning:Missing prior for {name}.")

            if not np.isfinite(chi2):
                return -np.inf
            
            ln_like = -0.5 * chi2

            return ln_like + lp
        
        norm_params = ('P', 'e', 'Tp')
        if 'jitter' in param_order:
            norm_params += ('jitter',)
        initial = GaiaInitialGuess(data, rng=rng, **self.priors).get_initial_guess(norm_params, self.nwalkers)

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, objective, args=(data,), pool=self.pool)
        sampler.run_mcmc(initial, self.niter, progress=True)

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

       
        posterior = []
        valid_logp = []
        for sample, l_prob in zip(samples, lnprobs):
            params_dict = param_transforms(**dict(zip(param_order,sample)))
            if 'jitter' in param_order:
                params_dict['jitter'] = sample[param_order.index('jitter')]
            mu, _, _ = matrix_method(params_dict, data)

            delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, B, G, A, F = mu
            posterior.append([*sample, delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, A, B, F, G])
            valid_logp.append(l_prob)


        post_labels = [*param_order, 'dalpha', 'ddelta', 'parallax', 'mu_alpha', 'mu_delta', f'A{system}', f'B{system}', f'F{system}', f'G{system}']

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
        if self.jitter is not None:
            results_dict['jitter'] = self.jitter
        results_dict['priors'] = self.priors
        results_dict['raw_sampler'] = None
        results_dict['backend'] = 'emcee'
        results_dict['fit_method'] = 'linear'
        fit_results = FitResults(**results_dict)
       
        return fit_results