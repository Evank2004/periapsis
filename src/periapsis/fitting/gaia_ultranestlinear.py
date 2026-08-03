from periapsis.prior.bounds import Bounds

from .fitter import Fitter
from periapsis.data.gaia import GaiaData
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.utils.solvers import gaia_single_motion
from periapsis.prior.fixed_prior import FixedPrior
from periapsis.prior.log_uniform_prior import LogUniformPrior
from periapsis.model import Orbit
from periapsis.params.transforms import covered_parameters, build_transform_functions, overconstrained_parameters, wrapped_parameters

import numpy as np
import ultranest

class UltranestGaiaFitter(Fitter):
    def __init__(self, output_params=('P', 'e', 'Tp'), jitter=None, min_num_live_points=400, min_ess=400, dlogz=0.5, dKL=0.5, frac_remain=0.01, Lepsilon=0.001, max_iters=None, max_ncalls=None, **priors):
        super().__init__(**priors)
        self.min_num_live_points = min_num_live_points
        self.min_ess = min_ess
        self.dlogz = dlogz
        self.dKL = dKL
        self.frac_remain = frac_remain
        self.Lepsilon = Lepsilon
        self.max_iters = max_iters
        self.max_ncalls = max_ncalls

        # FIXME - don't always sample jitter.
        if isinstance(jitter, FixedPrior):
            self.jitter = float(jitter.value)
        elif isinstance(jitter, (int, float)):
            self.jitter = float(jitter)
        else:
            self.jitter = None
            if jitter is not None:
                self.priors['jitter'] = jitter
                if 'jitter' not in output_params:
                    output_params += ('jitter',)
                
        if self.jitter is None and 'jitter' not in self.priors:
            if 'jitter' not in output_params:
                output_params += ('jitter',)
            self.priors['jitter'] = LogUniformPrior(0.001, 0.5) #mas 

        self.output_params = frozenset(output_params)
        self.output_param_order = tuple(output_params)
        self.prior_params = set(self.priors.keys())
        self.fixed_prior_params = {p for p in self.prior_params if isinstance(self.priors[p], FixedPrior)}
        self.non_fixed_prior_params = {p for p in self.prior_params if not isinstance(self.priors[p], FixedPrior)}
        self.non_bound_prior_params = {p for p in self.prior_params if not isinstance(self.priors[p], Bounds)}
        self.bound_params = {p for p in self.prior_params if isinstance(self.priors[p], Bounds)}
        self.non_bound_fixed_prior_params = {p for p in self.prior_params if not isinstance(self.priors[p], (Bounds, FixedPrior))}
        self.sample_order = tuple(self.non_bound_fixed_prior_params)
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
        

    def fit(self, data: GaiaData, quiet=False) -> FitResults:
        if not isinstance(data, GaiaData):
            raise ValueError("UltranestGaiaFitter currently supports only GaiaData.")

        param_order = self.sample_order
        full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        matrix_output_params = ["dalpha", "ddelta", "parallax", "mu_alpha", "mu_delta", f"B{data.system}", f"G{data.system}", f"A{data.system}", f"F{data.system}"]
        standard_param_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params], ('P', 'e', 'Tp',))
        likelihood_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params, *matrix_output_params], self.bound_params)
        
        def prior_transform(cube):
            cube = np.array(cube, copy=True)
            sampled_priors = {}
            i = 0
            for name in param_order:
                sampled_priors[name] = self.priors[name].unp(cube[i])
                i += 1
            params = self.prior_to_sampled_transform(**sampled_priors, **{name: self.priors[name].value for name in self.fixed_prior_params})
            return np.array([*[sampled_priors[name] for name in param_order], *[params[name] for name in self.output_param_order if name not in param_order]])
        
        ref_epoch = getattr(data, 'ref_epoch', 0.0)
        reject_logl = -1e300

        def matrix_method(params_dict, data):
            P,e,Tp = params_dict['P'],params_dict['e'],params_dict['Tp']

            nobs = len(data.t)

            ti = data.t - Tp

            M = 2*np.pi * ti/P
            E = solve_kepler(M,e)

            
            X = np.cos(E) - e
            Y = np.sqrt(1.0 - e**2) * np.sin(E)

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
            elif self.jitter is not None:
                err = np.sqrt(data.err**2 + self.jitter**2)
            else:
                err = data.err

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
        

        def objective(params_dict, data):
            # if 'jitter' in param_order:
            #     params_dict['jitter'] = params[param_order.index('jitter')]
            mu, mu_err, chi2 = matrix_method(params_dict,data)
            
            ln_like = -0.5 * chi2

            return mu, mu_err, ln_like 

        def log_likelihood(params):
            standard_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
            params_dict = dict(zip(standard_param_order, params))
            params_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            standard_params_dict = standard_param_transform(**params_dict)
            mu, mu_err, chi2 = objective(standard_params_dict, data)
            
            full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order], *matrix_output_params]
            param_dict = dict(zip(full_param_order, params))
            param_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            param_dict.update({name: mu[i] for i, name in enumerate(matrix_output_params)})
            bound_param_dict = likelihood_transform(**param_dict)
            for name in self.bound_params:
                if not (self.priors[name].lower <= bound_param_dict[name] <= self.priors[name].upper):
                    return reject_logl # TODO possibly slope inwards towards the bounds instead of hard cutoff

            model = Orbit(**param_dict)
            chi2 = data.chi2(model)
            return -0.5 * chi2   

        
        single_motion = gaia_single_motion(data.spsi, data.cpsi, data.t, data.plx_fac, data.x, data.err)
        print("Running UltraNest with min_num_live_points =", self.min_num_live_points, "and min_ess =", self.min_ess)
        sampler = ultranest.ReactiveNestedSampler(
            param_names=param_order,
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

        ultranest_samples = np.array(results['samples'])

        standard_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        params_dict = dict(zip(standard_param_order, ultranest_samples.T))
        params_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
        standard_params_dict = standard_param_transform(**params_dict)

        posterior = []
        valid_logl = []
        for i, sample in enumerate(ultranest_samples):
            params_dict = {name: value[i] for name, value in standard_params_dict.items()}
            # if 'jitter' in param_order:
            #     params_dict['jitter'] = sample[param_order.index('jitter')]
            mu, _, _ = matrix_method(params_dict, data)
            ll = log_likelihood(sample)
            if not np.isfinite(ll) or ll <= reject_logl:
                continue

            
            delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, B, G, A, F = mu
            posterior.append([*sample, delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, A, B, F, G])
                
            valid_logl.append(ll)

        
        if len(valid_logl) == 0:
            raise RuntimeError("No valid posterior samples found. Check your priors and data for consistency.")
        logl = np.array(valid_logl)
        posterior = np.array(posterior)
        post_labels = [*param_order, 'dalpha', 'ddelta', 'parallax', 'mu_alpha', 'mu_delta', f'A{data.system}', f'B{data.system}', f'F{data.system}', f'G{data.system}']

        best_i = np.argmax(logl)
        best_params = dict(zip(post_labels, posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(posterior, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = self.priors[prior].value
            median_params[prior] = self.priors[prior].value

        columns = {label: [] for label in post_labels}
        for sample in posterior:
            for label, value in zip(post_labels, sample):
                columns[label].append(value)

        results_dict: dict[str, object] = {label: np.array(columns[label]) for label in post_labels}
        results_dict['ESS'] = results['ess']

        results_dict['logZ'] = results['logz']
        results_dict['logZerr'] = results['logzerr']
        results_dict['param_names'] = post_labels
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['Single_motion_params'] = single_motion
        results_dict['logl'] = logl
        results_dict['samples'] = posterior
        results_dict['n_samples_raw'] = int(len(ultranest_samples))
        results_dict['n_samples_valid'] = int(len(posterior))
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        if self.jitter is not None:
            results_dict['jitter'] = self.jitter
        results_dict['priors'] = self.priors
        results_dict['raw_sampler'] = sampler
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'linear'
        fit_results = FitResults(**results_dict)
        return fit_results