from .fitter import Fitter
from periapsis.data.gaia import GaiaData
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.utils.solvers import gaia_single_motion

import numpy as np
import ultranest
from typing import Any, cast
import logging

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

class UltranestGaia(Fitter):
    def __init__(self,nlive,min_ess, **priors):
        super().__init__(**priors)
        self.nlive = nlive
        self.min_ess = min_ess
        

    def fit(self, data: GaiaData) -> FitResults:
        param_order = [name for name in ('P', 'e', 't0') if name in self.prior_kwargs]
        if len(param_order) != 3:
            missing = [name for name in ('P', 'e', 't0') if name not in self.prior_kwargs]
            raise ValueError(f"UltranestGaia requires priors for P, e, and t0. Missing: {missing}")
        
        def matrix_method(params_dict, data):
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
        

        def objective(params):
            params_dict = dict(zip(param_order, params))
            _, _, chi2 = matrix_method(params_dict,data)


            if not np.isfinite(chi2):
                return -np.inf
            
            ln_like = -0.5 * chi2

            return ln_like 
        
        def prior_transform(cube):
            # UltraNest requires tranforming a unit hypercube
            cube = np.array(cube)
            for i, name in enumerate(param_order):
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    cube[i] = prior.unp(cube[i])
                else:
                    raise ValueError(f"Missing prior transformer for parameter: {name}")
            return cube
        
        single_motion = gaia_single_motion(data.spsi, data.cpsi, data.plx_fac, data.t, data.x, data.err)

        sampler = ultranest.ReactiveNestedSampler(param_order, objective, 
                prior_transform)
        
        results = cast(dict[str, Any], sampler.run(
            min_num_live_points=self.nlive,
            min_ess=self.min_ess,
            frac_remain=0.5,
            show_status=False,
            viz_callback=cast(Any, False),
        ))

        ultranest_samples = np.array(results['samples'])

        posterior = []
        valid_logl = []
        for sample in ultranest_samples:
            params_dict = dict(zip(param_order, sample))
            mu, _, _ = matrix_method(params_dict, data)
            ll = objective(sample)
            if not np.isfinite(ll):
                continue

            
            if mu is None:
                continue
            
            P, e, t0 = sample
            delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, B, G, A, F = mu
            
            posterior.append([P, e, t0, delta_alpha, delta_delta, parallax, mu_alpha, mu_delta, A, B, F, G])
            valid_logl.append(ll)


        logl = np.array(valid_logl)
        posterior = np.array(posterior)

        post_labels = ['P','e','t0','dalpha','ddelta','parallax','mu_alpha','mu_delta','A','B','F','G']

        best_i = np.argmax(logl)
        best_params = dict(zip(post_labels, posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(posterior, axis=0)))

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
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        
        results_dict['raw_sampler'] = sampler
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'linear'
        results_dict['priors'] = self.prior_kwargs
        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results