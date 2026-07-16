from .fitter import Fitter
from orbit_package.data.data import Data
from orbit_package.data.common import AstrometryData
from orbit_package.fitting.results import FitResults
from orbit_package.initial.initial import InitialFit
from orbit_package.utils.solvers import solve_kepler
from orbit_package.utils.solvers import transform_theile
from orbit_package.utils.solvers import solve_mass
from scipy.optimize import dual_annealing
from orbit_package.utils.helpers import _match_param_keys
import numpy as np
import ultranest
from typing import Any, cast
import logging

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


class UltranestLinearFitter(Fitter):
    def __init__(self,nlive,min_ess, m1=None, m2_max=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nlive = nlive
        self.min_ess = min_ess
        self.m2_max = m2_max
        

    def fit(self, data: Data) -> FitResults:
        param_order = [name for name in ('P', 'e', 't0') if name in self.prior_kwargs]
        if len(param_order) != 3:
            missing = [name for name in ('P', 'e', 't0') if name not in self.prior_kwargs]
            raise ValueError(f"UltranestLinearFitter requires priors for P, e, and t0. Missing: {missing}")
       
        ref_epoch = getattr(data, 'ref_epoch', 0.0)
        reject_logl = -1e300
        
        def matrix_method(params_dict, data, E):
            nobs = len(data.t)

            dt = data.t - ref_epoch

            M = np.zeros((2 * nobs, 8))

            eta = np.concatenate((data.x, data.y))
            sigma = np.concatenate((data.x_err, data.y_err))

            X = np.cos(E) - params_dict['e']
            Y = np.sqrt(1 - params_dict['e'] ** 2) * np.sin(E)

            M[:nobs, 0] = 1  # dx
            M[:nobs, 1] = dt  # dpmra
            M[:nobs, 2] = X  # A
            M[:nobs, 3] = Y  # F

            # now bottom half y obs
            M[nobs:, 4] = 1  # dy
            M[nobs:, 5] = dt  # dpmdec
            M[nobs:, 6] = X  # B
            M[nobs:, 7] = Y  # G

            w = 1 / sigma
            eta_w = eta * w
            M_w = M * w[:, None]

            MTM = M_w.T @ M_w
            MT_eta = M_w.T @ eta_w # matching equation
            # now we can solve for mu using np.linalg.solve
            mu = np.linalg.solve(MTM, MT_eta) # dx,pmra,A,F,dy,pmdec,B,G

            model_werr = M_w @ mu # this is the model prediction with the error already over
            # this is (obs - model)/err
            resids = eta_w - model_werr
            chi2 = np.sum(resids**2)
            
            return mu, chi2
        
        def objective(data, params):
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
                return np.inf

            
            for name, value in [('dx', dx), ('dpmra', dpmra), ('dy', dy), ('dpmdec', dpmdec)]:
                prior = self.prior_kwargs.get(name)
                if prior is not None and hasattr(prior, 'min') and hasattr(prior, 'max'):
                    if not (prior.min <= value <= prior.max):
                        return np.inf

            if self.m1 is not None and self.m2_max is not None:
                a1, _, _, _ = transform_theile(A, B, F, G)
                m2 = solve_mass(a1, params_dict['P'], self.m1)
                if (not np.isfinite(m2)) or m2 > self.m2_max or m2 < 0:
                    return np.inf

            return chi2
        
            

        def log_likelihood(params):

            chi2 = objective(data, params)
            if not np.isfinite(chi2):
                return reject_logl

            return -0.5*chi2 
        
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

        pm_fit = self._proper_motion_fit(data)

        sampler = ultranest.ReactiveNestedSampler(
            param_order,
            log_likelihood,
            prior_transform,
            draw_multiple=False,
            vectorized=False,
        )

        results = cast(dict[str, Any], sampler.run(
            min_num_live_points=self.nlive,
            min_ess=self.min_ess,
            frac_remain=0.5,
            show_status=False,
            viz_callback=cast(Any, False),
        ))

        ultranest_samples = np.array(results['samples'])

        full_posterior = []
        valid_logl = []
        for param in ultranest_samples:
            P,e,t0 = param
            ll = log_likelihood(param)
            if (not np.isfinite(ll)) or (ll <= reject_logl / 2):
                continue

            M = 2*np.pi * (data.t - ref_epoch - t0*P) / P
            E = solve_kepler(M,e)
            params_dict = {'P': P, 'e': e, 't0': t0}
            try:
                mu, _ = matrix_method(params_dict,data,E)
            except np.linalg.LinAlgError:
                continue
            dx = mu[0]
            dpmra = mu[1]
            A = mu[2]
            F = mu[3]
            dy = mu[4]
            dpmdec = mu[5]
            B = mu[6]
            G = mu[7]
            full_posterior.append((P,e,t0,A,B,F,G,dx,dy,dpmra,dpmdec))
            valid_logl.append(ll)

        if len(full_posterior) == 0:
            raise RuntimeError(
                "UltraNest produced no valid posterior samples after nuisance-parameter cutoffs. "
                "Try widening nuisance priors or relaxing m2_max."
            )

        logl = np.array(valid_logl)
        full_posterior_arr = np.array(full_posterior)

        post_labels = ['P','e','t0','A','B','F','G','dx','dy','dpmra','dpmdec']

        best_i = int(np.argmax(logl))
        best_params = dict(zip(post_labels, full_posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(full_posterior_arr, axis=0)))

        columns = {label: [] for label in post_labels}
        for sample in full_posterior:
            for label, value in zip(post_labels, sample):
                columns[label].append(value)

        results_dict: dict[str, object] = {label: np.array(columns[label]) for label in post_labels}
        
        results_dict['ESS'] = results['ess']
        results_dict['logZ'] = results['logz']
        results_dict['logZerr'] = results['logzerr']
        results_dict['param_names'] = post_labels
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['PM_fit'] = pm_fit
        results_dict['logl'] = logl
        results_dict['samples'] = full_posterior_arr
        results_dict['n_samples_raw'] = int(len(ultranest_samples))
        results_dict['n_samples_valid'] = int(len(full_posterior_arr))
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)

        results_dict['raw_sampler'] = sampler
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'linear'

        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results


        