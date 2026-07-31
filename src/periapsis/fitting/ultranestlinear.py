from periapsis.data.common import AstrometryData
from periapsis.params.transforms import build_transform_functions, overconstrained_parameters, covered_parameters
from periapsis.model import Orbit
from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.prior import FixedPrior, Bounds
import numpy as np
import ultranest


class UltranestLinearFitter(Fitter):
    def __init__(self, output_params=('P', 'e', 'Tp',), min_num_live_points=400, min_ess=400, dlogz=0.5, dKL=0.5, frac_remain=0.01, Lepsilon=0.001, max_iters=None, max_ncalls=None, **priors):
        super().__init__(**priors)
        self.min_num_live_points = min_num_live_points
        self.min_ess = min_ess
        self.dlogz = dlogz
        self.dKL = dKL
        self.frac_remain = frac_remain
        self.Lepsilon = Lepsilon
        self.max_iters = max_iters
        self.max_ncalls = max_ncalls
        self.output_params = frozenset(output_params)
        self.output_param_order = tuple(output_params)
        self.prior_params = set(priors.keys())
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
        

    def fit(self, data: Data, quiet = False) -> FitResults:
        if not isinstance(data, AstrometryData):
            raise ValueError("UltranestLinearFitter currently supports AstrometryData.")

        param_order = self.sample_order
        full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        matrix_output_params = ["dx", "dpmra", f"A{data.system}", f"F{data.system}", "dy", "dpmdec", f"B{data.system}", f"G{data.system}"]
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
        
        def objective(data, params_dict):
            dt = data.t - ref_epoch
            ti = dt - (params_dict['Tp'] - ref_epoch)

            M = 2 * np.pi * ti / params_dict['P']
            E = solve_kepler(M, params_dict['e'])

            mu, chi2 = matrix_method(params_dict, data, E)

            return mu, chi2
        
        def log_likelihood(params):
            standard_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
            params_dict = dict(zip(standard_param_order, params))
            params_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            standard_params_dict = standard_param_transform(**params_dict)
            mu, chi2 = objective(data, standard_params_dict)

            full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order], *matrix_output_params]
            param_dict = dict(zip(full_param_order, params))
            param_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            param_dict.update({name: mu[i] for i, name in enumerate(matrix_output_params)})
            if "Tepoch" not in param_dict:
                param_dict["Tepoch"] = ref_epoch
            bound_param_dict = likelihood_transform(**param_dict)
            for name in self.bound_params:
                if not (self.priors[name].lower <= bound_param_dict[name] <= self.priors[name].upper):
                    return reject_logl # TODO possibly slope inwards towards the bounds instead of hard cutoff

            model = Orbit(**param_dict)
            chi2 = data.chi2(model)
            return -0.5 * chi2   

        pm_fit = self._proper_motion_fit(data)

        sampler = ultranest.ReactiveNestedSampler(
            param_names=tuple(param_order),
            loglike=log_likelihood, 
            transform=prior_transform,
            derived_param_names=tuple([name for name in self.output_param_order if name not in param_order]),
            wrapped_params=None, # TODO
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

        full_posterior = []
        valid_logl = []
        for i, param in enumerate(ultranest_samples):
            P,e,Tp = standard_params_dict['P'][i], standard_params_dict['e'][i], standard_params_dict['Tp'][i]
            ll = log_likelihood(param)
            if (not np.isfinite(ll)) or (ll <= reject_logl / 2):
                continue

            M = 2*np.pi * (data.t - Tp) / P
            E = solve_kepler(M,e)
            params_dict = {'P': P, 'e': e, 'Tp': Tp}
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
            full_posterior.append((P,e,Tp,A,B,F,G,dx,dy,dpmra,dpmdec))
            valid_logl.append(ll)

        if len(full_posterior) == 0:
            raise RuntimeError(
                "UltraNest produced no valid posterior samples after nuisance-parameter cutoffs. "
                "Try widening nuisance priors."
            )

        logl = np.array(valid_logl)
        full_posterior_arr = np.array(full_posterior)

        post_labels = ['P','e','Tp',f'A{data.system}',f'B{data.system}',f'F{data.system}',f'G{data.system}','dx','dy','dpmra','dpmdec']

        best_i = int(np.argmax(logl))
        best_params = dict(zip(post_labels, full_posterior[best_i]))
        median_params = dict(zip(post_labels, np.median(full_posterior_arr, axis=0)))
        for prior in self.fixed_prior_params:
            best_params[prior] = self.priors[prior].value
            median_params[prior] = self.priors[prior].value

        columns = {label: [] for label in post_labels}
        for sample in full_posterior:
            for label, value in zip(post_labels, sample):
                columns[label].append(value)

        results_dict: dict[str, object] = {label: np.array(columns[label]) for label in post_labels}
        
        results_dict['Ess'] = results['ess']
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
        results_dict['priors'] = self.priors

        if results_dict['ref_epoch'] is not None:
            results_dict['priors']['Tepoch'] = FixedPrior(results_dict['ref_epoch'])

        fit_results = FitResults(**results_dict)
        return fit_results


        