from periapsis.params.transforms import build_transform_functions, overconstrained_parameters, covered_parameters, wrapped_parameters
from periapsis.model import Orbit
from .fitter import Fitter
from periapsis.data import Data, AstrometryData, RadialVelocityData, JointData,GaiaData
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import solve_kepler
from periapsis.utils.helpers import _matrix_builder,_matrix_filler
from periapsis.prior import FixedPrior, Bounds
import periapsis.params as par
import numpy as np
import ultranest


class UltranestLinearFitter(Fitter):
    def __init__(self, output_params=(par.P, par.e, par.Tp,),ref_epoch=0,min_num_live_points=400, min_ess=400, dlogz=0.5, dKL=0.5, frac_remain=0.01, Lepsilon=0.001, max_iters=None, max_ncalls=None, **priors):
        super().__init__(ref_epoch, **priors)
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
        if not isinstance(data, (AstrometryData, RadialVelocityData, JointData, GaiaData)):
            raise ValueError("UltranestLinearFitter supports AstrometryData, RadialVelocityData, JointData, and GaiaData.")

        self.M, self.cols = _matrix_builder(data, self.ref_epoch)
        param_order = self.sample_order
        full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
        matrix_output_params = [name for name in self.cols.keys()]
        standard_param_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params], ('P', 'e', 'Tp',))
        likelihood_transform = build_transform_functions([*full_param_order, *self.fixed_prior_params, *matrix_output_params], self.bound_params)

    
        # Matrix method cached variables
        if isinstance(data, AstrometryData):
            mm_eta = np.concatenate((data.x, data.y))
            mm_sigma = np.concatenate((data.x_err, data.y_err))
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            
        
        
        elif isinstance(data, RadialVelocityData):
            mm_eta = data.rv
            mm_sigma = data.rv_err
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w
            

        elif isinstance(data,JointData):
            mm_eta = data._concat_obs()
            mm_sigma = data._err()
            mm_w = 1/mm_sigma
            mm_eta_w = mm_eta * mm_w

        def prior_transform(cube):
            cube = np.array(cube, copy=True)
            sampled_priors = {}
            i = 0
            for name in param_order:
                sampled_priors[name] = self.priors[name].unp(cube[i])
                i += 1
            params = self.prior_to_sampled_transform(**sampled_priors, **{name: self.priors[name].value for name in self.fixed_prior_params})
            return np.array([*[sampled_priors[name] for name in param_order], *[params[name] for name in self.output_param_order if name not in param_order]])
       
        reject_logl = -1e300
        
        def matrix_method(params_dict):

            _matrix_filler(self.M,self.cols,params_dict,data)
            
            M_w = self.M* mm_w[:, None] # multiply each row of M by corresponding weight

            MTM = M_w.T @ M_w
            MT_eta = M_w.T @ mm_eta_w # matching equation
            # now we can solve for mu using np.linalg.solve
            try:
                mu = np.linalg.solve(MTM, MT_eta) 
            except np.linalg.LinAlgError:
                mu,_,_,_ = np.linalg.lstsq(MTM, MT_eta,rcond=None)
            

            model_werr = M_w @ mu # this is the model prediction with the error already over
            # this is (obs - model)/err
            resids = mm_eta_w - model_werr
            chi2 = np.sum(resids**2)
                
            return mu, chi2
        
        
        def log_likelihood(params):
            standard_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order]]
            params_dict = dict(zip(standard_param_order, params))
            params_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            standard_params_dict = standard_param_transform(**params_dict)
            mu, chi2 = matrix_method(standard_params_dict)

            full_param_order = [*param_order, *[name for name in self.output_param_order if name not in param_order], *matrix_output_params]
            param_dict = dict(zip(full_param_order, params))
            param_dict.update({name: self.priors[name].value for name in self.fixed_prior_params})
            param_dict.update({name: mu[i] for i, name in enumerate(matrix_output_params)})
            bound_param_dict = likelihood_transform(**param_dict)
            for name in self.bound_params:
                if not (self.priors[name].lower <= bound_param_dict[name] <= self.priors[name].upper):
                    return reject_logl # TODO possibly slope inwards towards the bounds instead of hard cutoff

            
            return -0.5 * chi2   

        null_hypothesis = self._null_hypothesis_fit(data)

        sampler = ultranest.ReactiveNestedSampler(
            param_names=tuple(param_order),
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

        full_posterior = []
        valid_logl = []
        for i, param in enumerate(ultranest_samples):
            P,e,Tp = standard_params_dict[par.P][i], standard_params_dict[par.e][i], standard_params_dict[par.Tp][i]
            ll = log_likelihood(param)
            if (not np.isfinite(ll)) or (ll <= reject_logl / 2):
                continue

            params_dict = {'P': P, 'e': e, 'Tp': Tp}
            try:
                mu, _ = matrix_method(params_dict)
            except np.linalg.LinAlgError:
                continue
            full_posterior.append((*param, *(mu[self.cols[name]] for name in self.cols)))
            valid_logl.append(ll)

        if len(full_posterior) == 0:
            raise RuntimeError(
                "UltraNest produced no valid posterior samples after nuisance-parameter cutoffs. "
                "Try widening nuisance priors."
            )

        logl = np.array(valid_logl)
        full_posterior_arr = np.array(full_posterior)

        post_labels = [*param_order, *self.cols.keys()]

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
        results_dict['null_hypothesis'] = null_hypothesis
        results_dict['logl'] = logl
        results_dict['samples'] = full_posterior_arr
        results_dict['n_samples_raw'] = int(len(ultranest_samples))
        results_dict['n_samples_valid'] = int(len(full_posterior_arr))
        results_dict['ref_epoch'] = self.ref_epoch

        results_dict['raw_sampler'] = sampler
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'linear'
        results_dict['priors'] = self.priors

        fit_results = FitResults(**results_dict)
        return fit_results


        