import numpy as np
from periapsis.utils.helpers import _lsq_helper, _matrix_builder, _matrix_filler, _null_matrix_builder, _fill_periodogram_periodic, _sigma
from periapsis.data.gaia import GaiaData
from periapsis.params.transforms import build_transform_functions
from periapsis.prior import Bounds, FixedPrior
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint

from .initial import InitialGuess


class GaiaInitialGuess(InitialGuess):
    """Class for obtaining initial guess for Gaia data"""
    def __init__(self, data, rng: np.random.RandomState, ref_epoch=0.0, **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.M_base, self.cols = _matrix_builder(data, ref_epoch)
        

    def Delisle_periodogram(self,num_freq=10000):
        """Compute the Delisle periodogram to obtain an initial guess on Period"""

        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.001
        p_max = prior_p.max if prior_p is not None else 100 

        M_null,_ = _null_matrix_builder(self.data,self.ref_epoch)


        _,chi2H = _lsq_helper(M_null,self.data.x,self.data.err)

        min_freq = 1/p_max
        max_freq = 1/p_min
        frequencies = np.logspace(np.log10(min_freq),np.log10(max_freq),num_freq)
        periods = 1/frequencies
        power = np.zeros(num_freq)
  

        for i, nu in enumerate(frequencies):
            phase = 2 * np.pi * nu 
            M = self.M_base.copy()
            _fill_periodogram_periodic(M, self.cols,self.ref_epoch,self.data,phase)

            _,chi2K = _lsq_helper(M,self.data.x,self.data.err)

            z_GLS = (chi2H - chi2K) / chi2H

            power[i] = z_GLS

        P_guess = periods[np.argmax(power)]
        max_pwr = power[np.argmax(power)]

        return P_guess, max_pwr

    def _bounds(self,prior_param_names):
        """Returns bounds on the fitted parameters based on the priors"""
        bounds = []
        for name in prior_param_names:
            prior = self.priors.get(name)
            if prior is not None:
                bounds.append((prior.min, prior.max))
            else:
                raise ValueError(f"Missing prior bounds for parameter: {name}")   
        return bounds

    def neg_lnlike(self, params, data, priors, param_in):
        params_dict = dict(zip(param_in, params))
        params_dict.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        params_dict.update(self.PeTp_transform(**params_dict))

        _matrix_filler(self.M_base, self.cols, params_dict, data)
        sigma = _sigma(data, params_dict, data.err)
        _, chi2 = _lsq_helper(self.M_base, data.x, sigma)

        ln_prior = 0.0
        for name in param_in:
            ln_prior += priors[name].logpdf(params_dict[name])
        if not np.isfinite(ln_prior) or not np.isfinite(chi2):
            return np.inf
        return 0.5 * chi2 - ln_prior

    def get_initial_guess(self, param_order, nwalkers):
        """Return walkers around a globally and locally optimized Gaia solution."""
        P_guess, _ = self.Delisle_periodogram()
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(), ('P', 'e', 'Tp')
        )
        param_in = [
            name for name, prior in self.priors.items()
            if not isinstance(prior, (Bounds, FixedPrior))
        ]
        bounds = self._bounds(param_in)
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        initial_points = []
        for name in param_in:
            if name == 'P':
                initial_points.append(P_guess)
            else:
                initial_points.append(self.priors[name].sample(self.rng, size=1)[0])
        initial_points = np.clip(np.asarray(initial_points, dtype=float), lower, upper)

        def bounds_transform_fn(bound):
            transform = build_transform_functions(self.priors.keys(), [bound])
            def evaluate(values):
                params = dict(zip(param_in, values))
                params.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
                return transform(**params)[bound]
            return evaluate
                
        constraints = []
        for name, bound in self.priors.items():
            if not isinstance(bound, Bounds):
                continue
            constraints.append(NonlinearConstraint(bounds_transform_fn(name),  bound.lower, bound.upper))
                

        result = differential_evolution(
            self.neg_lnlike,
            bounds=bounds,
            args=(self.data, self.priors, param_in),
            maxiter=2000,
            polish=False,
            x0=initial_points,
        )
        orbit = minimize(
            self.neg_lnlike,
            x0=result.x,
            method='SLSQP',
            args=(self.data, self.priors, param_in),
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000},
        )

        best_values = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        best_values.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        best_values.update(self.PeTp_transform(**best_values))
        poss = []
        for name in param_order:
            if name not in best_values:
                raise ValueError(f"Missing initial guess for parameter: {name}")
            poss.append(
                best_values[name]
                + self.rng.normal(0, 1e-4, size=nwalkers) * best_values[name]
            )

        return np.column_stack(poss)
