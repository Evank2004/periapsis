import numpy as np
from periapsis.utils.helpers import _lsq_helper, _matrix_builder, _matrix_filler, _null_matrix_builder, _fill_periodogram_periodic, _sigma
from periapsis.data.gaia import GaiaData
from periapsis.params.transforms import build_transform_functions
from periapsis.prior import Bounds, FixedPrior
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
import periapsis.params as params
from .initial import InitialGuess
from periapsis.model.orbit import Orbit


class GaiaLinearInitialGuess(InitialGuess):
    """Class for obtaining initial guess for Gaia data for MCMCLinearFitter"""
    def __init__(self, data, rng: np.random.RandomState, ref_epoch=0.0, linear_params=(), **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.M_base, self.cols = _matrix_builder(
            data, ref_epoch, linear_params=linear_params
        )
        

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
            rng=self.rng,
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
            values = best_values[name] + self.rng.normal(
                0, 1e-4, size=nwalkers
            ) * max(abs(best_values[name]), 1.0)
            prior = self.priors.get(name)
            if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                values = np.clip(values, prior.min, prior.max)
            poss.append(values)

        return np.column_stack(poss)


class GaiaInitialGuess(GaiaLinearInitialGuess):
    """Class for obtaining initial guesses for the full nonlinear MCMC."""

    def __init__(self, data, rng: np.random.RandomState, ref_epoch=0.0, **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(), ("P", "e", "Tp")
        )

    def ln_like(self, params_dict, data):
        model = Orbit(**params_dict)
        offset_names = {
            name
            for name in params_dict
            if name.startswith("astro_") and name.endswith("_offset")
        }
        return data.log_likelihood(model, offset_names=offset_names)

    def ln_prior(self, params_dict, priors):
        lp = 0.0
        for name, value in params_dict.items():
            prior = priors.get(name)
            if prior is None:
                continue
            if isinstance(prior, FixedPrior):
                continue
            contribution = prior.logpdf(value)
            if not np.isfinite(contribution):
                return -np.inf
            lp += contribution
        return lp

    

    def _linear_seed(self):
        ''' Function for starting nonlinear fitter with a linear fit for Gaia data'''
        P_guess, _ = self.Delisle_periodogram()
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(), ("P", "e", "Tp")
        )
        param_in = []
        initial_points = []

        for name, prior in self.priors.items():
            if isinstance(prior, (Bounds, FixedPrior)):
                continue
            param_in.append(name)
            if name == params.P:
                initial_points.append(P_guess)
            else:
                initial_points.append(prior.sample(self.rng, size=1)[0])

        bounds = self._bounds(param_in)
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        initial_values = np.clip(np.asarray(initial_points, dtype=float), lower, upper)
        
        def bounds_transform_fn(bound):
            transform = build_transform_functions(
                [*param_in, *self.fixed_prior_params], [bound]
            )
            return lambda values: transform(
                **dict(zip(param_in, values)),
                **{
                    name: prior.value
                    for name, prior in self.fixed_prior_params.items()
                },
            )[bound]

        constraints = [
                    NonlinearConstraint(
                        bounds_transform_fn(name), bound.lower, bound.upper
                    )
                    for name, bound in self.priors.items()
                    if isinstance(bound, Bounds)
                ]
        
        linear_neg_lnlike = super().neg_lnlike

        result = differential_evolution(
            linear_neg_lnlike,
            bounds=bounds,
            args=(self.data, self.priors, param_in),
            maxiter=2000,
            polish=False,
            x0=initial_values,
            rng=self.rng,
        )

        orbit = minimize(
            linear_neg_lnlike,
            x0=result.x,
            method="SLSQP",
            args=(self.data, self.priors, param_in),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )
        best_vals = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        best_vals.update(
            {name: prior.value for name, prior in self.fixed_prior_params.items()}
        )
        best_vals.update(self.PeTp_transform(**best_vals))

        _matrix_filler(self.M_base, self.cols, best_vals, self.data)
        sigma = _sigma(self.data, best_vals, self.data.err)
        mu,_ = _lsq_helper(self.M_base, self.data.x, sigma)

        best_vals.update({
            name:mu[i] for name,i in self.cols.items()
        })

        return best_vals
        

    def get_initial_guess(self, param_order, nwalkers):
        """Return walkers around a full nonlinear Gaia solution."""

        best_values = self._linear_seed() # Initialize full parameter Gaia fit with linear solver, as it can provide a more robust initial guess for nonlinear fit


        transform = build_transform_functions(
            best_values.keys(), param_order
        )
        best_values.update(transform(**best_values))
    
        positions = []
        for name in param_order:
            if name not in best_values:
                raise ValueError(f"Missing initial guess for parameter: {name}")

            prior = self.priors.get(name)

            scale = 1e-4 * abs(best_values[name])

            if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                scale = max(scale, 1e-4 * (prior.max - prior.min))

            values = best_values[name] + self.rng.normal(
                0.0,
                scale,
                size=nwalkers,
            )

            if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                values = np.clip(values, prior.min, prior.max)

            positions.append(values)
        return np.column_stack(positions)