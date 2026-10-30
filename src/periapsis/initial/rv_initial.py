import numpy as np
from periapsis.data.common import RadialVelocityData
from periapsis.model.orbit import Orbit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from periapsis.params.transforms import covered_parameters, build_transform_functions
from periapsis.prior import Prior, Bounds
from periapsis.prior.fixed_prior import FixedPrior
from periapsis.utils.helpers import _matrix_builder, _matrix_filler, _lsq_helper, sigma as sig

from .initial import InitialGuess


class RVInitialGuess(InitialGuess):
    "Class for obtaining intial guess for RV data"
    def __init__(self, data: RadialVelocityData, rng: np.random.RandomState, ref_epoch=0.0, **priors):
        super().__init__(data, rng, ref_epoch, **priors)

    def _double_center(self,matrix):
        """Double center a matrix."""
        row_mean = np.mean(matrix, axis=1, keepdims=True)
        col_mean = np.mean(matrix, axis=0, keepdims=True)
        total_mean = np.mean(matrix)
        return matrix - row_mean - col_mean + total_mean

    def Zucker_pdc(self,num_freq=10000):
        """Compute the Zucker PDC to obtain an initial guess on Period"""
        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.001
        p_max = prior_p.max if prior_p is not None else 100
        freq_min = 1/p_max
        freq_max = 1/p_min
        frequencies = np.logspace(np.log10(freq_min),np.log10(freq_max),num_freq)
        periods = 1/frequencies
        pdc_vals = np.zeros(num_freq)

        x = self.data.rv
        t = self.data.t_canonical
        #sample distance matrix
        a = np.abs(x[:,np.newaxis] - x)
        A = self._double_center(a)
        sum_A = np.sum(A**2)
        delta_t = np.abs(t[:,np.newaxis] - t)

        for i,p in enumerate(periods):
            #phase difference matrix
            phi = np.mod(delta_t, p)
            #phase distance matrix
            b = phi * (p - phi)
            B = self._double_center(b)
            sum_B = np.sum(B**2)
            PDC = np.sum(A*B) / np.sqrt(sum_A * sum_B)

            pdc_vals[i] = PDC

        P_guess = periods[np.argmax(pdc_vals)]

        return P_guess

    def Bounds(self, param_names):
        """Returns bounds on the fitted parameters based on the priors"""
        bounds = []
        for name in param_names:
            prior = self.priors.get(name)
            if prior is not None:
                bounds.append((prior.min, prior.max))
            else:
                raise ValueError(f"Missing prior bounds for parameter: {name}")
                
        return bounds

    def neg_lnlike(self, params, data):
        param_in = [name for name, prior in self.priors.items() if not isinstance(prior, (Bounds, FixedPrior))]
        params_dict = dict(zip(param_in, params))
        params_dict.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        model = Orbit(**params_dict)
        offset_names = {
            name for name in params_dict
            if name.startswith("rv_") and name.endswith("_offset")
        }
        ln_like = data.log_likelihood(model, offset_names=offset_names)

        lp = 0
        for name, val in params_dict.items():
            prior = self.priors.get(name)
            if prior is not None:
                if isinstance(prior, FixedPrior):
                    continue  # Skip fixed priors
                lp += prior.logpdf(val)
                if np.isinf(lp):
                    return -np.inf
            else:
                print(f"Warning:Missing prior for {name}.")

        neg_ln_like = -(ln_like + lp)

        return neg_ln_like


    def get_initial_guess(self, param_order, nwalkers):
        """
        Returns an intial guess on fitted parameters based on the data
        """
        param_in = []
        P_guess = self.Zucker_pdc()
        initial_points = []
        for i in self.priors:
            if isinstance(self.priors[i], (Bounds, FixedPrior)):
                continue
            if i == 'P':
                initial_points.append(P_guess)
            else:
                prior = self.priors.get(i)
                if prior is not None:
                    initial_points.append(prior.sample(random_state=self.rng)[0])
                else:
                    print(f"Warning:Missing prior for {i}.")
            param_in.append(i)

        bounds = self.Bounds(param_in)
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        initial_points = np.clip(np.asarray(initial_points, dtype=float), lower, upper)
        
        result = differential_evolution(
            self.neg_lnlike, 
            bounds=bounds, 
            args=(self.data,),
            maxiter=2000,
            polish=False,
            x0=initial_points
        )

        orbit = minimize(
            self.neg_lnlike,
            x0=result.x,
            method='L-BFGS-B',
            args=(self.data,),
            bounds=bounds,
            options={'maxiter': 2000}
        )

        prior_guess = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        prior_guess.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        transform = build_transform_functions([*param_in, *self.fixed_prior_params], param_order)
        guess = transform(**prior_guess)

        poss = []
        for name in param_order:
            poss.append(guess[name] + self.rng.normal(0, 1e-4, size=nwalkers))
        pos = np.column_stack(poss)
        return pos


class RVLinearInitialGuess(RVInitialGuess):
    def __init__(self, data, rng, ref_epoch=0.0,linear_params=(), **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(), ("P", "e", "Tp")
        )
        self.M, self.cols = _matrix_builder(data, self.ref_epoch,linear_params=linear_params)
        self.eta = data.rv_canonical
        self.err = data.rv_err_canonical

    def ln_prior(self, params_dict, priors):
        lp = 0
        for name, value in params_dict.items():
            prior = priors.get(name)
            if prior is not None:
                if isinstance(prior, FixedPrior):
                    continue
                lp += prior.logpdf(value)
                if np.isinf(lp):
                    return -np.inf
            else:
                raise ValueError(f"Missing prior for parameter: {name}")
        return lp

    def neg_lnlike(self, params, data, priors, param_in):
        params_dict = dict(zip(param_in, params))
        params_dict.update({
            name: prior.value
            for name, prior in priors.items()
            if isinstance(prior, FixedPrior)
        })
        lp = self.ln_prior(params_dict, priors)
        if np.isinf(lp):
            return -np.inf

        PeTp_params = self.PeTp_transform(**params_dict)
        _, chi2 = matrix_method(self, PeTp_params, data)
        return 0.5 * chi2 - lp

    def get_initial_guess(self, param_order, nwalkers):
        param_in = []
        p_guess = self.Zucker_pdc()
        initial_points = []
        for name, prior in self.priors.items():
            if isinstance(prior, (Bounds, FixedPrior)):
                continue
            param_in.append(name)
            if name == "P":
                initial_points.append(p_guess)
            else:
                initial_points.append(prior.sample(self.rng, size=1)[0])

        bounds = self.Bounds(param_in)
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        initial_points = np.clip(np.asarray(initial_points), lower, upper)

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
            NonlinearConstraint(bounds_transform_fn(name), bound.lower, bound.upper)
            for name, bound in self.priors.items()
            if isinstance(bound, Bounds)
        ]

        result = differential_evolution(
            self.neg_lnlike,
            bounds=bounds,
            args=(self.data, self.priors, param_in),
            maxiter=2000,
            polish=False,
            x0=initial_points
        )
        orbit = minimize(
            self.neg_lnlike,
            x0=result.x,
            method="SLSQP",
            args=(self.data, self.priors, param_in),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )

        best_prior_values = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        best_prior_values.update({
            name: prior.value
            for name, prior in self.fixed_prior_params.items()
        })
        transform = build_transform_functions(
            [*param_in, *self.fixed_prior_params], param_order
        )
        best_values = transform(**best_prior_values)
        positions = [
            best_values[name] + self.rng.normal(0, 1e-4, size=nwalkers)
            for name in param_order
        ]
        return np.column_stack(positions)


def matrix_method(self, params_dict, data):
    _matrix_filler(self.M, self.cols, params_dict, data)
    sigma = sig(data, params_dict, self.err)
    _, chi2 = _lsq_helper(self.M, self.eta, sigma)
    return _, chi2
