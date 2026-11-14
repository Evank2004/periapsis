import numpy as np
from periapsis.data.common import RadialVelocityData
from periapsis.model.orbit import Orbit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from periapsis.params.transforms import covered_parameters, build_transform_functions
from periapsis.prior import Prior, Bounds
from periapsis.prior.fixed_prior import FixedPrior
from periapsis.utils import helpers

from .initial import InitialGuess


class RVInitialGuess(InitialGuess):
    "Class for obtaining intial guess for RV data"
    def __init__(self, data: RadialVelocityData, rng: np.random.RandomState, ref_epoch=0.0, npeaks=3, **priors):
        super().__init__(data, rng, ref_epoch, npeaks=npeaks, **priors)
        self.M_base, self.cols = helpers._matrix_builder(
            data,
            ref_epoch,
            linear_params=self.priors.keys(),
        )
        self.eta = data.rv_canonical
        self.err = data.rv_err_canonical
        


    def _double_center(self,matrix):
        """Double center a matrix."""
        row_mean = np.mean(matrix, axis=1, keepdims=True)
        col_mean = np.mean(matrix, axis=0, keepdims=True)
        total_mean = np.mean(matrix)
        return matrix - row_mean - col_mean + total_mean

    def Zucker_pdc(self,npeaks=None,num_freq=10000):
        """Compute the Zucker PDC to obtain an initial guess on Period"""
        if npeaks is None:
            npeaks = self.npeaks
        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.001
        p_max = prior_p.max if prior_p is not None else 100
        freq_min = 1/p_max
        freq_max = 1/p_min
        frequencies = np.logspace(np.log10(freq_min),np.log10(freq_max),num_freq)
        periods = 1/frequencies
        pdc_vals = np.zeros(num_freq)

        x = self.data.rv_canonical
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

        top_cands = np.argsort(pdc_vals)[::-1][:npeaks]
        top_periods = periods[top_cands]
        self.period_guesses = top_periods
        

        return top_periods

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


    def _linear_seed(self):
        linear_params = set(self.cols)
        fixed_params = set(self.fixed_prior_params)

        linear_derived_params = covered_parameters(
            linear_params | fixed_params
        )

        param_in = [
            name
            for name, prior in self.priors.items()
            if not isinstance(prior, (Bounds, FixedPrior))
            and name not in linear_derived_params
        ]

        P_guesses = self.Zucker_pdc()

        self.PeTp_transform = build_transform_functions(
            self.priors.keys(),
            ("P", "e", "Tp"),
        )

        initial_points = []
        for name in param_in:
            if name == "P":
                initial_points.append(P_guesses[0])
            else:
                initial_points.append(
                    self.priors[name].sample(self.rng, size=1)[0]
                )


        def linear_neg_lnlike(values, param_in):
            params_dict = dict(zip(param_in, values))
            params_dict.update({
                name: prior.value
                for name, prior in self.fixed_prior_params.items()
            })
            params_dict.update(self.PeTp_transform(**params_dict))

            solved_params, chi2 = helpers._linear_solution(
                self.M_base,
                self.cols,
                params_dict,
                self.data,
                self.eta,
                self.err,
            )
            lp = 0.0
            for name, value in solved_params.items():
                prior = self.priors.get(name)
                if prior is None or isinstance(prior, FixedPrior):
                    continue

                
                contribution = prior.logpdf(value)
                if not np.isfinite(contribution):
                    return np.inf

                lp += contribution

            if not np.isfinite(chi2):
                return np.inf

            return 0.5 * chi2 - lp


        bounds = self.Bounds(param_in)
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)

        initial_points = np.clip(
            np.asarray(initial_points, dtype=float),
            lower,
            upper,
        )


        def bounds_transform_fn(name, values):
            params_dict = dict(zip(param_in, values))
            params_dict.update({
                parameter: prior.value
                for parameter, prior in self.fixed_prior_params.items()
            })
            params_dict.update(self.PeTp_transform(**params_dict))

            solved_params, _ = helpers._linear_solution(
                self.M_base,
                self.cols,
                params_dict,
                self.data,
                self.eta,
                self.err,
            )
            return solved_params[name]

        constraints = [
            NonlinearConstraint(
                lambda values, name=name: bounds_transform_fn(name, values),
                bound.lower,
                bound.upper,
            )
            for name, bound in self.priors.items()
            if isinstance(bound, Bounds)
        ]


        results = []

        for period in P_guesses:
            candidate = initial_points.copy()
            candidate[param_in.index("P")] = period
            candidate = np.clip(candidate, lower, upper)

            result = differential_evolution(
                linear_neg_lnlike,
                bounds=bounds,
                args=(param_in,),
                maxiter=2000,
                polish=False,
                x0=candidate,
                rng=self.rng,
            )

            if np.isfinite(result.fun):
                results.append(result)

        if not results:
            raise RuntimeError("All RV initializer period candidates failed.")

        result = min(results, key=lambda item: item.fun)

        orbit = minimize(
            linear_neg_lnlike,
            x0=result.x,
            method="SLSQP",
            args=(param_in,),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )

        if orbit.success and np.isfinite(orbit.fun):
            best_vector = orbit.x
        else:
            best_vector = result.x
        
        best_values = dict(zip(
            param_in,
            np.clip(best_vector, lower, upper),
        ))
        

        best_values.update({
            name: prior.value
            for name, prior in self.fixed_prior_params.items()
        })
        best_values.update(self.PeTp_transform(**best_values))

        best_values, _ = helpers._linear_solution(
            self.M_base,
            self.cols,
            best_values,
            self.data,
            self.eta,
            self.err,
        )

        return best_values

    def get_initial_guess(self, param_order, nwalkers):
        best_values = self._linear_seed()
        linear_params = set(self.cols)
        fixed_params = set(self.fixed_prior_params)
        linear_derived_params = covered_parameters(
            linear_params | fixed_params
        )

        nonlinear_names = [
            name
            for name, prior in self.priors.items()
            if not isinstance(prior, (Bounds, FixedPrior))
            and name not in linear_derived_params
        ]

        positions = []

        for _ in range(nwalkers):
            params = dict(best_values)

            for name in nonlinear_names:
                prior = self.priors.get(name)
                if prior is None:
                    raise ValueError(
                        f"Missing prior for parameter: {name}"
                    )

                scale = max(
                    1e-4 * abs(best_values[name]),
                    1e-8,
                )

                if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                    lower = getattr(prior, "lower", getattr(prior, "min", None))
                    upper = getattr(prior, "upper", getattr(prior, "max", None))
                    if lower is not None and upper is not None:
                        scale = max(scale, 1e-4 * (upper - lower))

                value = best_values[name] + self.rng.normal(
                    0.0,
                    scale,
                )

                params[name] = np.clip(
                    value,
                    prior.min,
                    prior.max,
                )

            params.update(self.PeTp_transform(**params))

            params, _ = helpers._linear_solution(
                self.M_base,
                self.cols,
                params,
                self.data,
                self.eta,
                self.err,
            )

            transform = build_transform_functions(
                params.keys(),
                param_order,
            )
            params.update(transform(**params))
            positions.append([
                params[name]
                for name in param_order
            ])

        return np.asarray(positions)


class RVLinearInitialGuess(RVInitialGuess):
    def __init__(self, data, rng, ref_epoch=0.0, npeaks=3, linear_params=(), **priors):
        super().__init__(data, rng, ref_epoch, npeaks, **priors)
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(), ("P", "e", "Tp")
        )
        self.M, self.cols = helpers._matrix_builder(data, self.ref_epoch,linear_params=linear_params)
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
            return np.inf

        PeTp_params = self.PeTp_transform(**params_dict)
        _, chi2 = matrix_method(self, PeTp_params, data)
        if not np.isfinite(chi2):
            return np.inf
        
        return 0.5 * chi2 - lp

    def get_initial_guess(self, param_order, nwalkers):
        param_in = []
        p_guesses = self.Zucker_pdc()
        initial_points = []
        for name, prior in self.priors.items():
            if isinstance(prior, (Bounds, FixedPrior)):
                continue
            param_in.append(name)
            if name == "P":
                initial_points.append(p_guesses[0])
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

        results = []
        
        for period in p_guesses:
            candidate = initial_points.copy()
            candidate[param_in.index("P")] = period
            candidate = np.clip(candidate, lower, upper)

            result = differential_evolution(
                self.neg_lnlike,
                bounds=bounds,
                args=(self.data, self.priors, param_in),
                maxiter=2000,
                polish=False,
                x0=candidate,
                rng=self.rng,
            )

            if np.isfinite(result.fun):
                results.append(result)

        if not results:
            raise RuntimeError("All RV initializer period candidates failed.")

        result = min(results, key=lambda item: item.fun)

        orbit = minimize(
            self.neg_lnlike,
            x0=result.x,
            method="SLSQP",
            args=(self.data, self.priors, param_in),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )

        if orbit.success and np.isfinite(orbit.fun):
            best_vector = orbit.x
        else:
            best_vector = result.x

        
        best_prior_values = dict(zip(
            param_in,
            np.clip(best_vector, lower, upper),
        ))
        best_prior_values.update({
            name: prior.value
            for name, prior in self.fixed_prior_params.items()
        })
        transform = build_transform_functions(
            [*param_in, *self.fixed_prior_params], param_order
        )
        best_values = transform(**best_prior_values)
        positions = []
        for name in param_order:
            if name not in best_values:
                raise ValueError(f"Missing initial guess for parameter: {name}")

            prior = self.priors.get(name)

            scale = max(
                1e-4 * abs(best_values[name]),
                1e-8,
            )

            if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                lower = getattr(prior, "lower", getattr(prior, "min", None))
                upper = getattr(prior, "upper", getattr(prior, "max", None))
                if lower is not None and upper is not None:
                    scale = max(scale, 1e-4 * (upper - lower))

            values = best_values[name] + self.rng.normal(
                0.0,
                scale,
                size=nwalkers,
            )

            if prior is not None and not isinstance(prior, (Bounds, FixedPrior)):
                values = np.clip(values, prior.min, prior.max)

            positions.append(values)
        return np.column_stack(positions)


def matrix_method(self, params_dict, data):
    helpers._matrix_filler(self.M, self.cols, params_dict, data)
    sigma = helpers.sigma(data, params_dict, self.err)
    _, chi2 = helpers._lsq_helper(self.M, self.eta, sigma)
    return _, chi2
