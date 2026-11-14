from periapsis.model.orbit import Orbit
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, LinearConstraint
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from periapsis.params.transforms import build_transform_functions,covered_parameters
from periapsis.prior import FixedPrior, Bounds
import periapsis.utils.helpers as helpers
import periapsis.params as params
from periapsis.data import AstrometryData, PhysicalAstrometryData

from .initial import InitialGuess

class AstrometryInitialGuess(InitialGuess):
    """Class for obtaining an intial guess on fitted parameters"""
    def __init__(self, data, rng, ref_epoch=0.0,npeaks=3, **priors):
        super().__init__(data, rng, ref_epoch, npeaks=npeaks, **priors)
        self.ref_epoch = ref_epoch
        self.M_base, self.cols = helpers._matrix_builder(
        data,
        ref_epoch,
        linear_params=self.priors.keys(),
        )

        if isinstance(data, PhysicalAstrometryData):
            self.eta = np.concatenate((
                data.x_canonical,
                data.y_canonical,
            ))
            self.err = np.concatenate((
                data.x_err_canonical,
                data.y_err_canonical,
            ))
        elif isinstance(data, AstrometryData):
            self.eta = np.concatenate((
                data.x_ang_canonical,
                data.y_ang_canonical,
            ))
            self.err = np.concatenate((
                data.x_err_ang_canonical,
                data.y_err_ang_canonical,
            ))
        else:
            raise ValueError(
                "Data must be an instance of AstrometryData "
                "or PhysicalAstrometryData."
            )

    
    def lomb_scargle(self,npeaks=None):
        """Returns an initial guess on the period and semi major axis 
        based on a Lomb-Scargle periodogram"""
        if npeaks is None:
            npeaks = self.npeaks
        prior_p = self.priors.get(params.P)
        if prior_p is None:
            raise ValueError(
                "Computing a periodogram requires a direct prior on the period 'P'"
            )
        p_min = prior_p.min
        p_max = prior_p.max
        f_min = 1/p_max
        f_max = 1/p_min
        
        frequency = np.logspace(np.log10(f_min),np.log10(f_max),100000)

        if isinstance(self.data, PhysicalAstrometryData):
            p1 = LombScargle(self.data.t_canonical,self.data.x_canonical,self.data.x_err_canonical)
            p2 = LombScargle(self.data.t_canonical,self.data.y_canonical,self.data.y_err_canonical)
        elif isinstance(self.data, AstrometryData):
            p1 = LombScargle(self.data.t_canonical,self.data.x_ang_canonical,self.data.x_err_ang_canonical)
            p2 = LombScargle(self.data.t_canonical,self.data.y_ang_canonical,self.data.y_err_ang_canonical)
        else:
            raise ValueError("Data must be an instance of AstrometryData or PhysicalAstrometryData.")
        

        power1 = p1.power(frequency)
        power2 = p2.power(frequency)
        
        power_total = power1 + power2
        peaks,_ = find_peaks(power_total)
        top_peaks = peaks[np.argsort(power_total[peaks])][::-1][:npeaks]
        top_frequency = frequency[top_peaks]
        

        a = np.array([
            p1.model_parameters(frequency)
            for frequency in top_frequency
        ])
        b = np.array([
            p2.model_parameters(frequency)
            for frequency in top_frequency
        ])

        ampx = np.hypot(a[:, 1], a[:, 2])
        ampy = np.hypot(b[:, 1], b[:, 2])

        a1_guesses = np.hypot(ampx,ampy)
        p_guesses = 1/top_frequency
        self.period_guesses = p_guesses
        return a1_guesses, p_guesses 
    
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
    
    def ln_like(self,params_dict,data):
        """Returns the log likelihood of the given parameters based on the data"""
        model = Orbit(**params_dict)
        offset_names = {
            name for name in params_dict
            if name.startswith("astro_") and name.endswith("_offset")
        }
        return data.log_likelihood(model, offset_names=offset_names)
        
    
    def ln_prior(self,params_dict,priors):
        """Returns the log prior of the given parameters based on the priors"""
        lp = 0
        for name,val in params_dict.items():
            prior = priors.get(name)
            if prior is not None:
                if isinstance(prior, FixedPrior):
                    continue  # Skip fixed priors
                lp += prior.logpdf(val)
                if np.isinf(lp):
                    return -np.inf
            else:
                continue
        return lp
    
    def neg_lnlike(self,params,data,priors,param_in):
        params_dict = dict(zip(param_in,params))
        params_dict.update({name: prior.value for name, prior in priors.items() if isinstance(prior, FixedPrior)})
        return -(self.ln_prior(params_dict, priors) + self.ln_like(params_dict, data))
    


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

        a1_guesses, p_guesses = self.lomb_scargle()
        initial_points = []

        for name in param_in:
            if name == params.P:
                initial_points.append(p_guesses[0])
            elif name in {params.a, params.a1, params.a2}:
                initial_points.append(a1_guesses[0])
            else:
                initial_points.append(
                    self.priors[name].sample(self.rng, size=1)[0]
                )

        bounds = self._bounds(param_in)
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        initial_points = np.clip(
            np.asarray(initial_points, dtype=float),
            lower,
            upper,
        )

        self.PeTp_transform = build_transform_functions(
            self.priors.keys(),
            ("P", "e", "Tp"),
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

            ln_prior = self.ln_prior(solved_params, self.priors)

            if not np.isfinite(ln_prior) or not np.isfinite(chi2):
                return np.inf

            return 0.5 * chi2 - ln_prior

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

        for period in p_guesses:
            candidate = initial_points.copy()
            candidate[param_in.index(params.P)] = period
            candidate = np.clip(candidate, lower, upper)
            for a in (params.a, params.a1, params.a2):
                if a in param_in:
                    candidate[param_in.index(a)] = a1_guesses[0]
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
            raise RuntimeError("All initializer period candidates failed.")

        result = min(results, key=lambda r: r.fun)
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

        pos = np.asarray(positions)
        return pos


class AstrometryLinearInitialGuess(AstrometryInitialGuess):
    def __init__(self, data, rng, ref_epoch=0.0, npeaks=3,linear_params=(), **priors):
        super().__init__(data,rng,ref_epoch,npeaks=npeaks,**priors)
        self.PeTp_transform = build_transform_functions(
            self.priors.keys(),
            ("P", "e", "Tp"),
        )
        self.M, self.cols = helpers._matrix_builder(
            data,
            ref_epoch,
            linear_params=linear_params,
        )

        
        
    def neg_lnlike(self,params,data,priors,param_in):
        params_dict = dict(zip(param_in,params))
        params_dict.update({name: prior.value for name, prior in priors.items() if isinstance(prior, FixedPrior)})
        lp = self.ln_prior(params_dict, priors)
        if not np.isfinite(lp):
            return np.inf

        PeTp_params = self.PeTp_transform(**params_dict)
        _, chi2 = matrix_method(self,PeTp_params, data)
        if not np.isfinite(chi2):
            return np.inf
        
        return 0.5*chi2 - lp

    def get_initial_guess(self, param_order, nwalkers):
        """
        Returns an intial guess on fitted parameters based on the data
        """
        param_in = []
        a1_guesses, p_guesses = self.lomb_scargle()
        initial_points = []
        for i in self.priors:
            prior = self.priors[i]
            if isinstance(prior, (Bounds, FixedPrior)):
                continue
            param_in.append(i)
            if i == params.a or i == params.a1 or i == params.a2:
                initial_points.append(a1_guesses[0])
            elif i == params.P:
                initial_points.append(p_guesses[0])
            else:
                initial_points.append(prior.sample(self.rng, size=1)[0])


        bounds = self._bounds(param_in)
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        initial_points = np.clip(np.asarray(initial_points, dtype=float), lower, upper)

        def bounds_transform_fn(bound):
            transform = build_transform_functions([*param_in, *self.fixed_prior_params], [bound])
            return lambda x: transform(
                **dict(zip(param_in, x)),
                **{name: prior.value for name, prior in self.fixed_prior_params.items()},
            )[bound]
        
        constraints = []
        for name, bound in self.priors.items():
            if not isinstance(bound, Bounds):
                continue
            constraints.append(NonlinearConstraint(bounds_transform_fn(name),  bound.lower, bound.upper))
        
        results = []
        for period in p_guesses:
            candidate = initial_points.copy()
            candidate[param_in.index(params.P)] = period
            candidate = np.clip(candidate, lower, upper)
            for a in (params.a, params.a1, params.a2):
                if a in param_in:
                    candidate[param_in.index(a)] = a1_guesses[0]
                    candidate = np.clip(candidate, lower, upper)

            
            result = differential_evolution(
                self.neg_lnlike, 
                bounds=bounds, 
                args=(self.data, self.priors,param_in), 
                maxiter=2000,
                polish=False,
                x0=candidate,
                rng=self.rng,
            )
            if np.isfinite(result.fun):
                results.append(result)

        if not results:
            raise RuntimeError("All initializer period candidates failed.")
        
        result = min(results, key=lambda r: r.fun)
        
        orbit = minimize(
            self.neg_lnlike, 
            x0=result.x,
            method='SLSQP', 
            args=(self.data, self.priors,param_in),
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000}
        )

        if orbit.success and np.isfinite(orbit.fun):
            best_vector = orbit.x
        else:
            best_vector = result.x

        best_prior_values = dict(
            zip(param_in, np.clip(best_vector, lower, upper))
        )
        transform = build_transform_functions([*param_in, *self.fixed_prior_params], param_order)
        best_prior_values.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
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


def matrix_method(self,params_dict,data):

    helpers._matrix_filler(self.M,self.cols,params_dict,data)
    sigma = helpers.sigma(data,params_dict,self.err)
    mu,chi2 = helpers._lsq_helper(self.M,self.eta,sigma)
    
    return mu, chi2    
