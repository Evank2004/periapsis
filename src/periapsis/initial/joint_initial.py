from .initial import InitialGuess
import periapsis.utils.helpers as helpers
from .astrometry_initial import AstrometryInitialGuess
from .rv_initial import RVInitialGuess
from .gaia_initial import GaiaLinearInitialGuess
from periapsis.prior import Bounds, FixedPrior, LogUniformPrior
from periapsis.params.transforms import build_transform_functions
from periapsis.utils.solvers import solve_kepler
from periapsis.data import Data, JointData, GaiaData, RadialVelocityData, AstrometryData
from periapsis.params import covered_parameters
import numpy as np
from scipy.optimize import differential_evolution,minimize, NonlinearConstraint

class JointLinearInitialGuess(InitialGuess):
    def __init__(self, data: JointData, rng: np.random.RandomState, ref_epoch=0.0, npeaks=3,linear_params=(), **priors):
        super().__init__(data, rng, ref_epoch, npeaks=npeaks,**priors)
        self.data = data
        self.eta = data._concat_obs_canonical()
        self.sigma = data._err_canonical()
        self.w = 1/self.sigma
        self.m_eta_w = self.eta * self.w

        self.PeTp_transform = build_transform_functions(self.priors.keys(), ('P', 'e', 'Tp'))
        
        #pre allocate large matrix
        self.M_base, self.cols = helpers._matrix_builder(
            data, ref_epoch, linear_params=linear_params
        )


    def Delisle_joint(self,npeaks=None,num_freq=10000):
        """Returns an initial guess on period based on LS of both astrometry and RV data"""

        if npeaks is None:
            npeaks = self.npeaks

        prior_p = self.priors.get('P')
        p_min = getattr(prior_p, 'min', 0.001)
        p_max = getattr(prior_p, 'max', 100)

       
        M_null,_ = helpers.null_matrix_builder(self.data,self.ref_epoch)
        

        _,chi2H = helpers._lsq_helper(M_null,self.eta,self.sigma)

        min_freq = 1/p_max
        max_freq = 1/p_min
        frequencies = np.logspace(np.log10(min_freq),np.log10(max_freq),num_freq)
        periods = 1/frequencies
        power = np.zeros(num_freq)


        for i,nu in enumerate(frequencies):
            omega = 2*np.pi*nu

            M = self.M_base.copy()
            helpers._fill_periodogram_periodic(M, self.cols,self.ref_epoch,self.data,omega)

            _,chi2K = helpers._lsq_helper(M,self.eta,self.sigma)

            z_GLS = (chi2H-chi2K)/chi2H
            power[i] = z_GLS

        top_cands = np.argsort(power)[::-1][:npeaks]
        P_guesses = periods[top_cands]
        self.period_guesses = P_guesses
        return P_guesses

    def _bounds(self,prior_param_names):
        """Returns bounds on the fitted parameters based on the priors"""
        bounds = []
        for name in prior_param_names:
            prior = self.priors.get(name)
            if prior is None:
                raise ValueError(f"Missing prior bounds for parameter: {name}")

            lower = getattr(prior, "lower", getattr(prior, "min", None))
            upper = getattr(prior, "upper", getattr(prior, "max", None))

            if lower is None or upper is None:
                raise ValueError(
                    f"Finite lower and upper bounds are required for parameter: {name}"
                )

            bounds.append((lower, upper))  
        return bounds


    def _matrix_helper(self,params_dict):

        helpers._matrix_filler(self.M_base,self.cols,params_dict,self.data)

        sigma = helpers.sigma(self.data,params_dict,self.sigma)
        _,chi2 = helpers._lsq_helper(self.M_base,self.eta,sigma)


        return chi2


    

    def lnprior(self,params_dict):
        lp = 0.0
        for name, val in params_dict.items():
            prior = self.priors.get(name)
            if prior is not None:
                if isinstance(prior,FixedPrior):
                    continue
                lp += prior.logpdf(val)
                if np.isinf(lp):
                    return -np.inf
        return lp

        
    def neg_lnlike(self,params,param_in):
        params_dict = dict(zip(param_in, params))
        params_dict.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        lp = self.lnprior(params_dict)
        if not np.isfinite(lp):
            return np.inf
        PeTp_params = self.PeTp_transform(**params_dict)
        chi2 = self._matrix_helper(PeTp_params)
        if not np.isfinite(chi2):
            return np.inf
        
        return 0.5 * chi2 - lp
       
    def _linear_seed(self):
        """ Returns an initial guess from linearization method to make it easier for small optimizer run to converge"""

        linear_params = set(self.cols)
        fixed_params = set(self.fixed_prior_params)

        linear_derived_params = covered_parameters(
            linear_params | fixed_params
        )

        param_in = [name for name, prior in self.priors.items()
            if not isinstance(prior, (Bounds, FixedPrior))
            and name not in linear_derived_params
        ]

        P_guesses = self.Delisle_joint()
        initial_points = []

        for name in param_in:
            if name == "P":
                initial_points.append(P_guesses[0])
            else:
                initial_points.append(self.priors[name].sample(self.rng, size=1)[0])

        bounds = self._bounds(param_in)
        lower = np.array([b[0] for b in bounds],dtype=float)
        upper = np.array([b[1] for b in bounds],dtype=float)
        initial_points = np.clip(np.asarray(initial_points,dtype=float), lower, upper)

        results = []

        for periods in P_guesses:
            candidates = initial_points.copy()
            candidates[param_in.index('P')] = periods
            candidates = np.clip(candidates, lower, upper)

            result = differential_evolution(
                self.neg_lnlike,
                bounds=bounds,
                args=(param_in,),
                maxiter=2000,
                polish=False,
                x0=candidates,
                rng=self.rng,
            )

            if np.isfinite(result.fun):
                results.append(result)

        if not results:
            raise RuntimeError("All initializer period candidates failed.")
        
        result = min(results, key=lambda r: r.fun)

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

        orbit = minimize(
            self.neg_lnlike,
            x0=result.x,
            bounds=bounds,
            method = 'SLSQP',
            args=(param_in,),
            constraints=constraints,
            options={'maxiter': 2000}
        )

        if orbit.success and np.isfinite(orbit.fun):
            best_vector = orbit.x
        else:
            best_vector = result.x
        best_prior_values = dict(zip(param_in, np.clip(best_vector, lower, upper)))
        best_prior_values.update({name: prior.value for name, prior in self.fixed_prior_params.items()})
        
        best_values = dict(best_prior_values)
        best_values.update(
            self.PeTp_transform(**best_prior_values)
        )

        return best_values

    def _gauss_ball(self,best_values,param_order,nwalkers):
        """
        Returns a gaussian ball around the best values for each parameter
        """
        poss = []
        for name in param_order:
            value = best_values[name]
            prior = self.priors.get(name)
            if isinstance(prior, LogUniformPrior):
                # Jitter is a positive scale and can be many orders of magnitude
                # smaller than the canonical-unit perturbation used by other params.
                values = value * 10 ** self.rng.normal(0.0, 0.01, size=nwalkers)
                poss.append(np.clip(values, prior.min, prior.max))
            else:
                if name not in best_values:
                    raise ValueError(f"Missing initial guess for parameter: {name}")
                
                prior = self.priors.get(name)
    
                scale = max(
                    1e-4 * abs(best_values[name]),
                    1e-8
                )
    
                if prior is not None and not isinstance(prior, FixedPrior):
                    lower = getattr(prior, "lower", getattr(prior, "min", None))
                    upper = getattr(prior, "upper", getattr(prior, "max", None))
                    if lower is not None and upper is not None:
                        scale = max(scale, 1e-4 * (upper - lower))
    
                values = best_values[name] + self.rng.normal(
                    0.0,
                    scale,
                    size=nwalkers,
                )
    
                if prior is not None and not isinstance(prior, FixedPrior):
                    lower = getattr(prior, "lower", getattr(prior, "min", None))
                    upper = getattr(prior, "upper", getattr(prior, "max", None))
                    if lower is not None:
                        lower = np.nextafter(lower, upper if upper is not None else np.inf)
                    if upper is not None:
                        upper = np.nextafter(upper, lower if lower is not None else -np.inf)
                    values = np.clip(values, lower, upper)
    
                poss.append(values)

        pos = np.column_stack(poss)
        return pos

    def get_initial_guess(self, param_order, nwalkers):
        """
        Returns an initial guess for parameters based on Joint Data and priors
        """
        best_values = self._linear_seed()
        pos = self._gauss_ball(best_values, param_order, nwalkers)
        return pos


class JointInitialGuess(JointLinearInitialGuess):
    def __init__(self, data, rng, ref_epoch=0.0, npeaks=3, **priors):
        super().__init__(data,rng,ref_epoch, npeaks=npeaks,
            linear_params=priors.keys(),
            **priors,
        )

    def _linear_seed(self):
        """Return a linearized seed for the full nonlinear MCMC fit."""
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

        P_guesses = self.Delisle_joint()
        initial_points = []

        for name in param_in:
            if name == "P":
                initial_points.append(P_guesses[0])
            else:
                initial_points.append(
                    self.priors[name].sample(self.rng, size=1)[0]
                )

        bounds = self._bounds(param_in)
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        initial_points = np.clip(
            np.asarray(initial_points, dtype=float),
            lower,
            upper,
        )

        def linear_neg_lnlike(params, param_in):
            params_dict = dict(zip(param_in, params))
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
                self.sigma,
            )

            ln_prior = self.lnprior(solved_params)

            if not np.isfinite(ln_prior) or not np.isfinite(chi2):
                return np.inf

            return 0.5 * chi2 - ln_prior

        def evaluate_bound(name, values):
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
                self.sigma,
            )
            return solved_params[name]

        constraints = [
            NonlinearConstraint(
                lambda values, name=name: evaluate_bound(name, values),
                bound.lower,
                bound.upper,
            )
            for name, bound in self.priors.items()
            if isinstance(bound, Bounds)
        ]

        results = []
        for periods in P_guesses:
            candidates = initial_points.copy()
            candidates[param_in.index('P')] = periods
            candidates = np.clip(candidates, lower, upper)

            result = differential_evolution(
                linear_neg_lnlike,
                bounds=bounds,
                args=(param_in,),
                maxiter=2000,
                polish=False,
                x0=candidates,
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
            self.sigma,
        )
        return best_values

    
    def get_initial_guess(self, param_order, nwalkers):
        
        best_values = self._linear_seed()


        transform = build_transform_functions(
            best_values.keys(),param_order,
        )
        best_values.update(transform(**best_values))


        positions = []
        for name in param_order:
            if name not in best_values:
                raise ValueError(
                    f"Missing initial guess for parameter: {name}"
                    )

            prior = self.priors.get(name)

            scale = max(
                1e-4 * abs(best_values[name]),
                1e-8,
            )

            if prior is not None and not isinstance(prior, FixedPrior):
                lower = getattr(prior, "lower", getattr(prior, "min", None))
                upper = getattr(prior, "upper", getattr(prior, "max", None))
                if lower is not None and upper is not None:
                    scale = max(scale, 1e-4 * (upper - lower))

            values = best_values[name] + self.rng.normal(
                0.0,
                scale,
                size=nwalkers,
            )

            if prior is not None and not isinstance(prior, FixedPrior):
                lower = getattr(prior, "lower", getattr(prior, "min", None))
                upper = getattr(prior, "upper", getattr(prior, "max", None))
                if lower is not None:
                    lower = np.nextafter(lower, upper if upper is not None else np.inf)
                if upper is not None:
                    upper = np.nextafter(upper, lower if lower is not None else -np.inf)
                values = np.clip(values, lower, upper)

            positions.append(values)
        return np.column_stack(positions)
        
