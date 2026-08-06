import numpy as np
from periapsis.data.common import RadialVelocityData
from periapsis.model.orbit import Orbit
from scipy.optimize import minimize,differential_evolution
from periapsis.params.transforms import covered_parameters, build_transform_functions
from periapsis.prior import Prior
from periapsis.prior.fixed_prior import FixedPrior

from .initial import InitialGuess


class RVInitialGuess(InitialGuess):
    "Class for obtaining intial guess for RV data"
    def __init__(self, data: RadialVelocityData, rng: np.random.RandomState, **priors):
        super().__init__(data, rng, **priors)

    def _pdc(self, p,t,a):
        #phase difference matrix
        phi = np.mod(np.abs(t[:,np.newaxis] - t), p)

        #phase distance matrix
        b = phi * (p - phi)

        #double centering of matrix a and b
        A = a - np.mean(a,axis=0,keepdims=True) - np.mean(a,axis=1,keepdims=True) + np.mean(a)
        B = b - np.mean(b,axis=0,keepdims=True) - np.mean(b,axis=1,keepdims=True) + np.mean(b)

        PDC = np.sum(A*B) / np.sqrt(np.sum(A**2) * np.sum(B**2))

        return PDC

<<<<<<< HEAD
    def Zucker_pdc(self,num_freq=10000):
        """Compute the Zucker PDC to obtain an initial guess on Period"""
        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.1
        p_max = prior_p.max if prior_p is not None else 100
        freq_min = 1/p_max
        freq_max = 1/p_min
        frequencies = np.logspace(np.log10(freq_min),np.log10(freq_max),num_freq)
        periods = 1/frequencies
        pdc_vals = np.zeros(num_freq)
=======
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
>>>>>>> 88fa497 (ooops)

        x = self.data.rv
        t = self.data.t
        #sample distance matrix
        a = np.abs(x[:,np.newaxis] - x)

        for i,p in enumerate(periods):
            pdc_vals[i] = self._pdc(p,t,a)

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
        params_dict = dict(zip(self.priors.keys(), params))
        model = Orbit(**params_dict)
        ln_like = -0.5 * data.chi2(model)

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
            polish=False
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
        transform = build_transform_functions(param_in, param_order)
        guess = transform(**prior_guess)

        poss = []
        for name in param_order:
            poss.append(guess[name] + self.rng.normal(0, 1e-4, size=nwalkers))
        pos = np.column_stack(poss)
        return pos
             

