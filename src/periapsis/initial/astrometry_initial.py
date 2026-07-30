from periapsis.data.data import Data
from periapsis.model.orbit import Orbit
import numpy as np
from scipy.optimize import minimize,differential_evolution
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from periapsis.params.transforms import covered_parameters, build_transform_functions
from periapsis.prior.fixed_prior import FixedPrior

from .initial import InitialGuess

class AstrometryInitialGuess(InitialGuess):
    """Class for obtaining an intial guess on fitted parameters"""
    def __init__(self, data, rng, **priors):
        super().__init__(data, rng, **priors)

    
    def lomb_scargle(self):
        """Returns an initial guess on the period and semi major axis 
        based on a Lomb-Scargle periodogram"""
        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.1
        p_max = prior_p.max if prior_p is not None else 1000
        
        frequency = np.linspace(1/p_max,1/p_min,1000)
        p1 = LombScargle(self.data.t,self.data.x,self.data.x_err)
        p2 = LombScargle(self.data.t,self.data.y,self.data.y_err)
        

        power1 = p1.power(frequency)
        power2 = p2.power(frequency)
        
        power_total = power1 + power2
        peaks,_ = find_peaks(power_total)
        best_frequency = frequency[np.argmax(power_total)]

        a = p1.model_parameters(best_frequency) #amplitude of x
        b = p2.model_parameters(best_frequency) #amplitude of y

        ampx=np.hypot(a[1],a[2]) #these two functions give us amplitude of the sin waves
        ampy=np.hypot(b[1],b[2])
        ampx_max = np.max(ampx)
        ampy_max = np.max(ampy)

        a1_guess = np.hypot(ampx_max,ampy_max)
        p_guess = 1/best_frequency
        return a1_guess, p_guess 
    
    def _bounds(self,param_names):
        """Returns bounds on the fitted parameters based on the priors"""
        bounds = []
        for name in param_names:
            prior = self.priors.get(name)
            if prior is not None:
                bounds.append((prior.min, prior.max))
            else:
                raise ValueError(f"Missing prior bounds for parameter: {name}")   
        return bounds
    
    def ln_like(self,params_dict,data):
        """Returns the log likelihood of the given parameters based on the data"""
        model = Orbit(**params_dict)
        return -0.5 * data.chi2(model)
        
    
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
                raise ValueError(f"Missing prior bounds for parameter: {name}")
        return lp
    
    def neg_lnlike(self,params,data,priors,param_in):
        params_dict = dict(zip(param_in,params))
        return -(self.ln_prior(params_dict, priors) + self.ln_like(params_dict, data))
    

    def get_initial_guess(self, param_order, nwalkers):
        """
        Returns an intial guess on fitted parameters based on the data
        """
        param_in = []
        a1_guess, p_guess = self.lomb_scargle()
        initial_points = []
        for i in self.priors:
            param_in.append(i)
            prior = self.priors[i]
            if i == "a":
                initial_points.append(a1_guess)
            elif i == "P":
                initial_points.append(p_guess)
            else:
                initial_points.append(prior.sample(self.rng, size=1)[0])
        



        bounds = self._bounds(param_in)
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        initial_points = np.clip(np.asarray(initial_points, dtype=float), lower, upper)


        result = differential_evolution(
            self.neg_lnlike, 
            bounds=bounds, 
            args=(self.data, self.priors,param_in), 
            maxiter=2000,
            polish=False
        )

        orbit = minimize(
            self.neg_lnlike, 
            x0=result.x,
            method='L-BFGS-B', 
            args=(self.data, self.priors,param_in), 
            bounds=bounds,
            options={'maxiter': 2000}
        )

        best_prior_values = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        transform = build_transform_functions(param_in, param_order)
        best_values = transform(**best_prior_values)
        poss = []
        for name in param_order:
            poss.append(best_values[name] + self.rng.normal(0, 1e-4, size=nwalkers))
        pos = np.column_stack(poss)
        return pos
