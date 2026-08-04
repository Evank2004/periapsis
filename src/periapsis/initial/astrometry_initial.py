from periapsis.model.orbit import Orbit
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, LinearConstraint
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from periapsis.params.transforms import build_transform_functions
from periapsis.prior import FixedPrior, Bounds
from periapsis.utils.solvers import solve_kepler

from .initial import InitialGuess

class AstrometryInitialGuess(InitialGuess):
    """Class for obtaining an intial guess on fitted parameters"""
    def __init__(self, data, rng, **priors):
        super().__init__(data, rng, **priors)
        

    
    def lomb_scargle(self):
        """Returns an initial guess on the period and semi major axis 
        based on a Lomb-Scargle periodogram"""
        prior_p = self.priors.get('P')
        if prior_p is None:
            raise ValueError("Computing a periodogram requires a direct prior on the period 'P'. Please transform your priors to include a direct prior on 'P' or use a different method for initial guess.")
        p_min = prior_p.min
        p_max = prior_p.max
        
        frequency = np.linspace(1/p_max,1/p_min,100000)
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
        for i in self.priors.keys():
            prior = self.priors[i]
            if isinstance(prior, Bounds):
                continue
            param_in.append(i)
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

        def bounds_transform_fn(bound):
            transform = build_transform_functions(param_in, [bound])
            return lambda x: transform(**dict(zip(param_in, x)))[bound]

        constraints = []
        for name, bound in self.priors.items():
            if not isinstance(bound, Bounds):
                continue
            constraints.append(NonlinearConstraint(bounds_transform_fn(name),  bound.lower, bound.upper))

        orbit = minimize(
            self.neg_lnlike, 
            x0=result.x,
            method='SLSQP', 
            args=(self.data, self.priors,param_in), 
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000}
        )

        best_prior_values = dict(zip(param_in, np.clip(orbit.x, lower, upper)))
        transform = build_transform_functions(param_in, param_order)
        best_values = transform(**best_prior_values)
        poss = []
        for name in param_order:
            poss.append(best_values[name] + self.rng.normal(0, 1e-4, size=nwalkers) * best_values[name])
        pos = np.column_stack(poss)
        return pos


class AstrometryLinearInitialGuess(AstrometryInitialGuess):
    def __init__(self, data, rng, **priors):
        super().__init__(data, rng, **priors)
        self.PeTp_transform = build_transform_functions(self.priors.keys(), ('P', 'e', 'Tp'))

    def neg_lnlike(self,params,data,priors,param_in):
        params_dict = dict(zip(param_in,params))
        lp = self.ln_prior(params_dict, priors)
        if np.isinf(lp):
            return -np.inf

        PeTp_params = self.PeTp_transform(**params_dict)
        _, chi2 = matrix_method(PeTp_params, data)
        
        return 0.5*chi2 - lp

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

        transform = build_transform_functions(param_in, param_order)

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
        
        best_values = transform(**best_prior_values)
        poss = []
        for name in param_order:
            poss.append(best_values[name] + self.rng.normal(0, 1e-4, size=nwalkers))
        pos = np.column_stack(poss)
        return pos


def matrix_method(params_dict,data):
    P,e,Tp = params_dict["P"], params_dict["e"], params_dict["Tp"]
    Ma = 2*np.pi * (data.t - Tp) / P
    E = solve_kepler(Ma,e)

    nobs = len(data.t)
    dt = data.t - data.ref_epoch

    M = np.zeros((2*nobs,8))

    eta =np.concatenate((data.x,data.y))
    sigma = np.concatenate((data.x_err,data.y_err))


    X = np.cos(E) - params_dict['e']
    Y = np.sqrt(1-params_dict['e']**2)*np.sin(E)

    M[:nobs,0] = 1 #dx
    M[:nobs,1] = dt #pmra
    M[:nobs,2] = X # A
    M[:nobs,3] = Y # F

    # now bottom half y obs
    M[nobs:,4] = 1 #dy
    M[nobs:,5] = dt #pmdec
    M[nobs:,6] = X # B
    M[nobs:,7] = Y # G

    #now we need to get covariance matrix
    # which diagnol matrix, with err_x^2 on top and err_y^2 on bottom
    # so we can just say C^-1 is equivalent to (A*w) ....
    w = 1/sigma # just do 1/sigma to keep track of where the weights have been applied
    # now we can calculate M^T C^-1 M and M^T C^-1 eta
    eta_w = eta * w 
    M_w = M * w[:, None] # multiply each row of M by corresponding weight

    MTM = M_w.T @ M_w
    MT_eta = M_w.T @ eta_w # matching equation
    # now we can solve for mu using np.linalg.solve
    mu = np.linalg.solve(MTM, MT_eta) # dx,pmra,B,G,dy,pmdec,A,F

    model_werr = M_w @ mu # this is the model prediction with the error already over
    # this is (obs - model)/err
    resids = eta_w - model_werr
    chi2 = np.sum(resids**2)
    
    return mu, chi2    
