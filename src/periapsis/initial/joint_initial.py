from .initial import InitialGuess
from periapsis.utils.helpers import _lsq_helper, _matrix_builder, _matrix_filler,_null_matrix_builder,_fill_periodogram_periodic
from .astrometry_initial import AstrometryInitialGuess
from .rv_initial import RVInitialGuess
from .gaia_initial import GaiaInitialGuess
from periapsis.prior import Bounds,FixedPrior
from periapsis.params.transforms import build_transform_functions
from periapsis.utils.solvers import solve_kepler
from periapsis.data import Data, JointData, GaiaData, RadialVelocityData, AstrometryData
import numpy as np
from scipy.optimize import differential_evolution,minimize, NonlinearConstraint

class JointInitialGuess(InitialGuess):
    def __init__(self, data: JointData, rng: np.random.RandomState, ref_epoch=0.0, **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.data = data
        self.eta = data._concat_obs()
        self.sigma = data._err()
        self.w = 1/self.sigma
        self.m_eta_w = self.eta * self.w

        self.PeTp_transform = build_transform_functions(self.priors.keys(), ('P', 'e', 'Tp'))
        
        #pre allocate large matrix
        self.M_base,self.cols = _matrix_builder(data, ref_epoch)


    def Delisle_joint(self,num_freq=10000):
        """Returns an initial guess on period based on LS of both astrometry and RV data"""

        prior_p = self.priors.get('P')
        p_min = getattr(prior_p, 'min', 0.001)
        p_max = getattr(prior_p, 'max', 100)

       
        M_null,_ = _null_matrix_builder(self.data,self.ref_epoch)
        

        _,chi2H = _lsq_helper(M_null,self.eta,self.sigma)

        min_freq = 1/p_max
        max_freq = 1/p_min
        frequencies = np.logspace(np.log10(min_freq),np.log10(max_freq),num_freq)
        periods = 1/frequencies
        power = np.zeros(num_freq)


        for i,nu in enumerate(frequencies):
            omega = 2*np.pi*nu

            M = self.M_base.copy()
            _fill_periodogram_periodic(M, self.cols,self.ref_epoch,self.data,omega)

            _,chi2K = _lsq_helper(M,self.eta,self.sigma)

            z_GLS = (chi2H-chi2K)/chi2H
            power[i] = z_GLS

        P_guess = periods[np.argmax(power)]
        return P_guess

    def _bounds(self,prior_param_names):
        """Returns bounds on the fitted parameters based on the priors"""
        bounds = []
        for name in prior_param_names:
            prior = self.priors.get(name)
            if prior is not None:
                bounds.append((getattr(prior, 'min'), getattr(prior, 'max')))
            else:
                raise ValueError(f"Missing prior bounds for parameter: {name}")   
        return bounds


    def _matrix_helper(self,params_dict):

        _matrix_filler(self.M_base,self.cols,params_dict,self.data)

        M_w = self.M_base*self.w[:,np.newaxis]

        MT_M = M_w.T @ M_w
        MT_eta = M_w.T @ self.m_eta_w

        mu = np.linalg.solve(MT_M, MT_eta)
        model_werr = M_w @ mu
        resids = self.m_eta_w - model_werr
        chi2 = np.sum(resids**2)

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
        lp = self.lnprior(params_dict)
        if not np.isfinite(lp):
            return np.inf
        PeTp_params = self.PeTp_transform(**params_dict)
        chi2 = self._matrix_helper(PeTp_params)
        return 0.5 * chi2 - lp
       


    def get_initial_guess(self, param_order, nwalkers):
        """
        Returns an initial guess for parameters based on Joint Data and priors
        """
        param_in = []
        P_guess = self.Delisle_joint()
        initial_points = []
        for i in self.priors.keys():
            prior = self.priors[i]
            if isinstance(prior,Bounds):
                continue
            param_in.append(i)
            if i == 'P':
                initial_points.append(P_guess)
            else:
                initial_points.append(prior.sample(self.rng, size=1)[0])

        

        bounds = self._bounds(param_in)
        lower = np.array([b[0] for b in bounds],dtype=float)
        upper = np.array([b[1] for b in bounds],dtype=float)
        initial_points = np.clip(np.asarray(initial_points,dtype=float), lower, upper)

        result = differential_evolution(
            self.neg_lnlike,
            bounds=bounds,
            args=(param_in,),
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
            bounds=bounds,
            method = 'SLSQP',
            args=(param_in,),
            constraints=constraints,
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
