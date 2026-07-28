import numpy as np
from periapsis.data.common import RadialVelocityData
from periapsis.utils.helpers import _match_param_keys
from periapsis.model.orbit import Orbit
from scipy.optimize import minimize,differential_evolution


class RVInitialFit():
    "Class for obtaining intial guess for RV data"
    def __init__(self, data: RadialVelocityData, **priors):
        self.data = data
        self.priors = _match_param_keys(priors)
        self.rng = np.random.default_rng()

        def _pdc(p,t,a):
            #phase difference matrix
            phi = np.mod(np.abs(t[:,np.newaxis] - t), p)

            #phase distance matrix
            b = phi * (p - phi)

            #double centering of matrix a and b
            A = a - np.mean(a,axis=0,keepdims=True) - np.mean(a,axis=1,keepdims=True) + np.mean(a)
            B = b - np.mean(b,axis=0,keepdims=True) - np.mean(b,axis=1,keepdims=True) + np.mean(b)

            PDC = np.sum(A*B) / np.sqrt(np.sum(A**2) * np.sum(B**2))

            return PDC

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

            x = self.data.rv
            t = self.data.t
            #sample distance matrix
            a = np.abs(x[:,np.newaxis] - x)

            for i,p in enumerate(periods):
                pdc_vals[i] = _pdc(p,t,a)

            P_guess = periods[np.argmax(pdc_vals)]

            return P_guess

        def Bounds(param_names):
            """Returns bounds on the fitted parameters based on the priors"""
            bounds = []
            for name in param_names:
                prior = self.priors.get(name)
                if prior is not None:
                    bounds.append((prior.min, prior.max))
                else:
                    print(f"Warning:Missing prior for {name}.")
                    
            return bounds

        def neg_lnlike(params_dict,data):

            params_dict = _match_param_keys(params_dict)
            model = Orbit.rv(data.t, **params_dict, system=data.system)
            ln_like = -0.5 * data.chi2(model)

            lp = 0
            for name, val in params_dict.items():
                prior = self.priors.get(name)
                if prior is not None:
                    lp += prior.logpdf(val)
                    if np.isinf(lp):
                        return -np.inf
                else:
                    print(f"Warning:Missing prior for {name}.")

            neg_ln_like = -(ln_like + lp)

            return neg_ln_like


        def get_initial_guess(self):
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
                        initial_points.append(prior.rvs(random_state=self.rng))
                    else:
                        print(f"Warning:Missing prior for {i}.")
                param_in.append(i)

            bounds = self.Bounds(param_in)
            lower = np.array([b[0] for b in bounds], dtype=float)
            upper = np.array([b[1] for b in bounds], dtype=float)
            initial_points = np.clip(np.asarray(initial_points, dtype=float), lower, upper)
            
            result = differential_evolution(
                neg_lnlike, 
                bounds=bounds, 
                args=(self.data,),
                maxiter=2000,
                polish=False
            )

            orbit = minimize(
                neg_lnlike,
                x0=result.x,
                method='L-BFGS-B',
                args=(self.data,),
                bounds=bounds,
                options={'maxiter': 2000}
            )

            return dict(zip(param_in, np.clip(orbit.x, lower, upper)))
             

