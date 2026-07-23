import numpy as np
from periapsis.utils.helpers import _match_param_keys
from periapsis.utils.helpers import _helper_for_periodogram
from periapsis.data.gaia import GaiaData


class GaiaInitialFit():
    """Class for obtaining initial guess for Gaia data"""
    def __init__(self, data, **priors):
        self.data = data
        self.priors = _match_param_keys(priors)
        self.rng = np.random.default_rng()

    def Delisle_periodogram(self,num_freq=10000):
        """Compute the Delisle periodogram to obtain an initial guess on Period"""

        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.1
        p_max = prior_p.max if prior_p is not None else 100 

        A_base = np.column_stack([self.data.spsi,self.data.cpsi,
                                  self.data.plx_fac,
                                  self.data.spsi*self.data.t,
                                  self.data.cpsi*self.data.t])


        _,chi2H = _helper_for_periodogram(A_base,self.data.x,self.data.err)

        min_freq = 1/p_max
        max_freq = 1/p_min
        frequencies = np.logspace(np.log10(min_freq),np.log10(max_freq),num_freq)
        periods = 1/frequencies
        power = np.zeros(num_freq)

        base_col = [self.data.spsi,self.data.cpsi,
                    self.data.plx_fac,
                    self.data.spsi*self.data.t,
                    self.data.cpsi*self.data.t]
        

        for i, nu in enumerate(frequencies):
            phase = 2 * np.pi * nu * self.data.t
            cosp = np.cos(phase)
            sinp = np.sin(phase)

            cols = base_col + [
            cosp * self.data.spsi,  # B
            sinp * self.data.spsi,  # G
            cosp * self.data.cpsi,  # A
            sinp * self.data.cpsi   # F
        ]

            A = np.column_stack(cols)
            _,chi2K = _helper_for_periodogram(A,self.data.x,self.data.err)

            z_GLS = (chi2H - chi2K) / chi2H

            power[i] = z_GLS

        P_guess = periods[np.argmax(power)]
        max_pwr = power[np.argmax(power)]

        return P_guess, max_pwr
    

    def initial_guess(self):
        "Returns Period guess from Delisle periodogram and random samples of other parameters"
        P_guess, _ = self.Delisle_periodogram()
        initial = []
        for i in self.priors:
            if i == 'P':
                initial.append(P_guess)
            else:
                prior = self.priors[i]
                initial.append(self.rng.uniform(prior.min,prior.max))

        return dict(zip(self.priors.keys(),initial))