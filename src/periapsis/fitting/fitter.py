from abc import ABC, abstractmethod
import numpy as np
from orbit_package.data.data import Data
from orbit_package.fitting.results import FitResults
from orbit_package.utils.helpers import _match_param_keys

class Fitter(ABC):
    def __init__(self, m1=None, **prior_kwargs):
        """
        A Fitter defines the configuration for fitting an orbit to data, including the priors on the orbital parameters.
        """
        self.m1 = m1
        self.prior_kwargs = _match_param_keys(prior_kwargs)

    @abstractmethod
    def fit(self, data: Data) -> FitResults:
        """
        Fits the orbit to the given data.

        Parameters
        ----------
        data : Data
            Data to fit the orbit to.

        Returns
        -------
        fit_results : FitResults
            The results of the fit
        """
        pass

    def _proper_motion_fit(self, data: Data):
        """
        Fits a proper motion model to the given data.

        Parameters
        ----------
        data : Data
            Data to fit the proper motion model to.

        Returns
        -------
        results : dict
            The results of the proper motion fit
        """
        ref_epoch = getattr(data, 'ref_epoch', np.mean(data.t))
        dt = data.t - ref_epoch

        if getattr(data, 'mu_x', None) is not None and getattr(data, 'mu_y', None) is not None:
            
            
            x0 = np.sum((data.x - data.mu_x * dt) / data.x_err**2) / np.sum(1 / data.x_err**2)
            y0 = np.sum((data.y - data.mu_y * dt) / data.y_err**2) / np.sum(1 / data.y_err**2)

            mu_x = data.mu_x
            mu_y = data.mu_y
    
            dof = 2 * len(data.t) - 2
        else:
            
            A_x = np.vstack([np.ones_like(dt)/data.x_err,dt/data.x_err]).T
            b_x = data.x/data.x_err
            x0,mu_x = np.linalg.lstsq(A_x, b_x,rcond=None)[0]
        
            A_y = np.vstack([np.ones_like(dt)/data.y_err,dt/data.y_err]).T
            b_y = data.y/data.y_err
            y0,mu_y = np.linalg.lstsq(A_y, b_y,rcond=None)[0]
            dof = 2*len(data.t)-4

        chi2_x = np.sum((data.x-(x0+mu_x*dt))**2/data.x_err**2)
        chi2_y = np.sum((data.y-(y0+mu_y*dt))**2/data.y_err**2)
        chi2 = chi2_x + chi2_y
        

        return {'params':{'x0':x0,'mu_x':mu_x,'y0':y0,'mu_y':mu_y},
                'chi2':chi2,'dof':dof}

    def _astrometric_offset_seeds(self, data: Data):
        """Return sensible starting values for optional astrometric offsets."""
        pm_fit = self._proper_motion_fit(data)
        return {
            'dx': pm_fit['params']['x0'],
            'dy': pm_fit['params']['y0'],
            'dpmra': pm_fit['params']['mu_x'],
            'dpmdec': pm_fit['params']['mu_y'],
        }

    