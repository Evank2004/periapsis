from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lstsq as ls
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults

class Fitter(ABC):
    """
    A Fitter defines the configuration for fitting an orbit to data, including the priors on the orbital parameters.
    """

    def __init__(self, **priors):
        self.priors = priors

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
        ref_epoch = getattr(data, 'ref_epoch', np.mean(data.t)) #FIXME Tepoch in fixed priors?
        dt = data.t - ref_epoch

        if getattr(data, 'mu_x', None) is not None and getattr(data, 'mu_y', None) is not None:
            
            
            x0 = np.sum((data.x - data.mu_x * dt) / data.x_err**2) / np.sum(1 / data.x_err**2)
            y0 = np.sum((data.y - data.mu_y * dt) / data.y_err**2) / np.sum(1 / data.y_err**2)

            mu_x = data.mu_x
            mu_y = data.mu_y
    
            dof = 2 * len(data.t) - 2
        else:
            if not (
                np.all(np.isfinite(data.x))
                and np.all(np.isfinite(data.x_err))
                and np.all(np.isfinite(dt))
            ):
                raise ValueError(
                    "Data contains NaN or Inf values in x, x_err, or time arrays."
            )
            if not (
                np.all(np.isfinite(data.y))
                and np.all(np.isfinite(data.y_err))
            ):
                raise ValueError(
                "Data contains NaN or Inf values in y or y_err arrays."
            )
            
            A_x = np.vstack([np.ones_like(dt)/data.x_err,dt/data.x_err]).T
            b_x = data.x/data.x_err
            x0,mu_x = ls(A_x, b_x,lapack_driver="gelsy")[0]
        
            A_y = np.vstack([np.ones_like(dt)/data.y_err,dt/data.y_err]).T
            b_y = data.y/data.y_err
            y0,mu_y = ls(A_y, b_y,lapack_driver="gelsy")[0]
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

    def _systemic_velocity(self,data:Data):
        '''
        Returns the systemic velocity of the system from the data
        '''

        rv = data.rv 
        rv_err = data.rv_err
        N = len(rv)
        w = 1/rv_err**2
        gamma = np.sum(rv*w)/np.sum(w)

        chi2 = np.sum(((rv-gamma)/rv_err)**2)
        dof = N-1

        return {'gamma':gamma,'chi2':chi2,'dof':dof}