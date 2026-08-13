from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lstsq as ls
from periapsis.data import Data, AstrometryData, RadialVelocityData, JointData,GaiaData
from periapsis.utils.helpers import _lsq_helper, _null_matrix_builder
from periapsis.fitting.results import FitResults
from periapsis.prior.fixed_prior import FixedPrior

class Fitter(ABC):
    """
    A Fitter defines the configuration for fitting an orbit to data, including the priors on the orbital parameters.
    """

    def __init__(self, ref_epoch=0, **priors):
        self.priors = priors
        self.ref_epoch = ref_epoch
        self.priors['Tepoch'] = FixedPrior(ref_epoch)

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

    def _null_hypothesis_fit(self, data: Data):
        """
        Fits a null hypothesis model to the given data.
        """
        if isinstance(data,AstrometryData):
            mu_x = getattr(data, 'mu_x', None)
            mu_y = getattr(data, 'mu_y', None)
            if mu_x is not None and mu_y is not None:
                dt = data.t - self.ref_epoch
                n_obs = len(data.t)

                x_prime = data.x - mu_x * dt
                y_prime = data.y - mu_y * dt

                d = np.concatenate([x_prime, y_prime])
                sigma = np.concatenate([data.x_err, data.y_err])
                w = 1.0 / sigma

                M = np.zeros((2 * n_obs, 3))
                M[:n_obs, 0] = 1.0            # alpha0
                M[:n_obs, 2] = data.plxf_x    # parallax (RA component)
                M[n_obs:, 1] = 1.0            # delta0
                M[n_obs:, 2] = data.plxf_y    # parallax (Dec component)

                
                M_w = M * w[:, np.newaxis]
                d_w = d * w

                mu, _, _, _ = np.linalg.lstsq(M_w, d_w, rcond=None)
                alpha0, delta0, parallax = mu

                res_w = d_w - M_w @ mu
                chi2 = np.sum(res_w**2)
                dof = 2 * n_obs - 3
                return {
                    'params': {
                        'dalpha': alpha0,
                        'ddelta': delta0,
                        'mu_alpha': mu_x,
                        'mu_delta': mu_y,
                        'parallax': parallax,
                    },
                    'chi2': chi2,
                    'dof': dof,
                }
            else:
                A,cols = _null_matrix_builder(data,self.ref_epoch)
                x = np.concatenate([data.x,data.y])
                err = np.concatenate([data.x_err,data.y_err])
                mu,chi2 = _lsq_helper(A,x,err)
                dof = 2*len(data.t)-len(mu)
                return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
            
        elif isinstance(data,RadialVelocityData):
            A,cols = _null_matrix_builder(data,self.ref_epoch)
            x = data.rv
            err = data.rv_err
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(data.t)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
        elif isinstance(data,JointData):
            A,cols = _null_matrix_builder(data,self.ref_epoch)
            x = data._concat_obs()
            err = data._err()
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(x)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
        elif isinstance(data,GaiaData):
            A,cols = _null_matrix_builder(data,self.ref_epoch)
            x = data.x
            err = data.err
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(x)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}


    def _astrometric_offset_seeds(self, data: Data):
        """Return sensible starting values for optional astrometric offsets."""
        pm_fit = self._null_hypothesis_fit(data)
        return {
            'dalpha': pm_fit['params']['dalpha'],
            'ddelta': pm_fit['params']['ddelta'],
            'mu_alpha': pm_fit['params']['mu_alpha'],
            'mu_delta': pm_fit['params']['mu_delta'],
        }
