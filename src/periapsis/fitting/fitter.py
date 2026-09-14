from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lstsq as ls
from periapsis.data import Data, AstrometryData, RadialVelocityData, JointData,GaiaData
from periapsis.utils.helpers import _lsq_helper, _null_matrix_builder
from periapsis.fitting.results import FitResults
from periapsis.prior.fixed_prior import FixedPrior
from copy import deepcopy
from periapsis.params.units import CanonicalUnits


class Fitter(ABC):
    """
    A Fitter defines the configuration for fitting an orbit to data, including the priors on the orbital parameters.
    """

    def __init__(self, ref_epoch=None, **priors):
        self.priors = priors
        self.ref_epoch = ref_epoch
        self.priors['Tepoch'] = FixedPrior(0.0 if ref_epoch is None else ref_epoch)

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

    def _null_hypothesis_fit(self, data: Data,ref_epoch=None):
        """
        Fits a null hypothesis model to the given data.
        """
        if ref_epoch is None:
            ref_epoch = self.ref_epoch
            if ref_epoch is None:
                ref_epoch = getattr(data, 'ref_epoch', 0.0)
        if isinstance(data,AstrometryData):
           
            A,cols = _null_matrix_builder(data,ref_epoch)
            x = np.concatenate([data.x,data.y])
            err = np.concatenate([data.x_err,data.y_err])
            mu,chi2 = _lsq_helper(A,x,err)
            dof = 2*len(data.t)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
            
        elif isinstance(data,RadialVelocityData):
            A,cols = _null_matrix_builder(data,ref_epoch)
            x = data.rv
            err = data.rv_err
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(data.t)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
        elif isinstance(data,JointData):
            A,cols = _null_matrix_builder(data,ref_epoch)
            x = data._concat_obs()
            err = data._err()
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(x)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}
        elif isinstance(data,GaiaData):
            A,cols = _null_matrix_builder(data,ref_epoch)
            x = data.x
            err = data.err
            mu,chi2 = _lsq_helper(A,x,err)
            dof = len(x)-len(mu)
            return {'params':dict(zip(cols,mu)),'chi2':chi2,'dof':dof}


    def _astrometric_offset_seeds(self, data: Data):
        """Return sensible starting values for optional astrometric offsets."""
        pm_fit = self._null_hypothesis_fit(data)
        if pm_fit is None:
            raise RuntimeError("Null-hypothesis fitting returned no result.")
        return {
            'dalpha': pm_fit['params']['dalpha'],
            'ddelta': pm_fit['params']['ddelta'],
            'mu_alpha': pm_fit['params']['mu_alpha'],
            'mu_delta': pm_fit['params']['mu_delta'],
        }

    def _canonical_priors(self,data):
        """
        Returns deep-copy of priors scaled to canonical units"""

        canonical_priors = deepcopy(self.priors)
        data_ref_epoch = getattr(data, 'ref_epoch', None)
        if self.ref_epoch is None and data_ref_epoch is not None:
            canonical_priors['Tepoch'] = FixedPrior(data_ref_epoch)

        for name, prior in canonical_priors.items():
            if name == 'Tepoch' and self.ref_epoch is None and data_ref_epoch is not None:
                continue
            unit = data.parameter_unit(name)
            dimension = data.parameter_dimension(name)

            if unit is None or dimension is None:
                raise ValueError(f"Cannot determine unit or dimension for parameter '{name}'.")

            factor = unit.to(CanonicalUnits[dimension])
            prior.scale(factor)

        return canonical_priors

    def _reported_ref_epoch(self, data):
        if self.ref_epoch is not None:
            return self.ref_epoch
        data_ref_epoch = getattr(data, 'ref_epoch', None)
        if data_ref_epoch is None:
            return None
        unit = data.parameter_unit('t')
        return data_ref_epoch / unit.to(CanonicalUnits['time'])