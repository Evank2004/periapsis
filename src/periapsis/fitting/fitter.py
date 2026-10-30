from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lstsq as ls
from periapsis.data import Data, AstrometryData, PhysicalAstrometryData, RadialVelocityData, JointData,GaiaData
from periapsis.utils.helpers import _lsq_helper, null_matrix_builder
from periapsis.fitting.results import FitResults
from periapsis.prior.fixed_prior import FixedPrior
from periapsis.params.units import CanonicalUnits
from astropy.units import dimensionless_unscaled
import warnings
from typing import Optional


class Fitter(ABC):
    """
    A Fitter defines the configuration for fitting an orbit to data, including the priors on the orbital parameters.
    """

    def __init__(self, ref_epoch=None, **priors):
        self._use_data_ref_epoch = 'Tepoch' not in priors and ref_epoch is None
        self.priors = priors
        self.ref_epoch = ref_epoch
        if 'Tepoch' not in self.priors:
            self.priors['Tepoch'] = FixedPrior(0.0 if ref_epoch is None else ref_epoch)

    def fit(self, data: Data, rng: np.random.RandomState) -> FitResults:
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
        canonical_priors = self._canonical_priors(data)
        canonical_ref_epoch = canonical_priors['Tepoch'].value if 'Tepoch' in canonical_priors else self.ref_epoch

        null_hypothesis_canonical = self._null_hypothesis_fit(
            data,
            ref_epoch=canonical_ref_epoch,
        )
        if null_hypothesis_canonical is None:
             raise RuntimeError("Null-hypothesis fitting returned no result.")

        fit_results_dict = self._fit(data, canonical_priors, canonical_ref_epoch, rng)

        fit_results_dict['priors'] = self.priors
        fit_results_dict['canonical_priors'] = canonical_priors
        fit_results_dict['null_hypothesis'] = null_hypothesis_canonical

        fit_results_dict['data'] = data

        # FIXME
        # parameter_factors = {}
        # known_output_names = set(self.param_order) | set(canonical_priors)
        # for name in known_output_names:
        #     unit = data.parameter_unit(name)
        #     dimension = data.parameter_dimension(name)
        #     if unit is not None and dimension is not None:
        #         parameter_factors[name] = unit.to(CanonicalUnits[dimension])
        # fit_results_dict['parameter_factors'] = parameter_factors

        # FIXME
        fit_results_dict['ref_epoch'] = self._reported_ref_epoch(data)
        
        return FitResults(**fit_results_dict)

    @abstractmethod
    def _fit(self, data: Data, canonical_priors: dict, canonical_ref_epoch: Optional[float], rng: np.random.RandomState) -> dict:
        """
        Internal method to perform the actual fitting. This should be implemented by subclasses.

        Parameters
        ----------
        data : Data
            Data to fit the orbit to.
        canonical_priors : dict
            Priors scaled to canonical units.

        Returns
        -------
        fit_results : dict
            The results of the fit in canonical units.
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
        A, cols = null_matrix_builder(data, ref_epoch)
        if isinstance(data, AstrometryData):
            x = np.concatenate([data.x_ang_canonical,data.y_ang_canonical])
            err = np.concatenate([data.x_err_ang_canonical,data.y_err_ang_canonical])
        elif isinstance(data, PhysicalAstrometryData):
            x = np.concatenate([data.x_canonical,data.y_canonical])
            err = np.concatenate([data.x_err_canonical,data.y_err_canonical])
        elif isinstance(data, RadialVelocityData):
            x = data.rv_canonical
            err = data.rv_err_canonical
        elif isinstance(data, JointData):
            x = data._concat_obs_canonical()
            err = data._err_canonical()
        elif isinstance(data, GaiaData):
            x = data.x_canonical
            err = data.err_canonical
        mu, chi2 = _lsq_helper(A,x,err)
        dof = len(x) - len(mu)
        return {
            'params': dict(zip(cols, mu)),
            'chi2': chi2,
            'dof': dof
        }


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
        Returns a list of priors scaled to canonical units, using the data's units if a unit isn't defined on the prior.
        """

        canonical_priors = {param: pr.canonical for param, pr in self.priors.items()}
        data_ref_epoch = getattr(data, 'ref_epoch', None)
        if self._use_data_ref_epoch and data_ref_epoch is not None:
            canonical_priors['Tepoch'] = FixedPrior(data_ref_epoch, unit=data.parameter_unit('t'))

        for name, _ in canonical_priors.items():
            if name == 'Tepoch' and self._use_data_ref_epoch and data_ref_epoch is not None:
                continue  # The Tepoch prior should already be properly canonicalized
            if self.priors[name].unit is not None:
                continue  # Prior already has a unit defined, no need to scale
            else:
                unit = data.parameter_unit(name)
                if unit is None:
                    raise ValueError(f"Cannot determine unit for parameter '{name}'.")
                # If unit is not dimensionless, warn the user that we are assuming the prior is in the same units as the data!
                if not unit.is_equivalent(dimensionless_unscaled):
                    print(f"ASSUMPTION: Assuming prior on '{name}' is in units of {unit}. To suppress this warning, explicitly set the unit on the prior.")
                canonical_priors[name].set_unit(unit)

        return canonical_priors

    def _reported_ref_epoch(self, data):
        if self.ref_epoch is not None:
            return self.ref_epoch
        data_ref_epoch = getattr(data, 'ref_epoch', None)
        if data_ref_epoch is None:
            return None
        unit = data.parameter_unit('t')
        return data_ref_epoch / unit.to(CanonicalUnits['time'])
