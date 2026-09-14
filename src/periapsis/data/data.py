from abc import ABC, abstractmethod
from astropy import units as u
import numpy as np

from periapsis.model.orbit import Orbit

class Data(ABC):
    """
    A Data object represents the observational data that we want to fit an orbit to. 
    It can be extended to include different types of data, such as astrometry, radial velocities, etc.

    Classes that extend this class will map alternate representations to an absolute 7-dimensional t, x, y, z, vx, vy, vz format that can be used to fit orbits. At least one dimension besides time needs to be available for the given data.
    """

    t: np.ndarray

    @abstractmethod
    def chi2(self, orbit: Orbit) -> float:
        """
        Computes the chi-squared value of the given orbit parameters compared to the data. 
        
        Parameters
        ----------
        orbit: Orbit
            The orbit for which to compute the chi-squared value.

        Returns
        -------
        chi2 : float
            The chi-squared value of the given orbit parameters compared to the data.
        """
        pass

    @abstractmethod
    def log_likelihood(self, orbit: Orbit, offset_names=None) -> float:
        """
        Computes the log-likelihood of the given orbit parameters compared to the data. 
        
        Parameters
        ----------
        orbit: Orbit
            The orbit for which to compute the log-likelihood value.
        offset_names: list of str, optional
            A list of parameter names that should be treated as offsets in the model. 
            If provided, these parameters will be added to the model predictions before computing the log-likelihood.

        Returns
        -------
        log_likelihood : float
            The log-likelihood value of the given orbit parameters compared to the data.
        """
        pass

    @abstractmethod
    def has_astrometry(self) -> bool:
        """
        Returns True if the data contains astrometry information, False otherwise.
        """
        pass

    @abstractmethod
    def has_radial_velocity(self) -> bool:
        """
        Returns True if the data contains radial velocity information, False otherwise.
        """
        pass


    @abstractmethod
    def _astrometry(self,orbit:Orbit):
        """
        Returns x_obs,y_obs"""
        pass


    @abstractmethod
    def _radial_velocity(self,orbit:Orbit):
        """
        Returns rv_obs"""
        pass


    @abstractmethod
    def t_series(self):
        """
        Returns the time series of the data.
        """
        pass

    @property
    @abstractmethod
    def dof(self)-> int:
        """
        Returns the degrees of freedom of the data.
        """
        pass

    @abstractmethod
    def parameter_unit(self, param_name) -> u.Unit:
        pass

    @abstractmethod
    def parameter_dimension(self, param_name) -> str:
        pass