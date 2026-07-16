from abc import ABC, abstractmethod

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
    def chi2(self, orbit: Orbit):
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

    def t_series(self):
        """
        Returns x_obs,y_obs,rv_obs, t_obs"""
        
        pass