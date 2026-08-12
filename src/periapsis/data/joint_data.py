from typing import List
import numpy as np

from .data import Data
from periapsis.model.orbit import Orbit
from periapsis.data import AstrometryData,RadialVelocityData,GaiaData

class JointData(Data):
    def __init__(self, datas: List[Data]):
        self.datas = datas

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        total_dof = 0
        for data in self.datas:
            total_dof += data.dof()
        return total_dof

    def chi2(self, orbit: Orbit):
        total_chi2 = 0
        for data in self.datas:
            total_chi2 += data.chi2(orbit)
        return total_chi2

    
    def _flatten_joint(self):
        """
        Flattens a JointData object into a list of its constituent data objects.
        """
        if isinstance(self, JointData):
            return self.datas
        else:
            return [self]

    def as_astrometry_data(self):
        ''' Returns AstrometryData object containing all astrometry data of simialar system in Joint Data. 
        '''
        astrometry_datas = [data for data in self.datas if isinstance(data, AstrometryData)]
        if not astrometry_datas:
            raise ValueError("No astrometry data found in the joint data.")
        for system in set(data.system for data in astrometry_datas):
            system_datas = [data for data in astrometry_datas if data.system == system]
            t = np.concatenate([data.t for data in system_datas])
            x = np.concatenate([data.x for data in system_datas])
            y = np.concatenate([data.y for data in system_datas])
            x_err = np.concatenate([data.x_err for data in system_datas])
            y_err = np.concatenate([data.y_err for data in system_datas])
            plxf_x = np.concatenate([data.plxf_x for data in system_datas])
            plxf_y = np.concatenate([data.plxf_y for data in system_datas])
            return AstrometryData(t, x, y, x_err, y_err,plxf_x,plxf_y, system=system)
        
        

    def as_radial_velocity_data(self):
        ''' Returns RadialVelocityData object containing all radial velocity data of similair system. 
        '''
        rv_datas = [data for data in self.datas if isinstance(data, RadialVelocityData)]
        if not rv_datas:
            raise ValueError("No radial velocity data found in the joint data.")
        for system in set(data.system for data in rv_datas):
            system_datas = [data for data in rv_datas if data.system == system]
            t = np.concatenate([data.t for data in system_datas])
            rv = np.concatenate([data.rv for data in system_datas])
            rv_err = np.concatenate([data.rv_err for data in system_datas])
            return RadialVelocityData(t, rv, rv_err, system=system)


    def _concat_obs(self):
        """Returns the combined observations for all data."""
        observations = []
        for data in self.datas:
            if isinstance(data, AstrometryData):
                observations += ([data.x, data.y])
            elif isinstance(data, RadialVelocityData):
                observations.append(data.rv)
            elif isinstance(data, GaiaData):
                observations.append(data.x)
        return np.concatenate(observations)

    def _err(self):
        """Returns the combined error array for all data."""
        errors = []
        for data in self.datas:
            if isinstance(data, AstrometryData):
                errors+=([data.x_err, data.y_err])
            elif isinstance(data, RadialVelocityData):
                errors.append(data.rv_err)
            elif isinstance(data, GaiaData):
                errors.append(data.err)
        return np.concatenate(errors)


    def _astrometry(self, orbit: Orbit):
        xs, ys = [], []
        for data in self.datas:
            try:
                x, y = data._astrometry(orbit)
                xs.append(x)
                ys.append(y)
            except NotImplementedError:
                continue
        return np.concatenate(xs), np.concatenate(ys)



    def _radial_velocity(self, orbit: Orbit):
        rvs = []
        for data in self.datas:
            try:
                if hasattr(data, "_radial_velocity"):
                    rv = data._radial_velocity(orbit)
                    rvs.append(rv)
            except NotImplementedError:
                continue
        return np.concatenate(rvs)
    


    def t_series(self):
        ts = []
        for data in self.datas:
            t = data.t_series()
            ts.append(t)
        return np.concatenate(ts)


