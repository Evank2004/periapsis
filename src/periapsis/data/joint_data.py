from typing import List
import numpy as np

from .data import Data
from periapsis.model.orbit import Orbit
from periapsis.data.common import AstrometryData,RadialVelocityData

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

    def has_astrometry(self) -> bool:
        return any(data.has_astrometry() for data in self.datas)

    def has_radial_velocity(self) -> bool:
        return any(data.has_radial_velocity() for data in self.datas)

    def as_astrometry_data(self) -> AstrometryData:
        ''' Returns AstrometryData object containing all astrometry data in the joint data. 
        '''
        astrometry_datas = [data for data in self.datas if isinstance(data, AstrometryData)]
        if not astrometry_datas:
            raise ValueError("No astrometry data found in the joint data.")
        t = np.concatenate([data.t for data in astrometry_datas])
        x = np.concatenate([data.x for data in astrometry_datas])
        y = np.concatenate([data.y for data in astrometry_datas])
        x_err = np.concatenate([data.x_err for data in astrometry_datas])
        y_err = np.concatenate([data.y_err for data in astrometry_datas])
        return AstrometryData(t, x, y, x_err, y_err, system=self.datas[0].system) #FIXME handle multi-system case

    def as_radial_velocity_data(self) -> RadialVelocityData:
        ''' Returns RadialVelocityData object containing all radial velocity data in the joint data. 
        '''
        rv_datas = [data for data in self.datas if isinstance(data, RadialVelocityData)]
        if not rv_datas:
            raise ValueError("No radial velocity data found in the joint data.")
        t = np.concatenate([data.t for data in rv_datas])
        rv = np.concatenate([data.rv for data in rv_datas])
        rv_err = np.concatenate([data.rv_err for data in rv_datas])
        return RadialVelocityData(t, rv, rv_err, system = self.datas[0].system) #FIXME handle multi-system case


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
        