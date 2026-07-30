from typing import List
import numpy as np

from .data import Data
from periapsis.model.orbit import Orbit

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
        