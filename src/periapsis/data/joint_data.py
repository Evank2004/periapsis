from typing import List

from .data import Data
from orbit_package.model.orbit import Orbit

class JointData(Data):
    def __init__(self, datas: List[Data]):
        self.datas = datas

    def chi2(self, orbit: Orbit):
        total_chi2 = 0
        for data in self.datas:
            total_chi2 += data.chi2(orbit)
        return total_chi2