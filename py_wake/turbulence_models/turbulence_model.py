import numpy as np
from abc import abstractmethod


class TurbulenceModel():

    @abstractmethod
    def calc_added_turbulence(self):
        """Calculate added turbulence intensity caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4addturb

        See class documentation for examples and available arguments

        Returns
        -------
        add_turb_jlk : array_like
        """

    @abstractmethod
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        """Calculate effective turbulence intensity

        Parameters
        ----------
        TI_xxx : array_like
            Local turbulence intensity. xxx optionally includes destination turbine/site, wind directions, wind speeds
        add_turb_jxxx : array_like
            added turbulence caused by source turbines(j) on xxx (see above)

        Returns
        -------
        TI_eff_xxx : array_like
            Effective turbulence intensity xxx (see TI_xxx)
        """


class LinearSum():
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return TI_xxx + np.sum(add_turb_jxxx, 0)


class MaxSum():
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return TI_xxx + np.max(add_turb_jxxx, 0)


class SqrMaxSum():
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return np.sqrt(TI_xxx**2 + np.max(add_turb_jxxx, 0)**2)
