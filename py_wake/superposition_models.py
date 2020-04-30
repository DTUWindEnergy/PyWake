import numpy as np
from abc import ABC, abstractmethod


class SuperpositionModel(ABC):
    @abstractmethod
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        """Calculate effective wind speed

        This method must be overridden by subclass

        Parameters
        ----------
        WS_xxx : array_like
            Local wind speed. xxx optionally includes destination turbine/site, wind directions, wind speeds
        deficit_jxxx : array_like
            deficit caused by source turbines(j) on xxx (see above)

        Returns
        -------
        WS_eff_xxx : array_like
            Effective wind speed for xxx (see WS_xxx)

        """


class SquaredSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.sqrt(np.sum(deficit_jxxx**2, 0))


class LinearSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.sum(deficit_jxxx, 0)


class MaxSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.max(deficit_jxxx, 0)
