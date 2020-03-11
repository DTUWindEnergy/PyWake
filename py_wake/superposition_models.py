import numpy as np
from abc import ABC, abstractmethod


class SuperpositionModel(ABC):
    @abstractmethod
    def calc_effective_WS(self, WS_ilk, deficit_ijlk):
        """Calculate effective wind speed

        This method must be overridden by subclass

        Parameters
        ----------
        WS_ilk : array_like
            Local wind speed at turbine/site(i) for all wind
            directions(l) and wind speeds(k)
        deficit_ijlk : array_like
            deficit caused by upstream turbines(j) on all downstream turbines/points (i) for all wind directions(l)
            and wind speeds(k)

        Returns
        -------
        WS_eff_ilk : array_like
            Effective wind speed at turbine/site(i) for all wind
            directions(l) and wind speeds(k)

        """


class SquaredSum(SuperpositionModel):
    def calc_effective_WS(self, WS_ilk, deficit_ijlk):
        return WS_ilk - np.sqrt(np.sum(deficit_ijlk**2, 0))


class LinearSum(SuperpositionModel):
    def calc_effective_WS(self, WS_ilk, deficit_ijlk):
        return WS_ilk - np.sum(deficit_ijlk, 0)


class MaxSum(SuperpositionModel):
    def calc_effective_WS(self, WS_ilk, deficit_ijlk):
        return WS_ilk - np.max(deficit_ijlk, 0)
