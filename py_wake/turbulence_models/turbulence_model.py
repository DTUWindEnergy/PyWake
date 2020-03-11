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
    def calc_effective_TI(self, TI_lk, add_turb_jlk):
        """Calculate effective turbulence intensity

        Parameters
        ----------
        TI_lk : array_like
            Local turbulence intensity at x'th most upstream turbines for all wind
            directions(l) and wind speeds(k)
        add_turb_jlk : array_like
            deficit caused by upstream turbines(j) for all wind directions(l)
            and wind speeds(k)

        Returns
        -------
        TI_eff_lk : array_like
            Effective wind speed at the x'th most upstream turbines for all wind
            directions(l) and wind speeds(k)
        """


class MaxSum():
    def calc_effective_TI(self, TI_jlk, add_turb_ijlk):
        return TI_jlk + np.max(add_turb_ijlk, 0)
