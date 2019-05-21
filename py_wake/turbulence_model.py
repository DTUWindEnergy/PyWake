from py_wake.wake_model import WakeModel
import numpy as np
from abc import abstractmethod


class TurbulenceModel(WakeModel):

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
        pass

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
        pass

    def ti_map(self, WS_ilk, WS_eff_ilk, TI_ilk, TI_eff_ilk, dw_ijl, hcw_ijl, dh_ijl, ct_ilk, types_i, TI_jlk):
        """Calculate a wake (effecitve wind speed) map

        Parameters
        ----------
        WS_ilk : array_like
            Local wind speed [m/s] for each turbine(i), wind direction(l) and
            wind speed(k)
        WS_eff_ilk : array_like
            Local effective wind speed [m/s] for each turbine(i), wind
            direction(l) and wind speed(k)
        dw_ijl : array_like
            Down wind distance matrix between turbines(i) and map points (j) for
            all wind directions(l) [m]
        hcw_ijl : array_like
            Horizontal cross wind distance matrix between turbines(i) and map
            points(j) for all wind directions(l) [m]
        dh_ijl : array_like
            Vertical hub height distance matrix between turbines(i,i) for all
            wind directions(l) [m]
        ct_ilk : array_like
            Thrust coefficient for all turbine(i), wind direction(l) and
            wind speed(k)
        types_i : array_like
            Wind turbine type indexes
        WS_jlk : array_like
            Local wind speed [m/s] for map point(j), wind direction(l) and
            wind speed(k)

        Returns
        -------
        WS_eff_jlk : array_like
            Local effective wind speed [m/s] for all map points(j),
            wind direction(l) and wind speed(k)
        """
        return self._map(self.args4addturb, self.calc_added_turbulence, self.calc_effective_TI, WS_ilk, WS_eff_ilk, TI_ilk, TI_eff_ilk, dw_ijl, hcw_ijl, dh_ijl, ct_ilk, types_i, TI_jlk)


class MaxSum():
    def calc_effective_TI(self, TI_jlk, add_turb_ijlk):
        return TI_jlk + np.max(add_turb_ijlk, 0)
