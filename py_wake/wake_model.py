from abc import abstractmethod, ABC

from numpy import newaxis as na

import numpy as np


class WakeModel(ABC):
    """
    Base class for wake models

    Make a subclass and implement calc_deficit and calc_effective_WS.

    >>> class MyWakeModel(WakeModel):
    >>>    args4deficit = ['WS_lk', 'dw_jl'] # specify arguments required by calc_deficit
    >>>
    >>>    def calc_deficit(self, WS_lk, dw_jl):
    >>>        deficit_jlk = ...
    >>>        return deficit_jlk
    >>>
    >>>    def calc_effective_WS(self, WS_lk, deficit_jlk):
    >>>        deficit_sum_lk = ...
    >>>        return WS_lk - deficit_sum_lk

    Implementations of linear and squared sum method for calc_effective_WS are
    available through inherritance:

    >>> class MySquaredSumWakeModel(SquaredSum, WakeModel):
    >>>     ...

    >>> class MySquaredSumWakeModel(LinearSum, WakeModel):
    >>>     ...


    Suffixes:

    - d: turbines down wind order
    - i: turbines ordered by id
    - j: downstream points/turbines
    - k: wind speeds
    - l: wind directions
    - m: turbines and wind directions (il.flatten())
    - n: from_turbines, to_turbines and wind directions (iil.flatten())

    Arguments available for calc_deficit (specifiy in args4deficit):

    - WS_lk: Local wind speed without wake effects
    - WS_eff_lk: Local wind speed with wake effects
    - D_src_l: Diameter of source turbine
    - D_dst_jl: Diameter of destination turbine
    - dw_jl: Downwind distance from turbine i to point/turbine j
    - hcw_jl: Horizontal cross wind distance from turbine i to point/turbine j
    - cw_jl: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_lk: Thrust coefficient

    """

    args4deficit = ['WS_lk']

    def __init__(self, windTurbines):
        """Initialize WakeModel

        Parameters
        ----------
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines

        """
        self.windTurbines = windTurbines

    def calc_wake(self, WS_ilk, TI_ilk, dw_iil, hcw_iil, dh_iil, dw_order_indices_dl, types_i):
        """Calculate wake effects

        Calculate effective wind speed, turbulence intensity (not
        implemented yet), power and thrust coefficient

        Parameters
        ----------
        WS_ilk : array_like
            Local wind speed [m/s] for each turbine(i), wind direction(l) and
            wind speed(k)
        TI_ilk : array_like
            Local turbulence intensity for each turbine(i), wind direction(l) and
            wind speed(k)
        dw_iil : array_like
            Down wind distance matrix between turbines(i,i) for all wind
            directions(l) [m]
        hcw_iil : array_like
            Horizontal cross wind distance matrix between turbines(i,i) for all wind
            directions(l) [m]
        dh_iil : array_like
            Vertical hub height distance matrix between turbines(i,i) for all
            wind directions(l) [m]
        dw_order_indices_l : array_like
            Indices of turbines in down wind order(d) for all
            wind directions(l)
        types_i : array_like
            Wind turbine type indexes

        Returns
        -------
        WS_eff_ilk : array_like
            Effective wind speeds [m/s]
        TI_ilk : array_like
            Turbulence intensities. Should be effective, but not implemented yet
        power_ilk : array_like
            Power productions [w]
        ct_ilk : array_like
            Trust coefficients

        """
        I, L = dw_iil.shape[1:]
        i1, i2, _ = np.where((np.abs(dw_iil) + np.abs(hcw_iil) + np.eye(I)[:, :, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" %
                             (i1[i], i2[i]) for i in np.unique([i1, i2], 0)])
            raise ValueError(msg)

        K = WS_ilk.shape[2]
        deficit_nk = np.zeros((I * I * L, K))
        # deficit_ijlk = deficit_nk.reshape((I, I, L, K))  # from i to j

        indices = np.arange(I * I * L).reshape((I, I, L))
        WS_mk = WS_ilk.astype(np.float).reshape((I * L, K))
        WS_eff_mk = WS_mk.copy()
        dw_n = dw_iil.flatten()
        hcw_n = hcw_iil.flatten()
        dh_n = dh_iil.flatten()
        power_ilk = np.zeros((I, L, K))
        ct_ilk = np.zeros((I, L, K))
        types_i = np.asarray(types_i)
        D_i = self.windTurbines.diameter(types_i)
        H_i = self.windTurbines.hub_height(types_i)
        i_wd_l = np.arange(L)

        for j in range(I):
            i_wt_l = dw_order_indices_dl[:, j]
            m = i_wt_l * L + i_wd_l  # current wt (j'th most upstream wts for all wdirs)
#             n_uw = np.array([indices[dw_order_indices_l[l, :j], i, l] for i, l in zip(i_wt_l, i_wd_l)]).T
#
#             n_dw = np.array([indices[i, dw_order_indices_l[l, j + 1:], l] for i, l in zip(i_wt_l, i_wd_l)]).T

            n_uw = np.array([indices[uwi, i, l] for uwi, i, l in zip(dw_order_indices_dl[:, :j], i_wt_l, i_wd_l)]).T
            n_dw = np.array([indices[i, dwi, l] for dwi, i, l in zip(dw_order_indices_dl[:, j + 1:], i_wt_l, i_wd_l)]).T

            WS_eff_lk = self.calc_effective_WS(WS_mk[m], deficit_nk[n_uw])
            WS_eff_mk[m] = WS_eff_lk
            ct_lk, power_lk = self.windTurbines._ct_power(WS_eff_lk, types_i[i_wt_l])

            power_ilk[i_wt_l, i_wd_l] = power_lk
            ct_ilk[i_wt_l, i_wd_l, :] = ct_lk
            if j < I - 1:
                arg_funcs = {'WS_lk': lambda: WS_mk[m],
                             'WS_eff_lk': lambda: WS_eff_mk[m],
                             'D_src_l': lambda: D_i[i_wt_l],
                             'D_dst_jl': lambda: D_i[dw_order_indices_dl[:, j + 1:]].T,
                             'dw_jl': lambda: dw_n[n_dw],
                             'cw_jl': lambda: np.sqrt(hcw_n[n_dw]**2 + dh_n[n_dw]**2),
                             'hcw_jl': lambda: hcw_n[n_dw],
                             'dh_jl': lambda: dh_n[n_dw],
                             'h_l': lambda: H_i[i_wt_l],
                             'ct_lk': lambda: ct_lk}
                args = {k: arg_funcs[k]() for k in self.args4deficit}
                deficit_nk[n_dw] = self.calc_deficit(**args)

        WS_eff_ilk = WS_eff_mk.reshape((I, L, K))
        return WS_eff_ilk, TI_ilk, power_ilk, ct_ilk

    def wake_map(self, WS_ilk, WS_eff_ilk, dw_ijl, hcw_ijl, dh_ijl, ct_ilk, types_i, WS_jlk):
        D_i = self.windTurbines.diameter(types_i)
        H_i = self.windTurbines.hub_height(types_i)
        I, J, L = dw_ijl.shape
        K = WS_ilk.shape[2]

        deficit_ijlk = []
        for i in range(I):
            deficit_jlk = np.zeros((J, L, K))

            for l in range(L):
                m = dw_ijl[i, :, l] > 0

                arg_funcs = {'WS_lk': lambda: WS_ilk[i, l][na],
                             'WS_eff_lk': lambda: WS_eff_ilk[i, l][na],
                             'D_src_l': lambda: D_i[i][na],
                             'D_dst_jl': lambda: None,
                             'dw_jl': lambda: dw_ijl[i, :, l][m][:, na],
                             'cw_jl': lambda: np.sqrt(hcw_ijl[i, :, l][m]**2 + dh_ijl[i, :, l][m]**2)[:, na],
                             'hcw_jl': lambda: hcw_ijl[i, :, l][m][:, na],
                             'dh_jl': lambda: dh_ijl[i, :, l][m][:, na],
                             'h_l': lambda: H_i[i][na],
                             'ct_lk': lambda: ct_ilk[i, l][na]}

                args = {k: arg_funcs[k]() for k in self.args4deficit}

                deficit_jlk[:, l][m] = self.calc_deficit(**args)[:, 0]

            deficit_ijlk.append(deficit_jlk)
        deficit_ijlk = np.array(deficit_ijlk)
        return self.calc_effective_WS(WS_jlk, deficit_ijlk)

    @abstractmethod
    def calc_deficit(self):
        """Calculate wake deficit caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deficit

        See class documentation for examples and available arguments

        Returns
        -------
        deficit_jlk : array_like
        """
        pass

    @abstractmethod
    def calc_effective_WS(self, WS_lk, deficit_jlk):
        """Calculate effective wind speed

        This method must be overridden by subclass or by adding SquaredSum or
        LinearSum as base class, see examples in WakeModel documentation

        Parameters
        ----------
        WS_lk : array_like
            Local wind speed at x'th most upstream turbines for all wind
            directions(l) and wind speeds(k)
        deficit_jlk : array_like
            deficit caused by upstream turbines(j) for all wind directions(l)
            and wind speeds(k)

        Returns
        -------
        WS_eff_lk : array_like
            Effective wind speed at the x'th most upstream turbines for all wind
            directions(l) and wind speeds(k)

        See Also
        --------
        WakeModel

        """
        pass


class SquaredSum():
    def calc_effective_WS(self, WS_lk, deficit_jlk):
        return WS_lk - np.sqrt(np.sum(deficit_jlk**2, 0))


class LinearSum():
    def calc_effective_WS(self, WS_lk, deficit_jlk):
        return WS_lk - np.sum(deficit_jlk, 0)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm,\
            read_iea37_windrose
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.site._site import UniformSite
        from py_wake.aep_calculator import AEPCalculator

        class MyWakeModel(SquaredSum, WakeModel):
            args4deficit = ['WS_lk', 'dw_jl']

            def calc_deficit(self, WS_lk, dw_jl):
                # 10% deficit downstream
                return (WS_lk * .1)[na] * (dw_jl > 0)[:, :, na]

        _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        n_wt = 16
        x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)

        site = UniformSite(freq, ti=0.75)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        wake_model = MyWakeModel(windTurbines)
        aep_calculator = AEPCalculator(site, windTurbines, wake_model)

        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0, 30], ws=[9], levels=np.linspace(5, 9, 100))
        windTurbines.plot(x, y)

        plt.show()


main()
