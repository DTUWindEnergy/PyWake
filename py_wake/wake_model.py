from abc import abstractmethod, ABC

from numpy import newaxis as na

import numpy as np


class WakeModel(ABC):
    """
    Base class for wake models
    Make a subclass and implement calc_deficit and calc_effective_WS
    Implementations of linear and squared sum available through inherritance

    Prefixs:
    i: turbine
    j: downstream points/turbines
    k: wind speed
    l: wind direction
    m: turbine and wind direction (il.flatten())
    n: from_turbine, to_turbine and wind direction (iil.flatten())

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

    def __init__(self, windTurbines):
        self.windTurbines = windTurbines

    def calc_wake(self, WS_ilk, TI_ilk, dw_iil, cw_iil, dh_iil, dw_order_indices_l, types_i):
        I, L = dw_iil.shape[1:]
        i1, i2, _ = np.where((np.abs(dw_iil) + np.abs(cw_iil) + np.eye(I)[:, :, na]) == 0)
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
        cw_n = cw_iil.flatten()
        dh_n = dh_iil.flatten()
        power_ilk = np.zeros((I, L, K))
        ct_ilk = np.zeros((I, L, K))
        types_i = np.asarray(types_i)
        D_i = self.windTurbines.diameter(types_i)
        H_i = self.windTurbines.hub_height(types_i)
        i_wd_l = np.arange(L)

        for j in range(I):
            i_wt_l = dw_order_indices_l[:, j]
            m = i_wt_l * L + i_wd_l  # current wt (j'th most upstream wts for all wdirs)
#             n_uw = np.array([indices[dw_order_indices_l[l, :j], i, l] for i, l in zip(i_wt_l, i_wd_l)]).T
#
#             n_dw = np.array([indices[i, dw_order_indices_l[l, j + 1:], l] for i, l in zip(i_wt_l, i_wd_l)]).T

            n_uw = np.array([indices[uwi, i, l] for uwi, i, l in zip(dw_order_indices_l[:, :j], i_wt_l, i_wd_l)]).T
            n_dw = np.array([indices[i, dwi, l] for dwi, i, l in zip(dw_order_indices_l[:, j + 1:], i_wt_l, i_wd_l)]).T

            WS_eff_lk = self.calc_effective_WS(WS_mk[m], deficit_nk[n_uw])
            WS_eff_mk[m] = WS_eff_lk
            ct_lk, power_lk = self.windTurbines.ct_power(WS_eff_lk, types_i[i_wt_l])

            power_ilk[i_wt_l, i_wd_l] = power_lk
            ct_ilk[i_wt_l, i_wd_l, :] = ct_lk
            if j < I - 1:
                arg_funcs = {'WS_lk': lambda: WS_mk[m],
                             'WS_eff_lk': lambda: WS_eff_mk[m],
                             'D_src_l': lambda: D_i[i_wt_l],
                             'D_dst_jl': lambda: D_i[dw_order_indices_l[:, j + 1:]].T,
                             'dw_jl': lambda: dw_n[n_dw],
                             'cw_jl': lambda: np.sqrt(cw_n[n_dw]**2 + dh_n[n_dw]**2),
                             'hcw_jl': lambda: cw_n[n_dw],
                             'dh_jl': lambda: dh_n[n_dw],
                             'h_l': lambda: H_i[i_wt_l],
                             'ct_lk': lambda: ct_lk}
                args = {k: arg_funcs[k]() for k in self.args4deficit}
                deficit_nk[n_dw] = self.calc_deficit(**args)
#                                                      WS_mk[m], D_i[i_wt_l],
#                                                      D_i[dw_order_indices_l[:, j + 1:]].T,
#                                                      dw_n[n_dw],
#                                                      cw_n[n_dw],
#                                                      ct_lk)
        WS_eff_ilk = WS_eff_mk.reshape((I, L, K))
        return WS_eff_ilk, TI_ilk, power_ilk, ct_ilk

    def wake_map(self, WS_ilk, WS_eff_ilk, dw_ijl, cw_ijl, dh_ijl, ct_ilk, types_i, WS_jlk):
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
                             'cw_jl': lambda: np.sqrt(cw_ijl[i, :, l][m]**2 + dh_ijl[i, :, l][m]**2)[:, na],
                             'hcw_jl': lambda: cw_ijl[i, :, l][m][:, na],
                             'dh_jl': lambda: dh_ijl[i, :, l][m][:, na],
                             'h_l': lambda: H_i[i][na],
                             'ct_lk': lambda: ct_ilk[i, l][na]}

                args = {k: arg_funcs[k]() for k in self.args4deficit}

                deficit_jlk[:, l][m] = self.calc_deficit(**args)[:, 0]

            deficit_ijlk.append(deficit_jlk)
        deficit_ijlk = np.array(deficit_ijlk)
        return self.calc_effective_WS(WS_jlk, deficit_ijlk)

    @abstractmethod
    def calc_deficit(self, WS_lk, D_src_l, D_dst_jl, dw_jl, cw_jl, ct_lk):
        """
        ct or a???
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

        class MyWakeModel(WakeModel, SquaredSum):
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
