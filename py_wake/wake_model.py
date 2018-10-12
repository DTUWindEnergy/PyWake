import numpy as np
from abc import abstractmethod, ABC
from numpy import newaxis as na


class WakeModel(ABC):
    def __init__(self, windTurbines):
        self.windTurbines = windTurbines

    def calc_wake(self, WS_ilk, TI_ilk, dw_iil, cw_iil, dw_order_indices_l, types_i):
        I, L = dw_iil.shape[1:]
        K = WS_ilk.shape[2]
        deficit_nk = np.zeros((I * I * L, K))
        # deficit_ijlk = deficit_nk.reshape((I, I, L, K))  # from i to j

        indices = np.arange(I * I * L).reshape((I, I, L))
        WS_mk = WS_ilk.astype(np.float).reshape((I * L, K))
        WS_eff_mk = WS_mk.copy()
        dw_n = dw_iil.flatten()
        cw_n = cw_iil.flatten()
        power_ilk = np.zeros((I, L, K))
        ct_ilk = np.zeros((I, L, K))
        types_i = np.asarray(types_i)
        D_i = self.windTurbines.diameter(types_i)
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
                deficit_nk[n_dw] = self.calc_deficit(WS_mk[m], D_i[i_wt_l],
                                                     D_i[dw_order_indices_l[:, j + 1:]].T,
                                                     dw_n[n_dw],
                                                     cw_n[n_dw],
                                                     ct_lk)
        WS_eff_ilk = WS_eff_mk.reshape((I, L, K))
        return WS_eff_ilk, TI_ilk, power_ilk, ct_ilk

    def wake_map(self, WS_ilk, dw_ijl, cw_ijl, ct_ilk, types_i, WS_jlk):
        D_i = self.windTurbines.diameter(types_i)
        I, J, L = dw_ijl.shape
        K = WS_ilk.shape[2]

        deficit_ijlk = []
        for i in range(I):
            deficit_jlk = np.zeros((J, L, K))

            for l in range(L):
                m = dw_ijl[i, :, l] > 0
                deficit_jlk[:, l][m] = self.calc_deficit(
                    WS_ilk[i, l][na], D_i[i][na], None, dw_ijl[i, :, l][m][:, na], cw_ijl[i, :, l][m][:, na], ct_ilk[i, l][na])[:, 0]
            deficit_ijlk.append(deficit_jlk)

        return self.calc_effective_WS(WS_jlk, np.array(deficit_ijlk))

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
        from py_wake.site._site import UniformSite
        from py_wake.wind_turbines.iea37_wind_turbine import IEA37_WindTurbines
        from py_wake.aep._aep import AEP

        class MyWakeModel(WakeModel, SquaredSum):
            def calc_deficit(self, WS_lk, D_src_l, D_dst_jl, dw_jl, cw_jl, ct_lk):
                # 10% deficit downstream
                return (WS_lk * .1)[na] * (dw_jl > 0)[:, :, na]

        _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        n_wt = 16
        x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)

        site = UniformSite(freq, ti=0.75)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        import matplotlib.pyplot as plt
        x_j = np.linspace(-1500, 1500, 500)
        y_j = np.linspace(-1500, 1500, 300)

        wake_model = MyWakeModel(windTurbines)
        aep = AEP(site, windTurbines, wake_model)
        X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0, 30], ws=[8, 9, 10])
        plt.figure()
        c = plt.contourf(X, Y, Z, np.arange(2, 9.1, .01))
        plt.colorbar(c)

        plt.plot(x, y, '2k')
        for i, (x_, y_) in enumerate(zip(x, y)):
            plt.annotate(i, (x_, y_))
        plt.axis('equal')

        plt.show()


main()
