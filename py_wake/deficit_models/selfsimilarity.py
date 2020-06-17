import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models.no_wake import NoWakeDeficit


class SelfSimilarityDeficit(DeficitModel):
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, lambda_=0.587, eta=1.32):
        self.lambda_ = lambda_
        self.eta = eta

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """References:
            [1] N. Troldborg, A.R. Meyer Fortsing, Wind Energy, 2016
        """
        eps = 1e-10
        R_ijl = (D_src_il / 2)[:, na]
        x = -dw_ijlk / R_ijl[..., na]
        r12_ijlk = np.sqrt(self.lambda_ * (self.eta + x ** 2))   # Eq. (13) from [1]
        radial_factor_ijlk = (1 / np.cosh(np.sqrt(2) * cw_ijlk /
                                          (R_ijl[..., na] * r12_ijlk))) ** (8 / 9)  # Eq. (6) from [1]
        a0_ilk = 1 / 2 * (1 - np.sqrt(1 - 1.1 * ct_ilk))  # Eq. (6) from [1]
        axial_factor_ijlk = a0_ilk[:, na] * (1 - x / np.sqrt(x**2 + 1))          # Eq. (7) from [1]
        return WS_ilk[:, na] * (dw_ijlk < -eps) * axial_factor_ijlk * radial_factor_ijlk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.examples.data import hornsrev1
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative

        site = Hornsrev1Site()
        windTurbines = hornsrev1.HornsrevV80()
        ws = 10
        D = 80
        R = D / 2
        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[.8]]])
        ss = SelfSimilarityDeficit()

        x, y = -np.arange(200), np.array([0])
        deficit = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                  dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                  cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        plt.title('Fig 11 from [1]')
        plt.xlabel('x/R')
        plt.ylabel('a')
        plt.plot(x / R, deficit[0, :, 0, 0] / ws)

        plt.figure()
        x, y = np.array([-2 * R]), np.arange(200)
        deficit = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                  dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                  cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        plt.title('Fig 10 from [1]')
        r12 = np.sqrt(ss.lambda_ * (ss.eta + (x / R) ** 2))   # Eq. (13) from [1]
        print(x, r12)
        plt.xlabel('y/R12 (epsilon)')
        plt.ylabel('f')
        plt.plot((y / R) / r12, deficit[0, :, 0, 0] / deficit[0, 0, 0, 0])

        plt.figure()
        noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=ss)
        flow_map = noj_ss(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        flow_map.plot_wake_map()
        flow_map.plot_windturbines()

        plt.show()


main()
