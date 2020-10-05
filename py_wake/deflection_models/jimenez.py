from numpy import newaxis as na
import numpy as np
from py_wake.deflection_models import DeflectionModel


class JimenezWakeDeflection(DeflectionModel):
    """Implemented according to
    Jiménez, Á., Crespo, A. and Migoya, E. (2010), Application of a LES technique to characterize
    the wake deflection of a wind turbine in yaw. Wind Energ., 13: 559-572. doi:10.1002/we.380
    """

    args4deflection = ['D_src_il', 'yaw_ilk', 'ct_ilk']

    def __init__(self, N=20, beta=.1):
        self.beta = beta
        self.N = N

    def calc_deflection(self, dw_ijl, hcw_ijl, D_src_il, yaw_ilk, ct_ilk, **kwargs):
        dw_lst = (np.logspace(0, 1.1, self.N) - 1) / (10**1.1 - 1)
        dw_ijxl = dw_ijl[:, :, na] * dw_lst[na, na, :, na]

        denominator_ilk = np.cos(yaw_ilk)**2 * np.sin(yaw_ilk) * (ct_ilk / 2)
        nominator_ijxl = (1 + (self.beta / D_src_il)[:, na, na, :] * dw_ijxl)**2
        alpha = denominator_ilk[:, na, na] / nominator_ijxl[..., na]
        deflection_ijlk = np.trapz(np.sin(alpha), dw_ijxl[..., na], axis=2)
        return dw_ijl[..., na], hcw_ijl[..., na] + deflection_ijlk


def main():
    if __name__ == '__main__':
        from py_wake import Fuga
        from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
        site = IEA37Site(16)
        x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
        windTurbines = IEA37_WindTurbines()
        from py_wake.tests.test_files import tfp
        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
        noj = Fuga(path, site, windTurbines, deflectionModel=JimenezWakeDeflection())
        yaw_ilk = np.zeros((len(x), 1, 1))
        yaw_ilk[:, 0, 0] = [-30, 30, 0]
        noj(x, y, yaw_ilk=yaw_ilk, wd=270, ws=10).flow_map().plot_wake_map()
        import matplotlib.pyplot as plt
        plt.show()

        pass


main()
