from numpy import newaxis as na
from py_wake import np
from py_wake.deflection_models import DeflectionModel
from py_wake.utils.gradients import hypot
from py_wake.utils import gradients


class JimenezWakeDeflection(DeflectionModel):
    """Implemented according to
    Jiménez, Á., Crespo, A. and Migoya, E. (2010), Application of a LES technique to characterize
    the wake deflection of a wind turbine in yaw. Wind Energ., 13: 559-572. doi:10.1002/we.380
    """

    def __init__(self, N=20, beta=.1):
        self.beta = beta
        self.N = N

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ['hcw_ijlk', 'dh_ijlk']}

    def calc_deflection(self, dw_ijlk, hcw_ijlk, dh_ijlk, D_src_il, yaw_ilk, tilt_ilk, ct_ilk, **kwargs):
        dw_lst = (np.logspace(0, 1.1, self.N) - 1) / (10**1.1 - 1)
        dw_ijxlk = dw_ijlk[:, :, na] * dw_lst[na, na, :, na, na]
        theta_yaw_ilk, theta_tilt_ilk = gradients.deg2rad(yaw_ilk), gradients.deg2rad(-tilt_ilk)
        theta_ilk = hypot(theta_yaw_ilk, theta_tilt_ilk)
        theta_deflection_ilk = gradients.arctan2(theta_tilt_ilk, theta_yaw_ilk)
        denominator_ilk = np.cos(theta_ilk)**2 * np.sin(theta_ilk) * (ct_ilk / 2)
        nominator_ijxlk = (1 + (self.beta / D_src_il)[:, na, na, :, na] * np.maximum(dw_ijxlk, 0))**2
        alpha = denominator_ilk[:, na, na] / nominator_ijxlk
        deflection_ijlk = gradients.trapz(np.sin(alpha), dw_ijxlk, axis=2)
        self.hcw_ijlk = hcw_ijlk + deflection_ijlk * np.cos(theta_deflection_ilk[:, na])
        self.dh_ijlk = dh_ijlk + deflection_ijlk * np.sin(theta_deflection_ilk[:, na])
        return dw_ijlk, self.hcw_ijlk, self.dh_ijlk


def main():
    if __name__ == '__main__':
        from py_wake import Fuga
        from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
        site = IEA37Site(16)
        x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
        windTurbines = IEA37_WindTurbines()
        from py_wake.tests.test_files import tfp
        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'
        noj = Fuga(path, site, windTurbines, deflectionModel=JimenezWakeDeflection())
        yaw = [-30, 30, 0]
        noj(x, y, yaw=yaw, tilt=0, wd=270, ws=10).flow_map().plot_wake_map()
        import matplotlib.pyplot as plt
        plt.show()


main()
