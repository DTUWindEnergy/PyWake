from numpy import newaxis as na
from py_wake import np
from py_wake.deflection_models.deflection_model import DeflectionIntegrator


class JimenezWakeDeflection(DeflectionIntegrator):
    """Implemented according to
    Jiménez, Á., Crespo, A. and Migoya, E. (2010), Application of a LES technique to characterize
    the wake deflection of a wind turbine in yaw. Wind Energ., 13: 559-572. doi:10.1002/we.380
    """

    def __init__(self, N=20, beta=.1):
        DeflectionIntegrator.__init__(self, N)
        self.beta = beta

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ['hcw_ijlk', 'dh_ijlk']}

    def get_deflection_rate(self, theta_ilk, ct_ilk, D_src_il, dw_ijlkx, **kwargs):
        denominator_ilk = np.cos(theta_ilk)**2 * np.sin(theta_ilk) * (ct_ilk / 2)
        nominator_ijlkx = (1 + (self.beta / D_src_il)[:, na, :, na, na] * np.maximum(dw_ijlkx, 0))**2
        alpha_ijlkx = denominator_ilk[:, na, :, :, na] / nominator_ijlkx
        return np.tan(alpha_ijlkx)


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
