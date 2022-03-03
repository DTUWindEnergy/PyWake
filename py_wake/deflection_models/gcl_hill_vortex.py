from numpy import newaxis as na
import numpy as np
from py_wake.deflection_models import DeflectionModel
from py_wake.utils.gradients import hypot
from py_wake.utils import gradients


class GCLHillDeflection(DeflectionModel):
    """Deflection based on Hill's ring vortex theory

    Implemented according to
    Larsen, G. C., Ott, S., Liew, J., van der Laan, M. P., Simon, E., R.Thorsen, G., & Jacobs, P. (2020).
    Yaw induced wake deflection - a full-scale validation study.
    Journal of Physics - Conference Series, 1618, [062047].
    https://doi.org/10.1088/1742-6596/1618/6/062047
    """

    def __init__(self, N=20, wake_deficitModel=None):
        """
        Parameters
        ----------
        N : int, optional
            Number of logarithmic distributed downstream points included in the numerical integration
        wake_deficitModel : WakeDeficitModel, optional
            Wake deficit model used to calculate the center wake deficit needed by this model.
            If None, the windFarmModel.wake_deficitModel is used
        """
        self._wake_deficitModel = wake_deficitModel
        self.N = N

    @property
    def args4deflection(self):
        return set(['D_src_il', 'yaw_ilk', 'ct_ilk', 'tilt_ilk']) | \
            set(self.wake_deficitModel.args4deficit) - {'dw_ijlk', 'hcw_ijlk', 'cw_ijlk', 'dh_ijlk'}

    @property
    def wake_deficitModel(self):
        return self._wake_deficitModel or self.windFarmModel.wake_deficitModel

    def calc_deflection(self, WS_eff_ilk, dw_ijl, hcw_ijl, dh_ijl, yaw_ilk, tilt_ilk, **kwargs):

        dw_lst = (np.logspace(0, 1.1, self.N) - 1) / (10**1.1 - 1)
        dw_ijlkx = dw_ijl[:, :, :, na, na] * dw_lst[na, na, na, na, :]
        z = np.zeros_like(dw_ijlkx)
        U_w_ijlx = self.wake_deficitModel.calc_deficit(
            WS_eff_ilk=(WS_eff_ilk * np.cos(np.deg2rad(yaw_ilk)))[..., na],
            dw_ijlk=dw_ijlkx, hcw_ijlk=z, cw_ijlk=z, dh_ijlk=z,
            tilt_ilk=tilt_ilk[..., na],
            **{k: v[..., na] for k, v in kwargs.items()})

        theta_yaw_ilk, theta_tilt_ilk = np.deg2rad(yaw_ilk), np.deg2rad(-tilt_ilk)
        theta_ilk = hypot(theta_yaw_ilk, theta_tilt_ilk)
        theta_deflection_ilk = np.arctan2(theta_tilt_ilk, theta_yaw_ilk)

        U_d_ijlkx = -0.4 * U_w_ijlx * np.sin(theta_ilk)[:, na, :, :, na]
        U_a_ijlkx = WS_eff_ilk[:, na, :, :, na] - 0.4 * U_w_ijlx * np.cos(theta_ilk)[:, na, :, :, na]

        deflection_ijlk = gradients.trapz(U_d_ijlkx / U_a_ijlkx, dw_ijlkx, axis=4)
        self.hcw_ijlk = hcw_ijl[..., na] - deflection_ijlk * np.cos(theta_deflection_ilk[:, na])
        self.dh_ijlk = dh_ijl[..., na] + deflection_ijlk * np.sin(theta_deflection_ilk[:, na])
        return dw_ijl[..., na], self.hcw_ijlk, self.dh_ijlk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.deficit_models.gaussian import ZongGaussian
        from py_wake.site.xrsite import UniformSite
        from py_wake.examples.data.hornsrev1 import V80
        from py_wake.flow_map import XYGrid
        from py_wake.turbulence_models.crespo import CrespoHernandez
        from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit

        site = UniformSite(p_wd=[1], ti=0.06)
        x, y = [0], [0]  # site.initial_position[:2].T

        wt = V80()
        D = wt.diameter()
        wfm = ZongGaussian(site, wt, deflectionModel=GCLHillDeflection(),
                           turbulenceModel=CrespoHernandez())
        wfm2 = ZongGaussian(site, wt,
                            deflectionModel=GCLHillDeflection(wake_deficitModel=BastankhahGaussianDeficit()),
                            turbulenceModel=CrespoHernandez())

        ws = 10
        yaw = 30
        plt.figure(figsize=(12, 3))
        grid = XYGrid(x=np.linspace(-2 * D, D * 10, 100), y=np.linspace(-1.5 * D, 1.5 * D, 100))
        fm = wfm(x, y, yaw=yaw, wd=270, ws=ws).flow_map(grid)
        fm.plot_wake_map(normalize_with=D)
        center_line = fm.min_WS_eff()
        plt.title(f'{wfm}, {yaw}')
        plt.plot(center_line.x / D, center_line / D, label='Deflection centerline with ZongGaussianDeficit')

        fm = wfm2(x, y, yaw=yaw, wd=270, ws=ws).flow_map(grid)
        center_line = fm.min_WS_eff()
        plt.plot(center_line.x / D, center_line / D, label='Deflection centerline with BastankhahGaussianDeficit')

        plt.grid()
        plt.legend()
        plt.show()


main()
