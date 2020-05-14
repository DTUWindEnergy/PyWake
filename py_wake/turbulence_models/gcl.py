
from py_wake.deficit_models.noj import AreaOverlappingFactor
import numpy as np
from numpy import newaxis as na
from py_wake.turbulence_models.turbulence_model import TurbulenceModel, SqrMaxSum


class GCLTurbulenceModel(SqrMaxSum, TurbulenceModel, AreaOverlappingFactor):
    args4addturb = ['D_src_il', 'dw_ijlk', 'ct_ilk', 'D_dst_ijl', 'cw_ijlk', 'wake_radius_ijlk']

    def __init__(self, k=.1):
        AreaOverlappingFactor.__init__(self, k)

    def calc_added_turbulence(self, dw_ijlk, D_src_il, ct_ilk, wake_radius_ijlk,
                              D_dst_ijl, cw_ijlk, **_):
        """ Calculate the added turbulence intensity at downstream distance
        x at the wake of a turbine.

        Vectorized version to account multiple downwind distances.

        Parameters
        ----------
        x: array:float
            Downwind distance [m]
        D: float
            Rotor diameter [m]
        Ct: float
            Thrust coefficient [-]

        Returns
        -------
        TI_add: float
            Added turbulence intensity [-]
        """
        dw_ijlk_gt0 = np.maximum(dw_ijlk, 1e-10)
        r = 0.29 * np.sqrt(1 - np.sqrt(1 - ct_ilk))[:, na] / \
            ((dw_ijlk_gt0 / D_src_il[:, na, :, na])**(1 / 3))
        area_overlap_ijlk = self.overlapping_area_factor(wake_radius_ijlk, dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl)
        return area_overlap_ijlk * r * (dw_ijlk > 0)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake import NOJ
        import matplotlib.pyplot as plt
        # setup site, turbines and wakemodel
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wf_model = NOJ(site, windTurbines, turbulenceModel=GCLTurbulenceModel())

        # calculate AEP
        sim_res = wf_model(x, y)
        print(sim_res.TI_eff_ilk.flatten())

        # plot wake mape
        aep = sim_res.aep()
        flow_map = sim_res.flow_map(wd=0, ws=9.8)
        flow_map.plot_ti_map()
        plt.title('AEP: %.2f GWh' % aep)
        plt.show()


main()
