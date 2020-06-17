from numpy import newaxis as na

import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.utils.area_overlapping_factor import AreaOverlappingFactor


class NOJDeficit(DeficitModel, AreaOverlappingFactor):
    args4deficit = ['WS_ilk', 'D_src_il', 'D_dst_ijl', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, k=.1):
        AreaOverlappingFactor.__init__(self, k)

    def _calc_layout_terms(self, WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, **_):
        R_src_il = D_src_il / 2
        term_denominator_ijlk = (1 + self.k * dw_ijlk / R_src_il[:, na, :, na])**2
        term_denominator_ijlk += (term_denominator_ijlk == 0)
        A_ol_factor_ijlk = self.overlapping_area_factor(self.wake_radius(D_src_il, dw_ijlk),
                                                        dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl)

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijlk > 0) * (A_ol_factor_ijlk / term_denominator_ijlk)

    def calc_deficit(self, WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk)
        ct_ilk = np.minimum(ct_ilk, 1)   # treat ct_ilk for np.sqrt()
        term_numerator_ilk = (1 - np.sqrt(1 - ct_ilk))
        return term_numerator_ilk[:, na] * self.layout_factor_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        wake_radius_ijlk = (self.k * dw_ijlk + D_src_il[:, na, :, na] / 2)
        return wake_radius_ijlk


class NOJ(PropagateDownwind):
    def __init__(self, site, windTurbines, k=.1, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NOJDeficit(k),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wf_model = NOJ(site, windTurbines)
        print(wf_model)

        # run wind farm simulation
        sim_res = wf_model(x, y)

        # calculate AEP
        aep = sim_res.aep()
        print(aep)
        # plot wake map
        flow_map = sim_res.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map()
        flow_map.plot_windturbines()
        plt.title('AEP: %.2f GWh' % aep.sum())
        plt.show()


main()
