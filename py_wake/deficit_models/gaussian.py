from numpy import newaxis as na

import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


class BastankhahGaussianDeficit(DeficitModel):
    """Implemented according to
    Bastankhah M and Porté-Agel F.
    A new analytical model for wind-turbine wakes.
    J. Renew. Energy. 2014;70:116-23.
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, k=0.0324555):
        self.k = k

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        sqrt1ct_ilk = np.sqrt(1 - ct_ilk)
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk
        sigma_sqr_ijlk = (self.k * dw_ijlk / D_src_il[:, na, :, na] + .2 * np.sqrt(beta_ilk)[:, na])**2
        exponent_ijlk = -1 / (2 * sigma_sqr_ijlk) * (cw_ijlk**2 / D_src_il[:, na, :, na]**2)
        # maximum added to avoid sqrt of negative number
        radical_ijlk = np.maximum(0, (1. - ct_ilk[:, na] / (8. * sigma_sqr_ijlk)))
        deficit_ijlk = (WS_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * np.exp(exponent_ijlk)) * (dw_ijlk > 0)
        return deficit_ijlk


class IEA37SimpleBastankhahGaussianDeficit(DeficitModel):
    """Implemented according to
    https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/iea37-wakemodel.pdf

    Equivalent to BastankhahGaussian for beta=1/sqrt(8) ~ ct=0.9637188
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']
    args4update = ['ct_ilk']

    def __init__(self, ):
        self.k = 0.0324555

    def _calc_layout_terms(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, **_):
        sigma_ijlk = self.k * dw_ijlk * (dw_ijlk > 0) + (D_src_il / np.sqrt(8.))[:, na, :, na]
        self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijlk > 0) * \
            np.exp(-0.5 * (cw_ijlk / sigma_ijlk)**2)
        self.denominator_ijlk = 8. * (sigma_ijlk / D_src_il[:, na, :, na])**2

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, D_src_il, dw_ijlk, cw_ijlk)
        ct_factor_ijlk = (1. - ct_ilk[:, na] / self.denominator_ijlk)
        return self.layout_factor_ijlk * (1 - np.sqrt(ct_factor_ijlk))  # deficit_ijlk


class BastankhahGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, k=0.0324555, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=BastankhahGaussianDeficit(k=k),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class IEA37SimpleBastankhahGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
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

        wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)

        print(wf_model)

        # run wind farm simulation
        sim_res = wf_model(x, y)

        # calculate AEP
        aep = sim_res.aep()

        # plot wake map
        flow_map = sim_res.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map()
        flow_map.plot_windturbines()
        plt.title('AEP: %.2f GWh' % aep)
        plt.show()


main()
