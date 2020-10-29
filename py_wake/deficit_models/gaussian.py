from numpy import newaxis as na
from scipy.special import erf
import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models.deficit_model import ConvectionDeficitModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


class BastankhahGaussianDeficit(ConvectionDeficitModel):
    """Implemented according to
    Bastankhah M and Porté-Agel F.
    A new analytical model for wind-turbine wakes.
    J. Renew. Energy. 2014;70:116-23.
    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijlk', 'cw_ijlk', 'D_src_il', 'ct_ilk']

    def __init__(self, k=0.0324555, use_effective_ws=False):
        self._k = k
        self.use_effective_ws = use_effective_ws

    def k_ilk(self, **_):
        return np.reshape(self._k, (1, 1, 1))

    def _calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        WS_ref_ilk = (WS_ilk, WS_eff_ilk)[self.use_effective_ws]
        sqrt1ct_ilk = np.sqrt(1 - ct_ilk)
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk
        sigma_sqr_ijlk = (self.k_ilk(**kwargs)[:, na] * dw_ijlk /
                          D_src_il[:, na, :, na] + .2 * np.sqrt(beta_ilk)[:, na])**2
        # maximum added to avoid sqrt of negative number
        radical_ijlk = np.maximum(0, (1. - ct_ilk[:, na] / (8. * sigma_sqr_ijlk)))
        deficit_centre_ijlk = WS_ref_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * (dw_ijlk > 0)
        # make sigma dimensional
        sigma_sqr_ijlk *= D_src_il[:, na, :, na]**2

        return WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk

    def calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        _, sigma_sqr_ijlk, deficit_centre_ijlk, _ = self._calc_deficit(
            WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs)

        # term inside exp()
        exponent_ijlk = -1 / (2 * sigma_sqr_ijlk) * cw_ijlk**2

        # Point deficit
        deficit_ijlk = deficit_centre_ijlk * np.exp(exponent_ijlk)
        return deficit_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        sqrt1ct_ilk = np.sqrt(1 - ct_ilk)
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk
        sigma_ijlk = self.k_ilk(**kwargs)[:, na] * dw_ijlk / D_src_il[:, na, :, na] + .2 * np.sqrt(beta_ilk)[:, na]
        return 2 * sigma_ijlk * D_src_il[:, na, :, na]

    def calc_deficit_convection(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):

        WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk = self._calc_deficit(
            WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs)
        # Convection velocity
        uc_ijlk = WS_ref_ilk[:, na] * 0.5 * (1. + np.sqrt(radical_ijlk))

        return deficit_centre_ijlk, uc_ijlk, sigma_sqr_ijlk


class BastankhahGaussian(PropagateDownwind):
    """Predefined wind farm model"""

    def __init__(self, site, windTurbines, k=0.0324555,
                 rotorAvgModel=RotorCenter(), superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float
            Wake expansion factor
        rotorAvgModel : RotorAvgModel
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            Defaults to RotorCenter that uses the rotor center wind speed (i.e. one point) only
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=BastankhahGaussianDeficit(k=k),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel,
                                   groundModel=groundModel)


class NiayifarGaussianDeficit(BastankhahGaussianDeficit):
    """
    Implemented according to:
        Amin Niayifar and Fernando Porté-Agel
        Analytical Modeling of Wind Farms: A New Approach for Power Prediction
        Energies 2016, 9, 741; doi:10.3390/en9090741

    Features:
        - Wake expansion function of local turbulence intensity

    Description:
        The expansion rate 'k' varies linearly with local turbluence
        intensity: k = a1 I + a2. The default constants are set
        according to publications by Porte-Agel's group, which are based
        on LES simulations. Lidar field measurements by Fuertes et al. (2018)
        indicate that a = [0.35, 0.0] is also a valid selection.

    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_eff_ilk']

    def __init__(self, a=[0.38, 4e-3], use_effective_ws=False):
        self.a = a
        self.use_effective_ws = use_effective_ws

    def k_ilk(self, TI_eff_ilk, **_):
        k_ilk = self.a[0] * TI_eff_ilk + self.a[1]
        return k_ilk


class NiayifarGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
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
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=NiayifarGaussianDeficit(a=a),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel, groundModel=groundModel)


class IEA37SimpleBastankhahGaussianDeficit(BastankhahGaussianDeficit):
    """Implemented according to
    https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/iea37-wakemodel.pdf

    Equivalent to BastankhahGaussian for beta=1/sqrt(8) ~ ct=0.9637188
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'WS_eff_ilk']
    args4update = ['ct_ilk']

    def __init__(self, ):
        BastankhahGaussianDeficit.__init__(self, k=0.0324555)

    def _calc_layout_terms(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, **kwargs):
        eps = 1e-10
        sigma_ijlk = self.k_ilk(**kwargs) * dw_ijlk * (dw_ijlk > eps) + (D_src_il / np.sqrt(8.))[:, na, :, na]
        self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijlk > eps) * \
            np.exp(-0.5 * (cw_ijlk / sigma_ijlk)**2)
        self.denominator_ijlk = 8. * (sigma_ijlk / D_src_il[:, na, :, na])**2

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, D_src_il, dw_ijlk, cw_ijlk)
        ct_factor_ijlk = (1. - ct_ilk[:, na] / self.denominator_ijlk)
        return self.layout_factor_ijlk * (1 - np.sqrt(ct_factor_ijlk))  # deficit_ijlk


class IEA37SimpleBastankhahGaussian(PropagateDownwind):
    """Predefined wind farm model"""

    def __init__(self, site, windTurbines,
                 rotorAvgModel=RotorCenter(), superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            Defaults to RotorCenter that uses the rotor center wind speed (i.e. one point) only
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel, groundModel=groundModel)


class ZongGaussianDeficit(NiayifarGaussianDeficit):
    """
    Implemented according to:
        Haohua Zong and Fernando Porté-Agel
        A momentum-conserving wake superposition method for
        wind farm power prediction
        J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77

    Features:
        - Wake expansion function of local turbulence intensity
        - New wake width expression following the approach by
          Shapiro et al. (2018)

    Description:
        Extension of the Niayifar et al. (2016) implementation with Shapirio
        wake width expression, which uses the near-wake length estimation by
        Vermeulen (1980).

    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_eff_ilk']

    def nw_length(self, ct_ilk, D_src_il, TI_eff_ilk, lam=7.5, B=3):
        """
        Implementation of Vermeulen (1980) near-wake length according to:
            Amin Niayifar and Fernando Porté-Agel
            Analytical Modeling of Wind Farms: A New Approach for Power Prediction
            Energies 2016, 9, 741; doi:10.3390/en9090741

        """
        # major factors
        ct_ilk = np.minimum(ct_ilk, 0.999)
        m_ilk = 1 / np.sqrt(1 - ct_ilk)
        r0_ilk = D_src_il[:, :, na] / 2 * np.sqrt((m_ilk + 1) / 2)
        # wake expansion
        dr_alpha_ilk = 2.5 * TI_eff_ilk + 5e-3
        dr_m_ilk = (1 - m_ilk) * np.sqrt(1.49 + m_ilk) / (9.76 * (1 + m_ilk))
        dr_lambda = 1.2e-2 * B * lam
        # total expansion rate
        drdx_ilk = np.sqrt(dr_alpha_ilk**2 + dr_m_ilk**2 + dr_lambda**2)
        # fitted factor
        n_ilk = np.sqrt(0.214 + 0.144 * m_ilk) * (1 - np.sqrt(0.134 + 0.124 * m_ilk)) / \
            ((1 - np.sqrt(0.214 + 0.144 * m_ilk)) * (np.sqrt(0.134 + 0.124 * m_ilk)))
        # Near-wake length
        xnw_ilk = n_ilk * r0_ilk / drdx_ilk

        return xnw_ilk

    def ct_func(self, ct_ilk, dw_ijlk, D_src_il):
        # Slow growth of ct
        ctx_ijlk = ct_ilk[:, na] * (1 + erf(dw_ijlk / D_src_il[:, na, :, na])) / 2.
        return ctx_ijlk

    def _calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, TI_eff_ilk, **_):
        WS_ref_ilk = (WS_ilk, WS_eff_ilk)[self.use_effective_ws]

        # near-wake length
        xnw_ilk = self.nw_length(ct_ilk, D_src_il, TI_eff_ilk)
        # wake growth rate
        k_ilk = self.k_ilk(TI_eff_ilk)
        # wake width
        sigma_sqr_ijlk = ((0.35 + k_ilk[:, na, :, :] *
                           np.log(1 + np.exp((dw_ijlk - xnw_ilk[:, na, :, :]) / D_src_il[:, na, :, na]))) *
                          D_src_il[:, na, :, na])**2
        ctx_ijlk = self.ct_func(ct_ilk, dw_ijlk, D_src_il)

        radical_ijlk = np.maximum(0, (1. - ctx_ijlk * D_src_il[:, na, :, na]**2 / (8. * sigma_sqr_ijlk)))

        # Centreline deficit
        deficit_centre_ijlk = WS_ref_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * (dw_ijlk > 0)

        return WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, ct_ilk, TI_eff_ilk, **_):
        # near-wake length
        xnw_ilk = self.nw_length(ct_ilk, D_src_il, TI_eff_ilk)
        # wake growth rate
        k_ilk = self.a[0] * TI_eff_ilk + self.a[1]
        # wake width
        sigma_ijlk = ((0.35 + k_ilk[:, na, :, :] *
                       np.log(1 + np.exp((dw_ijlk - xnw_ilk[:, na, :, :]) / D_src_il[:, na, :, na]))) *
                      D_src_il[:, na, :, na])
        return 2 * sigma_ijlk


class ZongGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
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
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=ZongGaussianDeficit(a=a),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel, groundModel=groundModel)


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
        plt.title('AEP: %.2f GWh' % aep.sum())
        plt.show()


main()
