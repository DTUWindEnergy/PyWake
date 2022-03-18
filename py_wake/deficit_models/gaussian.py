from numpy import newaxis as na
from py_wake.utils.gradients import erf
import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models.deficit_model import ConvectionDeficitModel
from py_wake.ground_models.ground_models import NoGround, Mirror
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.utils.gradients import cabs
from py_wake.utils import gradients


class BastankhahGaussianDeficit(ConvectionDeficitModel):
    """Implemented according to
    Bastankhah M and Porté-Agel F.
    A new analytical model for wind-turbine wakes.
    J. Renew. Energy. 2014;70:116-23.
    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijlk', 'cw_ijlk', 'D_src_il', 'ct_ilk']

    def __init__(self, k=0.0324555, ceps=.2, use_effective_ws=False, groundModel=NoGround()):
        DeficitModel.__init__(self, groundModel=groundModel)
        self._k = k
        self._ceps = ceps
        self.use_effective_ws = use_effective_ws
        ConvectionDeficitModel.__init__(self)

    def k_ilk(self, **kwargs):
        shape = np.ones_like(kwargs.get('WS_ilk', np.ones((1, 1, 1))).shape)
        return np.reshape(self._k, shape)

    def epsilon_ilk(self, ct_ilk, **_):
        # not valid for CT >= 1.
        sqrt1ct_ilk = np.sqrt(1 - np.minimum(0.999, ct_ilk))
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk

        return self._ceps * np.sqrt(beta_ilk)

    def sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        # dimensional wake expansion
        return self.k_ilk(**kwargs)[:, na] * dw_ijlk + \
            self.epsilon_ilk(ct_ilk)[:, na] * D_src_il[:, na, :, na]

    def _calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        WS_ref_ilk = (WS_ilk, WS_eff_ilk)[self.use_effective_ws]
        # dimensional wake expansion
        sigma_sqr_ijlk = (self.sigma_ijlk(D_src_il, dw_ijlk, ct_ilk, **kwargs))**2
        # maximum added to avoid sqrt of negative number
        radical_ijlk = np.maximum(0, (1. - ct_ilk[:, na] * D_src_il[:, na, :, na]**2 / (8. * sigma_sqr_ijlk)))
        deficit_centre_ijlk = WS_ref_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * (dw_ijlk > 0)

        return WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk

    def calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        _, sigma_sqr_ijlk, deficit_centre_ijlk, _ = self._calc_deficit(WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk,
                                                                       ct_ilk, **kwargs)

        # term inside exp()
        exponent_ijlk = -1 / (2 * sigma_sqr_ijlk) * cw_ijlk**2

        # Point deficit
        deficit_ijlk = deficit_centre_ijlk * np.exp(exponent_ijlk)
        return deficit_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        # according to Niayifar, the wake radius is twice sigma
        sigma_ijlk = self.sigma_ijlk(D_src_il, dw_ijlk, ct_ilk, **kwargs)
        return 2. * sigma_ijlk

    def calc_deficit_convection(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):

        WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk = self._calc_deficit(WS_ilk, WS_eff_ilk,
                                                                                           D_src_il, dw_ijlk, ct_ilk,
                                                                                           **kwargs)
        # Convection velocity
        uc_ijlk = WS_ref_ilk[:, na] * 0.5 * (1. + np.sqrt(radical_ijlk))

        return deficit_centre_ijlk, uc_ijlk, sigma_sqr_ijlk


class BastankhahGaussian(PropagateDownwind):
    """Predefined wind farm model"""

    def __init__(self, site, windTurbines, k=0.0324555, ceps=.2,
                 rotorAvgModel=RotorCenter(), superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=NoGround()):
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
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=BastankhahGaussianDeficit(k=k, ceps=ceps, groundModel=groundModel),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


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
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il',
                    'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_ilk', 'TI_eff_ilk']

    def __init__(self, a=[0.38, 4e-3], ceps=.2, use_effective_ws=False, use_effective_ti=True,
                 groundModel=NoGround()):
        DeficitModel.__init__(self, groundModel=groundModel)
        self._ceps = ceps
        self.a = a
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti

    def k_ilk(self, **kwargs):
        TI_ref_ilk = (kwargs['TI_ilk'], kwargs['TI_eff_ilk'])[self.use_effective_ti]
        k_ilk = self.a[0] * TI_ref_ilk + self.a[1]
        return k_ilk


class NiayifarGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], ceps=.2, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=NoGround()):
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
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=NiayifarGaussianDeficit(a=a, ceps=ceps, groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class IEA37SimpleBastankhahGaussianDeficit(BastankhahGaussianDeficit):
    """Implemented according to
    https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/iea37-wakemodel.pdf

    Equivalent to BastankhahGaussian for beta=1/sqrt(8) ~ ct=0.9637188
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'WS_eff_ilk']
    args4update = ['ct_ilk']

    def __init__(self, groundModel=NoGround()):
        DeficitModel.__init__(self, groundModel=groundModel)
        BastankhahGaussianDeficit.__init__(self, k=0.0324555)

    def _calc_layout_terms(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, **kwargs):
        eps = 1e-10
        sigma_ijlk = self.k_ilk(WS_ilk=WS_ilk) * dw_ijlk * (dw_ijlk > eps) + (D_src_il / np.sqrt(8.))[:, na, :, na]
        self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijlk > eps) * np.exp(-0.5 * (cw_ijlk / sigma_ijlk)**2)
        self.denominator_ijlk = 8. * (sigma_ijlk / D_src_il[:, na, :, na])**2

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, D_src_il, dw_ijlk, cw_ijlk)
        ct_factor_ijlk = (1. - ct_ilk[:, na] / self.denominator_ijlk)
        # deficit_ijlk
        return self.layout_factor_ijlk * (1 - np.sqrt(ct_factor_ijlk))


class IEA37SimpleBastankhahGaussian(PropagateDownwind):
    """Predefined wind farm model"""

    def __init__(self, site, windTurbines,
                 rotorAvgModel=RotorCenter(), superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=NoGround()):
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
                                   wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(groundModel=groundModel),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class ZongGaussianDeficit(NiayifarGaussianDeficit):
    """
    Implemented according to:
        Haohua Zong and Fernando Porté-Agel
        "A momentum-conserving wake superposition method for wind farm power prediction"
        J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77

    Features:
        - Wake expansion function of local turbulence intensity
        - New wake width expression following the approach by Shapiro et al. (2018)

    Description:
        Extension of the Niayifar et al. (2016) implementation with adapted
        Shapiro wake model components, namely a gradual growth of the thrust
        force and an expansion factor not falling below the rotor diameter.
        Shapiro modelled the pressure and thrust force as a combined momentum
        source, that are distributed in the streamwise direction with a Gaussian
        kernel with a certain characteristic length. As a result the induction
        changes following an error function. Zong chose to use a characteristic
        length of D/sqrt(2) and applies it directly to the thrust not the induction
        as Shapiro. This leads to the full thrust being active only 2D downstream of
        the turbine. Zong's wake width expression is inspired by Shapiro's, however
        the start of the linear wake expansion region (far-wake) was related to
        the near-wake length by Vermeulen (1980). The epsilon factor that in the
        original Gaussian model was taken to be a function of CT is now a constant
        as proposed by Bastankhah (2016), as the near-wake length now effectively
        dictates the origin of the far-wake.

    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_ilk', 'TI_eff_ilk']

    def __init__(self, a=[0.38, 4e-3], deltawD=1. / np.sqrt(2), eps_coeff=1. / np.sqrt(8.), lam=7.5, B=3,
                 use_effective_ws=False, use_effective_ti=True, groundModel=NoGround()):
        DeficitModel.__init__(self, groundModel=groundModel)
        self.a = a
        self.deltawD = deltawD
        # different from Zong, as he effectively took it from Bastankhah 2016, so here
        # we use the original definition for consistency. In Zong eps_coeff=0.35.
        self.eps_coeff = eps_coeff
        self.lam = lam
        self.B = B
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti

    def nw_length(self, ct_ilk, D_src_il, TI_eff_ilk, **_):
        """
        Implementation of Vermeulen (1980) near-wake length according to:
            Amin Niayifar and Fernando Porté-Agel
            Analytical Modeling of Wind Farms: A New Approach for Power Prediction
            Energies 2016, 9, 741; doi:10.3390/en9090741

        """
        # major factors
        lam, B = self.lam, self.B
        ct_ilk = np.minimum(ct_ilk, 0.999)
        m_ilk = 1. / np.sqrt(1. - ct_ilk)
        r0_ilk = D_src_il[:, :, na] / 2. * np.sqrt((m_ilk + 1.) / 2.)
        # wake expansion
        dr_alpha_ilk = 2.5 * TI_eff_ilk + 5e-3
        dr_m_ilk = (1. - m_ilk) * np.sqrt(1.49 + m_ilk) / (9.76 * (1. + m_ilk))
        dr_lambda = 1.2e-2 * B * lam
        # total expansion rate
        drdx_ilk = np.sqrt(dr_alpha_ilk**2 + dr_m_ilk**2 + dr_lambda**2)
        # fitted factor
        n_ilk = np.sqrt(0.214 + 0.144 * m_ilk) * (1. - np.sqrt(0.134 + 0.124 * m_ilk)) / \
            ((1. - np.sqrt(0.214 + 0.144 * m_ilk)) * (np.sqrt(0.134 + 0.124 * m_ilk)))
        # Near-wake length
        xnw_ilk = n_ilk * r0_ilk / drdx_ilk

        return xnw_ilk

    def ct_func(self, ct_ilk, dw_ijlk, D_src_il, **_):
        # Slow growth of deficit until 2D downstream. Note that here we are using the
        # formulation originally proposed by Shapiro, but with the factor implied by
        # the relationship Zong used.
        ctx_ijlk = ct_ilk[:, na] * (1. + erf(dw_ijlk / (np.sqrt(2.) *
                                                        self.deltawD * D_src_il[:, na, :, na]))) / 2.
        return ctx_ijlk

    def epsilon_ilk(self, ct_ilk, **_):
        return self.eps_coeff * np.ones_like(ct_ilk)

    def sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        TI_ref_ilk = (kwargs['TI_ilk'], kwargs['TI_eff_ilk'])[self.use_effective_ti]
        # near-wake length
        xnw_ilk = self.nw_length(ct_ilk, D_src_il, TI_ref_ilk)
        # wake growth rate
        k_ilk = self.k_ilk(**kwargs)
        # wake spreading
        # initial size is here a combination of epsilon and the near-wake (needed modification, to ensure
        # the intial wake width is identical to the original formulation. Zong just used a fixed value)
        # non-dimensional wake expansion
        # logaddexp(0,x) ~ log(1+exp(x)) without overflow
        sigmaD_ijlk = (self.epsilon_ilk(ct_ilk)[:, na] + k_ilk[:, na, :, :] *
                       gradients.logaddexp(0, (dw_ijlk - xnw_ilk[:, na, :, :]) / D_src_il[:, na, :, na]))

        return sigmaD_ijlk * D_src_il[:, na, :, na]

    def _calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        WS_ref_ilk = (WS_ilk, WS_eff_ilk)[self.use_effective_ws]

        # dimensional wake expansion rate
        sigma_sqr_ijlk = (self.sigma_ijlk(D_src_il, dw_ijlk, ct_ilk, **kwargs))**2
        ctx_ijlk = self.ct_func(ct_ilk, dw_ijlk, D_src_il)

        radical_ijlk = np.maximum(0, (1. - ctx_ijlk * D_src_il[:, na, :, na]**2 / (8. * sigma_sqr_ijlk)))

        # Centreline deficit
        deficit_centre_ijlk = WS_ref_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * (dw_ijlk > 0)

        return WS_ref_ilk, sigma_sqr_ijlk, deficit_centre_ijlk, radical_ijlk


class ZongGaussian(PropagateDownwind):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], deltawD=1. / np.sqrt(2), lam=7.5, B=3,
                 superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None, groundModel=NoGround()):
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
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=ZongGaussianDeficit(a=a, deltawD=deltawD, lam=lam, B=B, groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class CarbajofuertesGaussianDeficit(ZongGaussianDeficit):
    """
    Modified Zong version with Gaussian constants from:
        Fernando Carbajo Fuertes, Corey D. Markfor and Fernando Porté-Agel
        "Wind TurbineWake Characterization with Nacelle-MountedWind Lidars
        for Analytical Wake Model Validation"
        Remote Sens. 2018, 10, 668; doi:10.3390/rs10050668

    Features:
        - Empirical correlation for epsilon
        - New constants for wake expansion factor equation

    Description:
        Carbajo Fuertes et al. derived Gaussian wake model parameters from
        nacelle liadar measurements from a 2.5MW turbine and found a
        variation of epsilon with wake expansion, this in fact identical
        to the formulation by Zong, only that the near-wake length is fixed
        for Carbajo Fuertes at xth = 1.91 x/D. We took the relationships
        found by them and incorporated them into the Zong formulation.

    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_ilk', 'TI_eff_ilk']

    def __init__(self, a=[0.35, 0], deltawD=1. / np.sqrt(2), use_effective_ws=False, use_effective_ti=True,
                 groundModel=NoGround()):
        DeficitModel.__init__(self, groundModel=groundModel)
        self.a = a
        self.deltawD = deltawD
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti

    def epsilon_ilk(self, ct_ilk, **_):
        return 0.34 * np.ones_like(ct_ilk)

    def nw_length(self, ct_ilk, *args, **kwargs):
        return 1.91 * np.ones_like(ct_ilk)


class TurboGaussianDeficit(NiayifarGaussianDeficit):
    """Implemented similar to Ørsted's TurbOPark model (https://github.com/OrstedRD/TurbOPark)"""

    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il',
                    'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_ilk', 'TI_eff_ilk']

    def __init__(self, A=.04, cTI=[1.5, 0.8], ceps=.25, use_effective_ws=False,
                 use_effective_ti=False, groundModel=Mirror()):
        DeficitModel.__init__(self, groundModel=groundModel)
        self.A = A
        self.cTI = cTI
        self._ceps = ceps
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti

    def sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        # dimensional wake expansion
        # expression unchanged from original formulation, however the intial wake width needs to
        # be adjusted to agree with the Gaussian model formulation. It is replaced by the original
        # formulation by Bastankhah
        # ----TurboNOJ identical part
        TI_ref_ilk = (kwargs['TI_ilk'], kwargs['TI_eff_ilk'])[self.use_effective_ti]
        c1, c2 = self.cTI
        # constants related to ambient turbulence
        alpha_ilk = c1 * TI_ref_ilk
        # avoid zero division
        ct_ilk = np.maximum(ct_ilk, 1e-20)
        beta_ilk = c2 * TI_ref_ilk / np.sqrt(ct_ilk)

        fac_ilk = self.A * TI_ref_ilk * D_src_il[..., na] / beta_ilk
        term1_ijlk = np.sqrt((alpha_ilk[:, na] + beta_ilk[:, na] * dw_ijlk / D_src_il[:, na, :, na])**2 + 1)
        term2_ilk = np.sqrt(1 + alpha_ilk**2)
        term3_ijlk = (term1_ijlk + 1) * alpha_ilk[:, na]
        term4_ijlk = (term2_ilk[:, na] + 1) * (alpha_ilk[:, na] +
                                               beta_ilk[:, na] * cabs(dw_ijlk) / D_src_il[:, na, :, na])
        # ----
        expansion_ijlk = fac_ilk[:, na] * (term1_ijlk - term2_ilk[:, na] - np.log(term3_ijlk / term4_ijlk))

        return expansion_ijlk + self.epsilon_ilk(ct_ilk)[:, na] * D_src_il[:, na, :, na]


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt
        from py_wake.deficit_models.noj import NOJDeficit, TurboNOJDeficit
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake.superposition_models import LinearSum
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.examples.data import hornsrev1

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
        wfm_nojturbo = PropagateDownwind(site, windTurbines, rotorAvgModel=RotorCenter(),
                                         wake_deficitModel=TurboNOJDeficit(use_effective_ws=True,
                                                                           use_effective_ti=False),
                                         superpositionModel=LinearSum(),
                                         turbulenceModel=STF2017TurbulenceModel())
        wfm_gauturbo = PropagateDownwind(site, windTurbines, rotorAvgModel=RotorCenter(),
                                         wake_deficitModel=TurboGaussianDeficit(use_effective_ws=True,
                                                                                use_effective_ti=False),
                                         superpositionModel=SquaredSum(),
                                         turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model(x, y)
        sim_res_nojturbo = wfm_nojturbo(x, y)
        sim_res_gauturbo = wfm_gauturbo(x, y)
        aep_nojturbo = sim_res_nojturbo.aep().sum()
        aep_gauturbo = sim_res_gauturbo.aep().sum()
        aep = sim_res.aep().sum()

        # plot wake map
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.5), tight_layout=True)
        levels = np.arange(0, 10.5, 0.5)
        print(wf_model)
        flow_map = sim_res.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax1, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax1)
        ax1.set_title('Bastankhah Gaussian, AEP: %.2f GWh' % aep)

        # plot wake map
        print(wfm_nojturbo)
        flow_map = sim_res_nojturbo.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax2, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax2)
        ax2.set_title('Turbo Jensen, AEP: %.2f GWh' % aep_nojturbo)

        # plot wake map
        print(wfm_gauturbo)
        flow_map = sim_res_gauturbo.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax3, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax3)
        ax3.set_title('Turbo Gaussian, AEP: %.2f GWh' % aep_gauturbo)
        plt.show()

        # plot wake width as in Nygaard 2020
        D = 1
        D_src_il = np.array([[D]])
        x = np.linspace(0, 60, 100)
        dw_ijlk = x[na, :, na, na]

        noj = NOJDeficit(k=0.04)
        noj_wr = noj.wake_radius(D_src_il, dw_ijlk)

        ct_ilk = np.array([[[8 / 9]]])  # thrust coefficient
        TI_ilk = np.array([[[0.06]]])
        TI_eff_ilk = np.array([[[0.06]]])
        tj = TurboNOJDeficit(A=0.6)
        tj_wr = tj.wake_radius(D_src_il, dw_ijlk, ct_ilk=ct_ilk, TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk)

        tjg = TurboGaussianDeficit(A=0.04)
        gau = BastankhahGaussianDeficit(k=0.04)
        tjg_wr = tjg.wake_radius(D_src_il, dw_ijlk, ct_ilk=ct_ilk, TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk)
        gau_wr = gau.wake_radius(D_src_il, dw_ijlk, ct_ilk=ct_ilk, TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk)

        plt.figure()
        plt.title('Wake width comparison, NOJ and Gauss (Nygaard2020) TI=6%')
        plt.plot(x, noj_wr[0, :, 0, 0], label='NOJ, k=0.04')
        plt.plot(x, tj_wr[0, :, 0, 0], label='TurboNOJ')
        plt.plot(x, tjg_wr[0, :, 0, 0], '--', label='TurboGauss')
        plt.plot(x, gau_wr[0, :, 0, 0], '--', label='Bastankhah')
        plt.xlabel('x/D')
        plt.ylabel('y/D')
        plt.grid()
        plt.legend()
        plt.show()

        # compare deficits
        site = Hornsrev1Site()
        windTurbines = hornsrev1.HornsrevV80()
        ws = 10
        D = 80
        R = D / 2
        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[.8]]])
        x, y = np.arange(20 * D), np.array([0])
        noj_def = noj.calc_deficit(WS_ilk=WS_ilk, WS_eff_ilk=WS_ilk, D_src_il=D_src_il, D_dst_ijl=D_src_il,
                                   TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk,
                                   dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                   cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        tj_def = tj.calc_deficit(WS_ilk=WS_ilk, WS_eff_ilk=WS_ilk, D_src_il=D_src_il, D_dst_ijl=D_src_il,
                                 TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk,
                                 dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                 cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        tjg_def = tjg.calc_deficit(WS_ilk=WS_ilk, WS_eff_ilk=WS_ilk, D_src_il=D_src_il,
                                   TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk,
                                   dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                   cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        gau_def = gau.calc_deficit(WS_ilk=WS_ilk, WS_eff_ilk=WS_ilk, D_src_il=D_src_il,
                                   TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk,
                                   dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                   cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)

        plt.figure()
        plt.title('Deficit')
        plt.xlabel('x/R')
        plt.ylabel('u/u_inf')
        plt.plot(x / R, 1. - noj_def[0, :, 0, 0] / ws, label='NOJ (k=0.04)')
        plt.plot(x / R, 1. - tj_def[0, :, 0, 0] / ws, label='TurboNOJ (A=0.6)')
        plt.plot(x / R, 1. - tjg_def[0, :, 0, 0] / ws, '--', label='TurboGauss (A=0.04)')
        plt.plot(x / R, 1. - gau_def[0, :, 0, 0] / ws, '-.', label='Bastankhah (k=0.04)')
        plt.legend()
        plt.show()


main()
