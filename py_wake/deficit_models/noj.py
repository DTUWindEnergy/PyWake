from numpy import newaxis as na
from py_wake import np
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit
from py_wake.deficit_models.deficit_model import DeficitModel
from py_wake.utils.gradients import cabs
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from warnings import catch_warnings, filterwarnings
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat
from py_wake.deficit_models.utils import ct2a_madsen


class NOJDeficit(NiayifarGaussianDeficit, WakeRadiusTopHat):

    def __init__(self, ct2a=ct2a_madsen, k=.1, rotorAvgModel=AreaOverlapAvgModel(), groundModel=None):
        DeficitModel.__init__(self, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                              use_effective_ws=False, use_effective_ti=False)
        self.a = [0, k]
        self.ct2a = ct2a

    def _calc_layout_terms(self, D_src_il, wake_radius_ijl, dw_ijlk, cw_ijlk, **kwargs):
        WS_ref_ilk = kwargs[self.WS_key]
        R_src_il = D_src_il / 2
        with catch_warnings():
            filterwarnings('ignore', r'invalid value encountered in true_divide')
            filterwarnings('ignore', r'invalid value encountered in divide')
            filterwarnings('ignore', r'divide by zero encountered in divide')

            term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijl / R_src_il[:, na, :])**2)[..., na], 1)

        in_wake_ijlk = wake_radius_ijl[..., na] > cw_ijlk

        with catch_warnings():
            filterwarnings('ignore', r'invalid value encountered in true_divide')
            filterwarnings('ignore', r'invalid value encountered in divide')
            self.layout_factor_ijlk = WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)

    def calc_deficit(self, ct_ilk, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(ct_ilk=ct_ilk, **kwargs)
        term_numerator_ilk = 2. * self.ct2a(ct_ilk)
        return term_numerator_ilk[:, na] * self.layout_factor_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, **kwargs):
        # k_ijlk = np.atleast_3d(self.k_ilk(kwargs.get('TI_eff_ilk', 0)))[:, na]
        if 'TI_eff_ilk' not in kwargs:
            kwargs['TI_eff_ilk'] = 0.0
            kwargs['TI_ilk'] = 0.0
        k_ijlk = np.atleast_3d(self.k_ilk(**kwargs))[:, na]
        wake_radius_ijlk = (k_ijlk * dw_ijlk + D_src_il[:, na, :, na] / 2)
        return wake_radius_ijlk

    def calc_deficit_convection(self, WS_ilk, WS_eff_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        raise NotImplementedError("calc_deficit_convection not implemented for NOJ")


class NOJ(PropagateDownwind):
    def __init__(self, site, windTurbines, rotorAvgModel=AreaOverlapAvgModel(),
                 ct2a=ct2a_madsen, k=.1, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None,
                 groundModel=None):
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
                                   wake_deficitModel=NOJDeficit(
                                       k=k, ct2a=ct2a, rotorAvgModel=rotorAvgModel, groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class NOJLocalDeficit(NOJDeficit):
    """
    Largely identical to NOJDeficit(), however using local quantities for the
    inflow wind speed and turbulence intensity. The latter input is a also a new
    addition as the wake expansion factor, k, is now a function of the local
    TI. The relationship between TI and k is taken from the linear connection
    Niayifar and Porte-Agel (2016) estbalished for the Gaussian wake model.
    The expansion rates in the Jensen and Gaussian describe the same process.
    """

    def __init__(self, ct2a=ct2a_madsen, a=[0.38, 4e-3], use_effective_ws=True, use_effective_ti=True,
                 rotorAvgModel=AreaOverlapAvgModel(), groundModel=None):
        DeficitModel.__init__(self, groundModel=groundModel, rotorAvgModel=rotorAvgModel,
                              use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)
        self.a = a
        self.ct2a = ct2a

    def _calc_layout_terms(self, **kwargs):
        """Layout factor cannot be precomputed as it may depend on TI_eff if a[0]!=0"""

    def calc_deficit(self, ct_ilk, D_src_il, wake_radius_ijlk, dw_ijlk, cw_ijlk, **kwargs):
        # ws-dependent layout factor
        WS_ref_ilk = kwargs[self.WS_key]
        R_src_il = D_src_il / 2
        with catch_warnings():
            filterwarnings('ignore', r'invalid value encountered in true_divide')
            term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijlk / R_src_il[:, na, :, na])**2), 1)

        in_wake_ijlk = wake_radius_ijlk > cw_ijlk

        with catch_warnings():
            filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            layout_factor_ijlk = WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)

        term_numerator_ilk = 2. * self.ct2a(ct_ilk)
        return term_numerator_ilk[:, na] * layout_factor_ijlk


class NOJLocal(PropagateDownwind):
    def __init__(self, site, windTurbines, rotorAvgModel=AreaOverlapAvgModel(),
                 ct2a=ct2a_madsen, a=[0.38, 4e-3], use_effective_ws=True,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=STF2017TurbulenceModel(),
                 groundModel=None):
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
                                   wake_deficitModel=NOJLocalDeficit(ct2a=ct2a, a=a, use_effective_ws=use_effective_ws,
                                                                     rotorAvgModel=rotorAvgModel,
                                                                     groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class TurboNOJDeficit(NOJDeficit):
    """
    Modified definition of the wake expansion given by Nygaard [1], which
    assumes the wake expansion rate to be proportional to the local turbulence
    intensity in the wake. Here the local turbulence intensity is defined as
    the combination of ambient and wake added turbulence. Using the added
    wake turbulence model by Frandsen and integrating, an analytical expression
    for the wake radius can be obtained.
    The definition in [1] of ambient turbulence is the free-stream TI and for
    this the model constant A has been tuned, however a fully consistent
    formulation of the model should probably use the locally effective TI,
    which includes the added turbulence from upstream turbines.
    [1] Nygaard 2020 J. Phys.: Conf. Ser. 1618 062072
        https://doi.org/10.1088/1742-6596/1618/6/062072
    """

    def __init__(self, ct2a=ct2a_madsen, A=.6, cTI=[1.5, 0.8], use_effective_ws=False, use_effective_ti=False,
                 rotorAvgModel=AreaOverlapAvgModel(), groundModel=None):
        DeficitModel.__init__(self, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                              use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)
        self.A = A
        # default constants from Frandsen
        self.cTI = cTI
        self.ct2a = ct2a

    def _calc_layout_terms(self, *args, **kwargs):
        """Wake radius depends on CT, i.e. it cannot be precalculated"""

    def calc_deficit(self, D_src_il, wake_radius_ijlk,
                     dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        WS_ref_ilk = kwargs[self.WS_key]
        R_src_il = D_src_il / 2
        with catch_warnings():  #
            filterwarnings('ignore', r'invalid value encountered in true_divide')
            term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijlk / R_src_il[:, na, :, na])**2), 1)

        in_wake_ijlk = wake_radius_ijlk > cw_ijlk

        term_numerator_ilk = 2. * self.ct2a(ct_ilk)
        with catch_warnings():
            filterwarnings('ignore', r'invalid value encountered in true_divide')
            return term_numerator_ilk[:, na] * WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)

    def wake_radius(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        TI_ref_ilk = kwargs[self.TI_key]
        ct_ilk = ct_ilk
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

        wake_radius_ijlk = 0.5 * (D_src_il[:, na, :, na] + fac_ilk[:, na] *
                                  (term1_ijlk - term2_ilk[:, na] - np.log(term3_ijlk / term4_ijlk)))

        return wake_radius_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wf_model = NOJ(site, windTurbines, k=0.04)
        wf_model_local = NOJLocal(site, windTurbines, turbulenceModel=STF2017TurbulenceModel())
        wf_model_turbo = PropagateDownwind(site, windTurbines, rotorAvgModel=None,
                                           wake_deficitModel=TurboNOJDeficit(
                                               use_effective_ws=True, use_effective_ti=False),
                                           superpositionModel=LinearSum(),
                                           turbulenceModel=STF2017TurbulenceModel())
        # wf_model_turbo = NOJLocal(
        #     site, windTurbines, turbulenceModel=STF2017TurbulenceModel())
        # run wind farm simulation
        sim_res = wf_model(x, y)
        sim_res_local = wf_model_local(x, y)
        sim_res_turbo = wf_model_turbo(x, y)
        # calculate AEP
        aep = sim_res.aep().sum()
        aep_local = sim_res_local.aep().sum()
        aep_turbo = sim_res_turbo.aep().sum()

        # plot wake map
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.5), tight_layout=True)
        levels = np.arange(0, 10.5, 0.5)
        print(wf_model)
        flow_map = sim_res.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax1, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax1)
        ax1.set_title('Original Jensen, AEP: %.2f GWh' % aep)

        # plot wake map
        print(wf_model_local)
        flow_map = sim_res_local.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax2, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax2)
        ax2.set_title('Local Jensen, AEP: %.2f GWh' % aep_local)

        # plot wake map
        print(wf_model_turbo)
        flow_map = sim_res_turbo.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax3, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax3)
        ax3.set_title('Turbo Jensen, AEP: %.2f GWh' % aep_turbo)

        plt.figure()
        flow_map.plot_ti_map()
        plt.title('TI map for NOJLocal with STF2017 turbulence model')
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
        tj = TurboNOJDeficit()
        tj_wr = tj.wake_radius(D_src_il, dw_ijlk, ct_ilk=ct_ilk, TI_ilk=TI_ilk, TI_eff_ilk=TI_eff_ilk)

        plt.figure()
        plt.title('Wake width comparison, NOJ orig and TurboNOJ (Nygaard2020) TI=6%')
        plt.plot(x, noj_wr[0, :, 0, 0], label='NOJ, k=0.04')
        plt.plot(x, tj_wr[0, :, 0, 0], label='TurboNOJ')
        plt.xlabel('x/D')
        plt.ylabel('y/D')
        plt.grid()
        plt.legend()
        plt.show()


main()
