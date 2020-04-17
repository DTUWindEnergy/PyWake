from abc import abstractmethod
from numpy import newaxis as na
import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.superposition_models import SuperpositionModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.deflection_models.deflection_model import DeflectionModel


class EngineeringWindFarmModel(WindFarmModel):
    """
    Base class for engineering wake models

    General suffixes:

    - i: turbines ordered by id
    - j: downstream points/turbines
    - k: wind speeds
    - l: wind directions

    Arguments available for calc_deficit (specifiy in args4deficit):

    - WS_ilk: Local wind speed without wake effects
    - TI_ilk: local turbulence intensity without wake effects
    - WS_eff_ilk: Local wind speed with wake effects
    - TI_eff_ilk: local turbulence intensity with wake effects
    - D_src_il: Diameter of source turbine
    - D_dst_ijl: Diameter of destination turbine
    - dw_ijl: Downwind distance from turbine i to point/turbine j
    - hcw_ijl: Horizontal cross wind distance from turbine i to point/turbine j
    - cw_ijl: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_ilk: Thrust coefficient

    """
    default_grid_resolution = 500

    def __init__(self, site, windTurbines, wake_deficitModel, superpositionModel,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None):

        WindFarmModel.__init__(self, site, windTurbines)

        assert isinstance(wake_deficitModel, DeficitModel)
        assert isinstance(superpositionModel, SuperpositionModel)
        assert blockage_deficitModel is None or isinstance(blockage_deficitModel, DeficitModel)
        assert deflectionModel is None or isinstance(deflectionModel, DeflectionModel)
        assert turbulenceModel is None or isinstance(turbulenceModel, TurbulenceModel)
        self.site = site
        self.windTurbines = windTurbines
        self.wake_deficitModel = wake_deficitModel
        self.superpositionModel = superpositionModel
        self.blockage_deficitModel = blockage_deficitModel
        self.deflectionModel = deflectionModel
        self.turbulenceModel = turbulenceModel

        self.wec = 1  # wake expansion continuation (wake-width scale factor) see
        # Thomas, J. J. and Ning, A., “A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,”
        # Journal of Physics: Conference Series, Vol. 1037, The Science of Making
        # Torque from Wind, Milano, Italy, jun 2018, p. 10.
        self.deficit_initalized = False

        self.args4deficit = self.wake_deficitModel.args4deficit
        if self.blockage_deficitModel:
            self.args4deficit = set(self.args4deficit) | set(self.blockage_deficitModel.args4deficit)
        if self.turbulenceModel:
            self.args4deficit = set(self.args4deficit) | set(self.turbulenceModel.args4addturb)

    def __str__(self):
        def name(o):
            return o.__class__.__name__

        models = [self.__class__.__bases__[0].__name__,
                  "%s-wake" % name(self.wake_deficitModel)]
        if self.blockage_deficitModel:
            models.append("%s-blockage" % name(self.blockage_deficitModel))
        models.append("%s-superposition" % (name(self.superpositionModel)))
        if self.deflectionModel:
            models.append("%s-deflection" % name(self.deflectionModel))
        if self.turbulenceModel:
            models.append("%s-turbulence" % name(self.turbulenceModel))
        return "%s(%s)" % (name(self), ", ".join(models))

    def _init_deficit(self, **kwargs):
        """Calculate layout dependent wake (and blockage) deficit terms"""
        self.wake_deficitModel._calc_layout_terms(**kwargs)
        self.wake_deficitModel.deficit_initalized = True
        if self.blockage_deficitModel:
            self.blockage_deficitModel._calc_layout_terms(**kwargs)
            self.blockage_deficitModel.deficit_initalized = True

    def _reset_deficit(self):
        self.wake_deficitModel.deficit_initalized = False
        if self.blockage_deficitModel:
            self.blockage_deficitModel.deficit_initalized = False

    def _calc_deficit(self, dw_ijlk, **kwargs):
        """Calculate wake (and blockage) deficit"""
        deficit = self.wake_deficitModel.calc_deficit(dw_ijlk=dw_ijlk, **kwargs)
        if self.blockage_deficitModel is None:
            deficit *= (dw_ijlk > 0)
            pass
        elif self.blockage_deficitModel != self:
            deficit = deficit * (dw_ijlk > 0) + \
                (dw_ijlk < 0) * self.blockage_deficitModel.calc_deficit(dw_ijlk=dw_ijlk, **kwargs)

        return deficit

    def calc_wt_interaction(self, x_i, y_i, h_i=None, type_i=0, wd=None, ws=None, yaw_ilk=None):
        """See WindFarmModel.calc_wt_interaction"""
        type_i, h_i, D_i = self.windTurbines.get_defaults(len(x_i), type_i, h_i)
        wd, ws = self.site.get_defaults(wd, ws)

        # Find local wind speed, wind direction, turbulence intensity and probability
        lw = self.site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws)

        # Calculate down-wind and cross-wind distances
        dw_iil, hcw_iil, dh_iil, dw_order_indices_dl = self.site.wt2wt_distances(x_i, y_i, h_i, lw.WD_ilk.mean(2))
        self._validate_input(dw_iil, hcw_iil)

        I, L = dw_iil.shape[1:]
        K = lw.WS_ilk.shape[2]
        WS_eff_ilk = lw.WS_ilk.copy()
        TI_eff_ilk = lw.TI_ilk.copy()
        if yaw_ilk is None:
            yaw_ilk = np.zeros((I, L, K))
        else:
            yaw_ilk = np.deg2rad(yaw_ilk)

        if self.wec != 1:
            hcw_iil = hcw_iil / self.wec

        # add eps to avoid non-differentiable 0
        if 'autograd' in np.__name__:
            eps = 2 * np.finfo(np.float).eps ** 2
        else:
            eps = 0
        cw_iil = np.sqrt(hcw_iil**2 + dh_iil**2 + eps)

        kwargs = {'localWind': lw,
                  'WS_eff_ilk': WS_eff_ilk, 'TI_eff_ilk': TI_eff_ilk,
                  'type_i': type_i, 'h_i': h_i, 'D_i': D_i, 'yaw_ilk': yaw_ilk,
                  'dw_iil': dw_iil, 'hcw_iil': hcw_iil, 'cw_iil': cw_iil, 'dh_iil': dh_iil,
                  'dw_order_indices_dl': dw_order_indices_dl, 'I': I, 'L': L, 'K': K}
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk = self._calc_wt_interaction(**kwargs)
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, lw

    @abstractmethod
    def _calc_wt_interaction(self, **kwargs):
        """calculate WT interaction"""

    def _flow_map(self, x_j, y_j, h_j,
                  wt_x_i, wt_y_i, wt_h_i, wt_type_i, yaw_ilk,
                  WD_ilk, WS_ilk, TI_ilk, WS_eff_ilk, TI_eff_ilk, ct_ilk,
                  wd, ws):
        """call this function via SimulationResult.flow_map"""
        # calculate distances
        wt_d_i = self.windTurbines.diameter(wt_type_i)

        lw_j = self.site.local_wind(x_i=x_j, y_i=y_j, h_i=h_j, wd=wd, ws=ws)
        if len(wt_x_i) == 0:
            # If not turbines just return local wind
            return lw_j, lw_j.WS_ilk, lw_j.TI_ilk

        I, J, L, K = [len(x) for x in [wt_x_i, x_j, wd, ws]]
        WS_eff_jlk = np.zeros((len(x_j), L, K))
        TI_eff_jlk = np.zeros((len(x_j), L, K))

        if self.deflectionModel:
            if yaw_ilk is None:
                yaw_ilk = np.zeros((I, L, K))
            yaw_ilk = np.deg2rad(yaw_ilk)
        if L > 1:
            print("|" + "-" * 100 + "|\n|", end="", flush=True)
        for l in range(L):
            if L > 1 and (l * 100) // L > ((l - 1) * 100) // L:
                print(".", end="", flush=True)
            dw_ijl, hcw_ijl, dh_ijl, _ = self.site.distances(wt_x_i, wt_y_i, wt_h_i, x_j, y_j, h_j,
                                                             wd_il=WD_ilk[:, l:l + 1, :].mean(2))

            if self.wec != 1:
                hcw_ijl = hcw_ijl / self.wec

            if I * J * K * 8 / 1024**2 > 10:
                # one wt at the time to avoid memory problems
                deficit_ijk = np.zeros((I, J, K))
                add_turb_ijk = np.zeros((I, J, K))
                for i in range(I):
                    arg_funcs = {'WS_ilk': lambda: WS_ilk[i, l][na, na],
                                 'WS_eff_ilk': lambda: WS_eff_ilk[i, l][na, na],
                                 'TI_ilk': lambda: TI_ilk[i, l][na, na],
                                 'TI_eff_ilk': lambda: TI_eff_ilk[i, l][na, na],
                                 'yaw_ilk': lambda: yaw_ilk[i, l][na, na],
                                 'D_src_il': lambda: wt_d_i[i][na, na],
                                 'D_dst_ijl': lambda: None,
                                 'dh_ijl': lambda: dh_ijl[i][na],
                                 'h_il': lambda: wt_h_i[i][na, na],
                                 'ct_ilk': lambda: ct_ilk[i, l][na, na]}

                    if self.deflectionModel:
                        dw_ijlk, hcw_ijlk = self.deflectionModel.calc_deflection(
                            dw_ijl[i][na], hcw_ijl[i][na],
                            **{k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})
                    else:
                        dw_ijlk, hcw_ijlk = dw_ijl[i][na, :, :, na], hcw_ijl[i][na, :, :, na]
                    arg_funcs['cw_ijlk'] = lambda: np.hypot(dh_ijl[i][na, :, :, na], hcw_ijlk)
                    arg_funcs['hcw_ijlk'] = lambda: hcw_ijlk

                    args = {k: arg_funcs[k]() for k in self.args4deficit if k != 'dw_ijlk'}
                    deficit_ijk[i] = self._calc_deficit(dw_ijlk=dw_ijlk, **args)[0, :, 0]
                    if self.turbulenceModel:
                        add_turb_ijk[i] = self.turbulenceModel.calc_added_turbulence(dw_ijlk=dw_ijlk, **args)[0, :, 0]

            else:

                arg_funcs = {'WS_ilk': lambda: WS_ilk[:, l][:, na],
                             'WS_eff_ilk': lambda: WS_eff_ilk[:, l][:, na],
                             'TI_ilk': lambda: TI_ilk[:, l][:, na],
                             'TI_eff_ilk': lambda: TI_eff_ilk[:, l][:, na],
                             'yaw_ilk': lambda: yaw_ilk[:, l][:, na],
                             'D_src_il': lambda: wt_d_i[:, na],
                             'D_dst_ijl': lambda: None,
                             'dh_ijl': lambda: dh_ijl,
                             'h_il': lambda: wt_h_i[:, na],
                             'ct_ilk': lambda: ct_ilk[:, l][:, na]}

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijl, hcw_ijl,
                        **{k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})
                else:
                    dw_ijlk, hcw_ijlk = dw_ijl[..., na], hcw_ijl[..., na]
                arg_funcs['cw_ijlk'] = lambda: np.hypot(dh_ijl[..., na], hcw_ijlk)
                arg_funcs['dw_ijlk'] = lambda: dw_ijlk
                arg_funcs['hcw_ijlk'] = lambda: hcw_ijlk

                args = {k: arg_funcs[k]() for k in self.args4deficit}
                deficit_ijk = self._calc_deficit(**args)[:, :, 0]
                if self.turbulenceModel:
                    add_turb_ijk = self.turbulenceModel.calc_added_turbulence(**args)[:, :, 0]
            WS_eff_jlk[:, l] = self.superpositionModel.calc_effective_WS(lw_j.WS_ilk[:, l], deficit_ijk)
            if self.turbulenceModel:
                TI_eff_jlk[:, l] = self.turbulenceModel.calc_effective_TI(lw_j.TI_ilk[:, l], add_turb_ijk)
        return lw_j, WS_eff_jlk, TI_eff_jlk

    def _validate_input(self, dw_iil, hcw_iil):
        I_ = dw_iil.shape[0]
        i1, i2, _ = np.where((np.abs(dw_iil) + np.abs(hcw_iil) + np.eye(I_)[:, :, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" %
                             (i1[i], i2[i]) for i in range(len(i1))])
            raise ValueError(msg)


class PropagateDownwind(EngineeringWindFarmModel):
    """Downstream wake deficits calculated and propagated in downstream direction.
    Very fast, but ignoring blockage effects
    """

    def __init__(self, site, windTurbines, wake_deficitModel, superpositionModel,
                 deflectionModel=None, turbulenceModel=None):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel,
                                          blockage_deficitModel=None, deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)

    def _calc_wt_interaction(self, localWind,
                             WS_eff_ilk, TI_eff_ilk,
                             type_i, h_i, D_i, yaw_ilk,
                             dw_iil, hcw_iil, cw_iil, dh_iil, dw_order_indices_dl, I, L, K):
        """
        Additional suffixes:

        - m: turbines and wind directions (il.flatten())
        - n: from_turbines, to_turbines and wind directions (iil.flatten())

        """
        lw = localWind
        deficit_nk = np.zeros((I * I * L, K))
        ct_ilk = np.zeros_like(lw.WS_ilk)

        def ilk2mk(x_ilk):
            return x_ilk.astype(np.float).reshape((I * L, K))

        if not self.deflectionModel:
            dw_ijlk, hcw_ijlk = dw_iil[..., na], hcw_iil[..., na]

        if self.turbulenceModel:
            add_turb_nk = np.zeros((I * I * L, K))
            TI_mk = ilk2mk(lw.TI_ilk)
            TI_eff_mk = ilk2mk(TI_eff_ilk)

        indices = np.arange(I * I * L).reshape((I, I, L))
        WS_mk = ilk2mk(lw.WS_ilk)
        WS_eff_mk = ilk2mk(WS_eff_ilk)
        yaw_mk = ilk2mk(yaw_ilk)
        dw_n, hcw_n, cw_n, dh_n = [a.flatten() for a in [dw_iil, hcw_iil, cw_iil, dh_iil]]
        power_ilk = np.zeros((I, L, K))

        i_wd_l = np.arange(L)

        # Iterate over turbines in down wind order
        for j in range(I):
            i_wt_l = dw_order_indices_dl[:, j]
            m = i_wt_l * L + i_wd_l  # current wt (j'th most upstream wts for all wdirs)

            # generate indexes of up wind(n_uw) and down wind(n_dw) turbines
            n_uw = indices[:, i_wt_l, i_wd_l][dw_order_indices_dl[:, :j].T, np.arange(L)]
            n_dw = indices[i_wt_l, :, i_wd_l][np.arange(L), dw_order_indices_dl[:, j + 1:].T]

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
            else:  # 2..n most upstream turbines (wake)
                WS_eff_lk = self.superpositionModel.calc_effective_WS(WS_mk[m], deficit_nk[n_uw])
                WS_eff_mk[m] = WS_eff_lk
                if self.turbulenceModel:
                    TI_eff_mk[m] = self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb_nk[n_uw])

            ct_lk, power_lk = self.windTurbines._ct_power(WS_eff_lk, type_i[i_wt_l])

            power_ilk[i_wt_l, i_wd_l] = power_lk
            ct_ilk[i_wt_l, i_wd_l, :] = ct_lk

            if j < I - 1:
                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_eff_ilk': lambda: WS_eff_mk[m][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[m][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'D_dst_ijl': lambda: D_i[dw_order_indices_dl[:, j + 1:]].T[na],
                             'dh_ijl': lambda: dh_n[n_dw][na],
                             'h_il': lambda: h_i[i_wt_l][na],
                             'ct_ilk': lambda: ct_ilk.reshape((I * L, K))[m][na]}

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijl=dw_n[n_dw][na], hcw_ijl=hcw_n[n_dw][na],
                        **{k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})

                else:
                    dw_ijlk, hcw_ijlk = dw_n[n_dw][na, :, :, na], hcw_n[n_dw][na, :, :, na],

                arg_funcs['hcw_ijlk'] = lambda: hcw_ijlk
                arg_funcs['cw_ijlk'] = lambda: np.hypot(dh_n[n_dw][na, :, :, na], hcw_ijlk)
                args = {k: arg_funcs[k]() for k in self.args4deficit if k != "dw_ijlk"}

                # Calcualte deficit
                deficit_nk[n_dw] = self.wake_deficitModel.calc_deficit(dw_ijlk=dw_ijlk, **args)[0]
                if self.turbulenceModel:
                    # Calculate added turbulence
                    add_turb_nk[n_dw] = self.turbulenceModel.calc_added_turbulence(dw_ijlk=dw_ijlk, **args)

        WS_eff_ilk = WS_eff_mk.reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_ilk = TI_eff_mk.reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk


class All2AllIterative(EngineeringWindFarmModel):
    """Wake and blockage deficits calculated from all wt to all points of interest (wt/map points).
    The calculations are iteratively repeated until convergence (change of effective wind speed < convergence_tolerance)"""

    def __init__(self, site, windTurbines, wake_deficitModel, superpositionModel,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        convergence_tolerance : float
            maximum accepted change in WS_eff_ilk [m/s]
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel,
                                          blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)
        self.convergence_tolerance = convergence_tolerance

    def _calc_wt_interaction(self, localWind,
                             WS_eff_ilk, TI_eff_ilk,
                             type_i, h_i, D_i, yaw_ilk,
                             dw_iil, hcw_iil, cw_iil, dh_iil, dw_order_indices_dl, I, L, K):
        lw = localWind
        power_ilk = np.zeros((I, L, K))
        WS_eff_ilk_last = WS_eff_ilk.copy()

        ct_ilk = self.windTurbines.ct(lw.WS_ilk, type_i)
        D_src_il = D_i[:, na]
        args = {'WS_ilk': lw.WS_ilk,
                'TI_ilk': lw.TI_ilk,
                'TI_eff_ilk': lw.TI_ilk,
                'yaw_ilk': yaw_ilk,
                'D_src_il': D_src_il,
                'D_dst_ijl': D_src_il[na],
                'cw_ijlk': cw_iil[..., na],
                'dh_ijl': dh_iil,
                'h_il': h_i[:, na]
                }

        # Iterate until convergence
        for j in range(I):

            ct_ilk, power_ilk = self.windTurbines._ct_power(WS_eff_ilk, type_i)
            args['ct_ilk'] = ct_ilk
            args['WS_eff_ilk'] = WS_eff_ilk
            if self.deflectionModel:
                dw_ijlk, hcw_ijlk = self.deflectionModel.calc_deflection(dw_ijl=dw_iil, hcw_ijl=dw_iil, **args)
                args['dw_ijlk'] = dw_ijlk
                args['hcw_ijlk'] = hcw_ijlk
                args['cw_ijlk'] = np.hypot(dh_iil[..., na], hcw_ijlk)
            else:
                args['dw_ijlk'] = dw_iil[..., na]
                args['hcw_ijlk'] = hcw_iil[..., na]
                self._init_deficit(**args)
            if self.turbulenceModel:
                args['TI_eff_ilk'] = TI_eff_ilk

            # Calculate deficit
            deficit_iilk = self._calc_deficit(**args)

            # Calculate effective wind speed
            WS_eff_ilk = self.superpositionModel.calc_effective_WS(lw.WS_ilk, deficit_iilk)
            if self.turbulenceModel:
                add_turb_ijlk = self.turbulenceModel.calc_added_turbulence(**args)
                TI_eff_ilk = self.turbulenceModel.calc_effective_TI(lw.TI_ilk, add_turb_ijlk)

            # Check if converged
            diff = np.abs(WS_eff_ilk_last - WS_eff_ilk)
            max_diff = np.max(diff)
            if self.convergence_tolerance and max_diff < self.convergence_tolerance:
                # print("All2AllIterative converge after %d iterations" % j)
                break
            # i_, l_, k_ = list(zip(*np.where(diff == max_diff)))[0]
            # print("Iteration: %d, max diff: %f, WT: %d, WD: %d, WS: %d" % (j, max_diff, i_, l_, WS_ilk[i_, l_, k_]))
            WS_eff_ilk_last = WS_eff_ilk.copy()
        self._reset_deficit()
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
        from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit

        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        x, y = site.initial_position.T

        windTurbines = IEA37_WindTurbines()
        from py_wake.deficit_models.noj import NOJDeficit
        from py_wake.superposition_models import SquaredSum

        # NOJ wake model
        noj = PropagateDownwind(site, windTurbines, wake_deficitModel=NOJDeficit(), superpositionModel=SquaredSum())

        # NOJ wake and selfsimilarity blockage
        noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NOJDeficit(), superpositionModel=SquaredSum(),
                                  blockage_deficitModel=SelfSimilarityDeficit())

        for wm in [noj, noj_ss]:
            plt.figure()
            wm(x=x, y=y, wd=[30], ws=[9]).flow_map().plot_wake_map()
        plt.show()


main()
