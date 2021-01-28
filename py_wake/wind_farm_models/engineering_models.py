from abc import abstractmethod
from numpy import newaxis as na
import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.superposition_models import SuperpositionModel, LinearSum, WeightedSum
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.utils.gradients import use_autograd_in, autograd
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter, RotorAvgModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.deficit_models.deficit_model import ConvectionDeficitModel
from py_wake.ground_models.ground_models import NoGround, GroundModel
from tqdm import tqdm


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
    - TI_ilk: Local turbulence intensity without wake effects
    - TI_std_ilk: Standard deviation of local turbulence intensity
    - WS_eff_ilk: Local wind speed with wake effects
    - TI_eff_ilk: Local turbulence intensity with wake effects
    - D_src_il: Diameter of source turbine
    - D_dst_ijl: Diameter of destination turbine
    - dw_ijlk: Downwind distance from turbine i to point/turbine j
    - hcw_ijlk: Horizontal cross wind distance from turbine i to point/turbine j
    - dh_ijl: vertical distance from turbine i to point/turbine j
    - cw_ijlk: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_ilk: Thrust coefficient

    """
    default_grid_resolution = 500

    def __init__(self, site, windTurbines, wake_deficitModel, rotorAvgModel, superpositionModel,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 groundModel=None):

        WindFarmModel.__init__(self, site, windTurbines)

        assert isinstance(wake_deficitModel, DeficitModel)
        assert isinstance(rotorAvgModel, RotorAvgModel)
        assert isinstance(superpositionModel, SuperpositionModel)
        assert blockage_deficitModel is None or isinstance(blockage_deficitModel, DeficitModel)
        assert deflectionModel is None or isinstance(deflectionModel, DeflectionModel)
        assert turbulenceModel is None or isinstance(turbulenceModel, TurbulenceModel)
        if groundModel is None:
            groundModel = NoGround()
        assert isinstance(groundModel, GroundModel)
        if isinstance(superpositionModel, WeightedSum):
            assert isinstance(wake_deficitModel, ConvectionDeficitModel)
            assert rotorAvgModel.__class__ is RotorCenter, "Multiple rotor average points not implemented for WeightedSum"
        assert 'TI_eff_ilk' not in wake_deficitModel.args4deficit or turbulenceModel  # TI_eff requires a turbulence model
        self.wake_deficitModel = wake_deficitModel
        self.rotorAvgModel = rotorAvgModel

        self.superpositionModel = superpositionModel
        self.blockage_deficitModel = blockage_deficitModel
        self.deflectionModel = deflectionModel
        self.turbulenceModel = turbulenceModel
        self.groundModel = groundModel

        self.wec = 1  # wake expansion continuation (wake-width scale factor) see
        # Thomas, J. J. and Ning, A., “A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,”
        # Journal of Physics: Conference Series, Vol. 1037, The Science of Making
        # Torque from Wind, Milano, Italy, jun 2018, p. 10.
        self.deficit_initalized = False

        self.args4deficit = self.wake_deficitModel.args4deficit
        self.args4deficit = set(self.args4deficit) | {'yaw_ilk'} | set(self.rotorAvgModel.args4rotor_avg_deficit)
        if self.blockage_deficitModel:
            self.args4deficit = set(self.args4deficit) | set(self.blockage_deficitModel.args4deficit)
        if self.groundModel:
            self.args4deficit = set(self.args4deficit) | set(self.groundModel.args4deficit)
        self.args4all = set(self.args4deficit)
        if self.turbulenceModel:
            if self.turbulenceModel.rotorAvgModel is None:
                self.turbulenceModel.rotorAvgModel = rotorAvgModel
            self.args4addturb = set(self.turbulenceModel.args4addturb) | set(
                self.turbulenceModel.rotorAvgModel.args4rotor_avg_deficit)
            self.args4all = self.args4all | set(self.turbulenceModel.args4addturb)
        if self.deflectionModel:
            self.args4all = self.args4all | set(self.deflectionModel.args4deflection)

    def __str__(self):
        def name(o):
            return o.__class__.__name__

        models = [self.__class__.__bases__[0].__name__,
                  "%s-wake" % name(self.wake_deficitModel)]
        if self.blockage_deficitModel:
            models.append("%s-blockage" % name(self.blockage_deficitModel))
        models.append("%s-rotor-average" % (name(self.rotorAvgModel)))
        models.append("%s-superposition" % (name(self.superpositionModel)))
        if self.deflectionModel:
            models.append("%s-deflection" % name(self.deflectionModel))
        if self.turbulenceModel:
            models.append("%s-turbulence" % name(self.turbulenceModel))
        return "%s(%s)" % (name(self), ", ".join(models))

    def _init_deficit(self, **kwargs):
        """Calculate layout dependent wake (and blockage) deficit terms"""
        self.rotorAvgModel._calc_layout_terms(self.wake_deficitModel, **kwargs)
        self.wake_deficitModel.deficit_initalized = True
        if self.blockage_deficitModel:
            if self.blockage_deficitModel != self.wake_deficitModel:
                self.blockage_deficitModel._calc_layout_terms(**kwargs)
            self.blockage_deficitModel.deficit_initalized = True

    def _reset_deficit(self):
        self.wake_deficitModel.deficit_initalized = False
        if self.blockage_deficitModel:
            self.blockage_deficitModel.deficit_initalized = False

    def _add_blockage(self, deficit, dw_ijlk, **kwargs):
        # the split line between wake and blockage is set slightly upstream to handle
        # numerical inaccuracy in the trigonometric functions that calculates dw_ijlk
        rotor_pos = -1e-10
        blockage = np.zeros_like(deficit)
        if self.blockage_deficitModel is None:
            deficit *= (dw_ijlk > rotor_pos)
        elif (self.blockage_deficitModel != self.wake_deficitModel):
            blockage = self.groundModel(lambda **kwargs: self.rotorAvgModel(self.blockage_deficitModel.calc_blockage_deficit, **kwargs),
                                        dw_ijlk=dw_ijlk, **kwargs)
            deficit *= (dw_ijlk > rotor_pos)
        return deficit, blockage

    def _calc_deficit(self, dw_ijlk, **kwargs):
        """Calculate wake (and blockage) deficit"""
        deficit = self.groundModel(lambda **kwargs: self.rotorAvgModel(self.wake_deficitModel.calc_deficit_downwind, **kwargs),
                                   dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit + blockage

    def _calc_deficit_convection(self, dw_ijlk, **kwargs):
        """Calculate wake convection deficit (and blockage)"""
        deficit, uc, sigma_sqr = self.rotorAvgModel.calc_deficit_convection(
            self.wake_deficitModel, dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit, uc, sigma_sqr, blockage

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
        WS_eff_ilk = lw.WS.ilk((I, L, K)).copy()
        TI_eff_ilk = lw.TI.ilk((I, L, K)).copy()
        if yaw_ilk is None:
            yaw_ilk = np.zeros((I, L, K))
        else:
            yaw_ilk = np.zeros((I, L, K)) + np.deg2rad(yaw_ilk)

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
        return self._calc_wt_interaction(**kwargs) + (lw,)

    @abstractmethod
    def _calc_wt_interaction(self, **kwargs):
        """calculate WT interaction"""

    def _flow_map(self, x_j, y_j, h_j, sim_res_data):
        """call this function via SimulationResult.flow_map"""
        # calculate distances
        wt_d_i = self.windTurbines.diameter(sim_res_data.type)
        wt_x_i, wt_y_i, wt_h_i, wd, ws = [sim_res_data[k] for k in ['x', 'y', 'h', 'wd', 'ws']]

        lw_j = self.site.local_wind(x_i=x_j, y_i=y_j, h_i=h_j, wd=wd, ws=ws)
        I, J, L, K = [len(x) for x in [wt_x_i, x_j, wd, ws]]

        WS_eff_jlk = np.zeros((len(x_j), L, K))
        TI_eff_jlk = np.zeros((len(x_j), L, K))

        for l in tqdm(range(L), disable=L <= 1 or not self.verbose, desc='Calculate flow map', unit='wd'):

            dw_ijl, hcw_ijl, dh_ijl, _ = self.site.distances(wt_x_i, wt_y_i, wt_h_i, x_j, y_j, h_j,
                                                             wd_il=sim_res_data.WD.ilk((I, L, K))[:, l:l + 1, :].mean(2))

            if self.wec != 1:
                hcw_ijl = hcw_ijl / self.wec

            def get_ilk(k):
                def wrap():
                    v = sim_res_data[k].ilk((I, L, K))
                    l_ = [l, 0][v.shape[1] == 1]
                    return v[:, l_][:, na]
                return wrap
            arg_funcs = {'WS_ilk': get_ilk('WS'),
                         'WS_eff_ilk': get_ilk('WS_eff'),
                         'TI_ilk': get_ilk('TI'),
                         'TI_eff_ilk': get_ilk('TI_eff'),
                         'yaw_ilk': lambda: np.deg2rad(get_ilk('Yaw')()),
                         'D_src_il': lambda: wt_d_i[:, na],
                         'D_dst_ijl': lambda: np.zeros_like(dh_ijl),
                         'h_il': lambda: wt_h_i.data[:, na],
                         'ct_ilk': get_ilk('CT')}
            if self.deflectionModel:
                dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                    dw_ijl=dw_ijl, hcw_ijl=hcw_ijl, dh_ijl=dh_ijl,
                    ** {k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})
            else:
                dw_ijlk, hcw_ijlk, dh_ijlk = dw_ijl[..., na], hcw_ijl[..., na], dh_ijl[..., na]
            arg_funcs.update({'cw_ijlk': lambda: np.hypot(dh_ijl[..., na], hcw_ijlk),
                              'dw_ijlk': lambda: dw_ijlk, 'hcw_ijlk': lambda: hcw_ijlk, 'dh_ijlk': lambda: dh_ijlk})

            args = {k: arg_funcs[k]() for k in self.args4deficit if k != 'dw_ijlk'}
            arg_funcs['wake_radius_ijlk'] = lambda: self.wake_deficitModel.wake_radius(dw_ijlk=dw_ijlk, **args)
            if self.turbulenceModel:
                args.update({k: arg_funcs[k]() for k in self.turbulenceModel.args4addturb
                             if k not in self.args4deficit and k != 'dw_ijlk'})

            if I * J * K * 8 / 1024**2 > 10:
                # one wt at the time to avoid memory problems
                deficit_ijk = np.zeros((I, J, K))
                add_turb_ijk = np.zeros((I, J, K))
                uc_ijk = np.zeros((I, J, K))
                sigma_sqr_ijk = np.zeros((I, J, K))
                for i in tqdm(range(I), disable=I <= 1 or not self.verbose,
                              desc="Calculate flow map for wd=%d" % l, unit='wt'):
                    args_i = {k: v[i][na] for k, v in args.items()}
                    if isinstance(self.superpositionModel, WeightedSum):
                        deficit, uc, sigma_sqr, blockage = self._calc_deficit_convection(
                            dw_ijlk=dw_ijlk[i][na], **args_i)
                        deficit_ijk[i] = deficit[0, :, 0]
                        uc_ijk[i] = uc[0, :, 0]
                        sigma_sqr_ijk[i] = sigma_sqr[0, :, 0]
                    else:
                        deficit_ijk[i] = self._calc_deficit(dw_ijlk=dw_ijlk[i][na], **args_i)[0, :, 0]

                    if self.turbulenceModel:
                        add_turb_ijk[i] = self.turbulenceModel.calc_added_turbulence(
                            dw_ijlk=dw_ijlk[i][na], **args_i)[0, :, 0]
            else:
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit, uc, sigma_sqr, blockage = self._calc_deficit_convection(dw_ijlk=dw_ijlk, **args)
                    deficit_ijk = deficit[:, :, 0]
                    uc_ijk = uc[:, :, 0]
                    sigma_sqr_ijk = sigma_sqr[:, :, 0]
                else:
                    deficit_ijk = self._calc_deficit(dw_ijlk=dw_ijlk, **args)[:, :, 0]
                if self.turbulenceModel:
                    add_turb_ijk = self.turbulenceModel.calc_added_turbulence(dw_ijlk=dw_ijlk, **args)[:, :, 0]

            l_ = [l, 0][lw_j.WS_ilk.shape[1] == 1]
            if isinstance(self.superpositionModel, WeightedSum):
                cw_ijk = np.hypot(dh_ijl[..., na], hcw_ijlk)[:, :, 0]
                hcw_ijk, dh_ijk = hcw_ijlk[:, :, 0], dh_ijl[:, :, 0, na]
                WS_eff_jlk[:, l] = self.superpositionModel.calc_effective_WS(
                    lw_j.WS_ilk[:, l_], deficit_ijk, uc_ijk, sigma_sqr_ijk, cw_ijk, hcw_ijk, dh_ijk)
            else:
                WS_eff_jlk[:, l] = self.superpositionModel.calc_effective_WS(lw_j.WS_ilk[:, l_], deficit_ijk)

            if self.turbulenceModel:
                l_ = [l, 0][lw_j.TI_ilk.shape[1] == 1]
                TI_eff_jlk[:, l] = self.turbulenceModel.calc_effective_TI(lw_j.TI_ilk[:, l_], add_turb_ijk)
        return lw_j, WS_eff_jlk, TI_eff_jlk

    def _validate_input(self, dw_iil, hcw_iil):
        I_ = dw_iil.shape[0]
        i1, i2, _ = np.where((np.abs(dw_iil) + np.abs(hcw_iil) + np.eye(I_)[:, :, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" %
                             (i1[i], i2[i]) for i in range(len(i1))])
            raise ValueError(msg)

    def dAEPdn(self, argnum, gradient_method):
        def aep(x, y, h=None, type=0, wd=None, ws=None, yaw_ilk=None):  # @ReservedAssignment
            if gradient_method == autograd:
                with use_autograd_in():
                    return self.aep(x, y, h, type, wd, ws, yaw_ilk)
            else:
                return self.aep(x, y, h, type, wd, ws, yaw_ilk)
        return gradient_method(aep, argnum)

    def dAEPdxy(self, gradient_method, normalize_probabilities=False, with_wake_loss=True, gradient_method_kwargs={}):

        def wrap(x, y, h=None, type=0, wd=None, ws=None, yaw_ilk=None):  # @ReservedAssignment
            def aep(x, y, h, type, wd, ws, yaw_ilk):  # @ReservedAssignment
                if gradient_method == autograd:
                    with use_autograd_in():
                        return self.aep(x, y, h, type, wd, ws, yaw_ilk,
                                        normalize_probabilities=normalize_probabilities, with_wake_loss=with_wake_loss)
                else:
                    return self.aep(x, y, h, type, wd, ws, yaw_ilk,
                                    normalize_probabilities=normalize_probabilities, with_wake_loss=with_wake_loss)
            return (gradient_method(aep, 0, **gradient_method_kwargs)(x, y, h, type, wd, ws, yaw_ilk),
                    gradient_method(aep, 1, **gradient_method_kwargs)(x, y, h, type, wd, ws, yaw_ilk))
        return wrap


class PropagateDownwind(EngineeringWindFarmModel):
    """Downstream wake deficits calculated and propagated in downstream direction.
    Very fast, but ignoring blockage effects
    """

    def __init__(self, site, windTurbines, wake_deficitModel,
                 rotorAvgModel=RotorCenter(), superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=None,
                 groundModel=None):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.
            Default is RotorCenter, i.e. one point at rotor center
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, rotorAvgModel, superpositionModel,
                                          blockage_deficitModel=None, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, groundModel=groundModel)

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
        deficit_nk = []
        uc_nk = []
        sigma_sqr_nk = []
        cw_nk = []
        hcw_nk = []
        dh_nk = []

        def ilk2mk(x_ilk):
            return np.broadcast_to(x_ilk.astype(np.float), (I, L, K)).reshape((I * L, K))

        indices = np.arange(I * I * L).reshape((I, I, L))
        TI_mk = ilk2mk(lw.TI_ilk)
        WS_mk = ilk2mk(lw.WS_ilk)
        WS_eff_mk = []
        TI_eff_mk = []
        yaw_mk = ilk2mk(yaw_ilk)
        power_jlk = []
        ct_jlk = []

        if not self.deflectionModel:
            dw_ijlk, hcw_ijlk, dh_ijlk = dw_iil[..., na], hcw_iil[..., na], dh_iil[..., na]

        if self.turbulenceModel:
            add_turb_nk = np.zeros((I * I * L, K))

        def iil2n(iil):
            if isinstance(iil, np.ndarray):
                iil.resize(I * I * L)
                return iil
            else:
                # In case of autograd array box
                return iil.flatten()

        dw_n, hcw_n, dh_n = [iil2n(a) for a in [dw_iil, hcw_iil, dh_iil]]

        i_wd_l = np.arange(L)

        # Iterate over turbines in down wind order
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose, desc="Calculate flow interaction", unit="wt"):
            i_wt_l = dw_order_indices_dl[:, j]
            m = i_wt_l * L + i_wd_l  # current wt (j'th most upstream wts for all wdirs)

            # generate indexes of up wind(n_uw) and down wind(n_dw) turbines
            n_uw = indices[:, i_wt_l, i_wd_l][dw_order_indices_dl[:, :j].T, np.arange(L)]
            n_dw = indices[i_wt_l, :, i_wd_l][np.arange(L), dw_order_indices_dl[:, j + 1:].T]

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
                WS_eff_mk.append(WS_eff_lk)
                if self.turbulenceModel:
                    TI_eff_mk.append(TI_mk[m])
            else:  # 2..n most upstream turbines (wake)
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit2WT = np.array([d_nk2[i] for d_nk2, i in zip(deficit_nk, range(j)[::-1])])
                    uc2WT = np.array([d_nk2[i] for d_nk2, i in zip(uc_nk, range(j)[::-1])])
                    sigmasqr2WT = np.array([d_nk2[i] for d_nk2, i in zip(sigma_sqr_nk, range(j)[::-1])])
                    cw2WT = np.array([d_nk2[i] for d_nk2, i in zip(cw_nk, range(j)[::-1])])
                    hcw2WT = np.array([d_nk2[i] for d_nk2, i in zip(hcw_nk, range(j)[::-1])])
                    dh2WT = np.array([d_nk2[i] for d_nk2, i in zip(dh_nk, range(j)[::-1])])

                    WS_eff_lk = self.superpositionModel.calc_effective_WS(
                        WS_mk[m], deficit2WT, uc2WT, sigmasqr2WT, cw2WT, hcw2WT, dh2WT)
                else:
                    deficit2WT = np.array([d_nk2[i] for d_nk2, i in zip(deficit_nk, range(j)[::-1])])
                    WS_eff_lk = self.superpositionModel.calc_effective_WS(WS_mk[m], deficit2WT)

                WS_eff_mk.append(WS_eff_lk)
                if self.turbulenceModel:
                    TI_eff_mk.append(self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb_nk[n_uw]))

            ct_lk, power_lk = self.windTurbines._ct_power(WS_eff_lk, type_i[i_wt_l], yaw_ilk[i_wt_l, i_wd_l])

            power_jlk.append(power_lk)
            ct_jlk.append(ct_lk)

            if j < I - 1:

                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_eff_ilk': lambda: WS_eff_mk[-1][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[-1][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'D_dst_ijl': lambda: D_i[dw_order_indices_dl[:, j + 1:]].T[na],
                             'h_il': lambda: h_i[i_wt_l][na],
                             'ct_ilk': lambda: ct_lk[na],
                             'wake_radius_ijlk': lambda: wake_radius_ijlk
                             }

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijl=dw_n[n_dw][na], hcw_ijl=hcw_n[n_dw][na], dh_ijl=dh_n[n_dw][na],
                        ** {k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})

                else:
                    dw_ijlk, hcw_ijlk, dh_ijlk = [v[n_dw][na, :, :, na] for v in [dw_n, hcw_n, dh_n]]

                # sqrt(a**2+b**2) as hypot does not support complex numbers
                cw_ijlk = np.sqrt(dh_ijlk**2 + hcw_ijlk**2)

                arg_funcs.update({'hcw_ijlk': lambda: hcw_ijlk, 'cw_ijlk': lambda: cw_ijlk, 'dh_ijlk': lambda: dh_ijlk})
                args = {k: arg_funcs[k]() for k in self.args4deficit if k != "dw_ijlk"}
                hcw_nk.append(hcw_ijlk[0])
                dh_nk.append(dh_ijlk[0])
                cw_nk.append(cw_ijlk[0])

                # Calculate deficit
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit, uc, sigma_sqr, blockage = self._calc_deficit_convection(dw_ijlk=dw_ijlk, **args)
                    # deficit_nk.append(deficit[0])
                    deficit += blockage
                    uc_nk.append(uc[0])
                    sigma_sqr_nk.append(sigma_sqr[0])
                else:
                    deficit = self._calc_deficit(dw_ijlk=dw_ijlk, **args)
                deficit_nk.append(deficit[0])

                if self.turbulenceModel:

                    if 'wake_radius_ijlk' in self.args4addturb:
                        wake_radius_ijlk = self.wake_deficitModel.wake_radius(dw_ijlk=dw_ijlk, **args)
                        arg_funcs['wake_radius_ijlk'] = lambda: wake_radius_ijlk

                    turb_args = {k: arg_funcs[k]() for k in self.args4addturb
                                 if k != "dw_ijlk"}

                    # Calculate added turbulence
                    add_turb_nk[n_dw] = self.turbulenceModel.rotorAvgModel(self.turbulenceModel.calc_added_turbulence,
                                                                           dw_ijlk=dw_ijlk, **turb_args)

        WS_eff_jlk, power_jlk, ct_jlk = np.array(WS_eff_mk), np.array(power_jlk), np.array(ct_jlk)

        dw_inv_indices = (np.argsort(dw_order_indices_dl, 1).T * L + np.arange(L)[na]).flatten()
        WS_eff_ilk = WS_eff_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))

        power_ilk = power_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))
        ct_ilk = ct_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_ilk = np.reshape(TI_eff_mk, (I * L, K))[dw_inv_indices].reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk


class All2AllIterative(EngineeringWindFarmModel):
    """Wake and blockage deficits calculated from all wt to all points of interest (wt/map points).
    The calculations are iteratively repeated until convergence (change of effective wind speed < convergence_tolerance)"""

    def __init__(self, site, windTurbines, wake_deficitModel,
                 rotorAvgModel=RotorCenter(), superpositionModel=LinearSum(),
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 groundModel=None,
                 convergence_tolerance=1e-6):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        rotorAvgModel : RotorAvgModel
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            Defaults to RotorCenter that uses the rotor center wind speed (i.e. one point) only
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
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, rotorAvgModel, superpositionModel,
                                          blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, groundModel=groundModel)
        self.convergence_tolerance = convergence_tolerance

    def _calc_wt_interaction(self, localWind,
                             WS_eff_ilk, TI_eff_ilk,
                             type_i, h_i, D_i, yaw_ilk,
                             dw_iil, hcw_iil, cw_iil, dh_iil, dw_order_indices_dl, I, L, K):
        lw = localWind
        power_ilk = np.zeros((I, L, K))
        WS_eff_ilk_last = WS_eff_ilk.copy()

        ct_ilk = self.windTurbines.ct(lw.WS.ilk((I, L, K)), type_i)
        D_src_il = D_i[:, na]
        args = {'WS_ilk': lw.WS.ilk((I, L, K)),
                'TI_ilk': lw.TI.ilk((I, L, K)),
                'TI_eff_ilk': lw.TI.ilk((I, L, K)),
                'yaw_ilk': yaw_ilk,
                'D_src_il': D_src_il,
                'D_dst_ijl': D_src_il[na],
                'cw_ijlk': cw_iil[..., na],
                'dh_ijlk': dh_iil[..., na],
                'h_il': h_i[:, na]
                }

        # Iterate until convergence
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose, desc="Calculate flow interaction", unit="wt"):

            ct_ilk, power_ilk = self.windTurbines._ct_power(WS_eff_ilk, type_i, yaw_ilk)
            args['ct_ilk'] = ct_ilk
            args['WS_eff_ilk'] = WS_eff_ilk
            if self.deflectionModel:
                dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                    dw_ijl=dw_iil, hcw_ijl=dw_iil, dh_ijl=dh_iil, **args)
                args.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                             'cw_ijlk': np.hypot(dh_iil[..., na], hcw_ijlk)})
            else:
                args.update({'dw_ijlk': dw_iil[..., na], 'hcw_ijlk': hcw_iil[..., na], 'dh_ijlk': dh_iil[..., na]})
                self._init_deficit(**args)
            if self.turbulenceModel:
                args['TI_eff_ilk'] = TI_eff_ilk
                if 'wake_radius_ijlk' in self.turbulenceModel.args4addturb:
                    args['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**args)

            # Calculate deficit
            if isinstance(self.superpositionModel, WeightedSum):
                deficit_iilk, uc_iilk, sigmasqr_iilk, blockage_iilk = self._calc_deficit_convection(**args)
            else:
                deficit_iilk = self._calc_deficit(**args)

            # Calculate effective wind speed
            if isinstance(self.superpositionModel, WeightedSum):
                WS_eff_ilk = self.superpositionModel.calc_effective_WS(lw.WS_ilk, deficit_iilk,
                                                                       uc_iilk, sigmasqr_iilk,
                                                                       args['cw_ijlk'],
                                                                       args['hcw_ijlk'],
                                                                       dh_iil[..., na])
                # Add blockage as linear effect
                WS_eff_ilk -= np.sum(blockage_iilk, 0)
            else:
                WS_eff_ilk = self.superpositionModel.calc_effective_WS(lw.WS_ilk, deficit_iilk)

            if self.turbulenceModel:
                add_turb_ijlk = self.turbulenceModel.rotorAvgModel(self.turbulenceModel.calc_added_turbulence, **args)
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
        from py_wake.deficit_models.gaussian import ZongGaussianDeficit
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake.flow_map import XYGrid
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

        # Zong convection superposition
        zongp_ss = PropagateDownwind(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=WeightedSum(),
                                     turbulenceModel=STF2017TurbulenceModel())

        # Zong convection superposition
        zong_ss = All2AllIterative(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=WeightedSum(),
                                   blockage_deficitModel=SelfSimilarityDeficit(), turbulenceModel=STF2017TurbulenceModel())

        for wm in [noj, noj_ss, zongp_ss, zong_ss]:
            sim = wm(x=x, y=y, wd=[30], ws=[9])
            plt.figure()
            sim.flow_map(XYGrid(resolution=200)).plot_wake_map()
            plt.title(' AEP: %.3f GWh' % sim.aep().sum())
        plt.show()


main()
