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
from py_wake.deficit_models.deficit_model import ConvectionDeficitModel, BlockageDeficitModel, WakeDeficitModel
from py_wake.ground_models.ground_models import NoGround, GroundModel
from tqdm import tqdm
from py_wake.wind_turbines._wind_turbines import WindTurbine, WindTurbines
from py_wake.utils.model_utils import check_model
from py_wake.utils.functions import mean_deg


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

    def __init__(self, site, windTurbines: WindTurbines, wake_deficitModel, rotorAvgModel, superpositionModel,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 groundModel=None):

        WindFarmModel.__init__(self, site, windTurbines)
        check_model(wake_deficitModel, WakeDeficitModel, 'wake_deficitModel')
        check_model(rotorAvgModel, RotorAvgModel, 'rotorAvgModel')
        check_model(superpositionModel, SuperpositionModel, 'superpositionModel')
        check_model(blockage_deficitModel, BlockageDeficitModel, 'blockage_deficitModel')
        check_model(deflectionModel, DeflectionModel, 'deflectionModel')
        check_model(turbulenceModel, TurbulenceModel, 'turbulenceModel')
        if groundModel is None:
            groundModel = NoGround()
        check_model(groundModel, GroundModel, 'groundModel')
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
        return deficit, blockage

    def _calc_deficit_convection(self, dw_ijlk, **kwargs):
        """Calculate wake convection deficit (and blockage)"""
        deficit, uc, sigma_sqr = self.rotorAvgModel.calc_deficit_convection(
            self.wake_deficitModel, dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit, uc, sigma_sqr, blockage

    def calc_wt_interaction(self, x_i, y_i, h_i=None, type_i=0, wd=None, ws=None, time=False,
                            yaw_ilk=None, tilt_ilk=None, **kwargs):
        """See WindFarmModel.calc_wt_interaction"""
        h_i, D_i = self.windTurbines.get_defaults(len(x_i), type_i, h_i)
        x_i, y_i, type_i = [np.asarray(v) for v in [x_i, y_i, type_i]]
        wd, ws = self.site.get_defaults(wd, ws)

        # Find local wind speed, wind direction, turbulence intensity and probability
        lw = self.site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws, time=time)

        # Calculate down-wind and cross-wind distances
        self._validate_input(x_i, y_i)

        I, L, K, = len(x_i), len(wd), (1, len(ws))[time is False]
        for v in ['WS', 'WD', 'TI']:
            if v in kwargs:
                lw.add_ilk(v, kwargs[v])

        WS_eff_ilk = lw.WS.ilk((I, L, K)).copy()
        TI_eff_ilk = lw.TI.ilk((I, L, K)).copy()

        # add eps to avoid non-differentiable 0
#        eps = 2 * np.finfo(float).eps ** 2 if 'autograd' in np.__name__ else 0

        self.site.distance.setup(x_i, y_i, h_i)
        # cw_iil = np.sqrt(hcw_iil**2 + dh_iil**2 + eps)
        wt_kwargs = kwargs
        ri, oi = self.windTurbines.function_inputs
        unused_inputs = set(wt_kwargs) - set(ri) - set(oi) - {'WS', 'WD', 'TI'}
        if unused_inputs:
            raise TypeError("""got unexpected keyword argument(s): '%s'
            required arguments: %s
            optional arguments: %s""" % ("', '".join(unused_inputs), ['ws'] + ri, oi))

        def arg2ilk(k, v):
            #             if v is None:
            #                 return v
            v = np.asarray(v)
            if v.shape not in {(), (I,), (I, L), (I, L, K), (L,), (L, K)}:
                valid_shapes = f"(), ({I}), ({I},{L}), ({I},{L},{K})"
                raise ValueError(
                    f"Argument, {k}(shape={v.shape}), has unsupported shape. Valid shapes are {valid_shapes}")
            if v.shape == (L,) or v.shape == (L, K):
                return np.broadcast_to(v[na], (I,) + v.shape)
            else:
                return v
        wt_kwargs = {k: arg2ilk(k, v)for k, v in wt_kwargs.items()}

        def add_arg(name, optional):
            if name in wt_kwargs:  # custom WindFarmModel.__call__ arguments
                return
            elif name in {'yaw', 'tilt', 'type'}:  # fixed WindFarmModel.__call__ arguments
                wt_kwargs[name] = {'yaw': yaw_ilk, 'tilt': tilt_ilk, 'type': type_i}[name]
            elif name in lw:
                wt_kwargs[name] = lw[name].values
            elif name in self.site.ds:
                wt_kwargs[name] = self.site.interp(self.site.ds[name], lw.coords).values
            elif name in ['TI_eff']:
                if self.turbulenceModel:
                    wt_kwargs['TI_eff'] = None
                elif optional is False:
                    raise KeyError("Argument, TI_eff, needed to calculate power and ct requires a TurbulenceModel")
            elif name in ['dw_ijl', 'cw_ijl', 'hcw_ijl']:
                pass
            elif optional:
                pass
            else:
                raise KeyError("Argument, %s, required to calculate power and ct not found" % name)
        for opt, lst in zip([False, True], self.windTurbines.function_inputs):
            for k in lst:
                add_arg(k, opt)

        if yaw_ilk is None:
            yaw_ilk = np.zeros((I, L, K))
        if tilt_ilk is None:
            tilt_ilk = np.zeros((I, L, K))

        kwargs = {'localWind': lw,
                  'WS_eff_ilk': WS_eff_ilk, 'TI_eff_ilk': TI_eff_ilk,
                  'x_i': x_i, 'y_i': y_i, 'h_i': h_i, 'D_i': D_i,
                  'yaw_ilk': yaw_ilk, 'tilt_ilk': tilt_ilk,
                  'I': I, 'L': L, 'K': K, **wt_kwargs}
        WS_eff_ilk, TI_eff_ilk, ct_ilk = self._calc_wt_interaction(**kwargs)
        if 'TI_eff' in wt_kwargs:
            wt_kwargs['TI_eff'] = TI_eff_ilk
        d_ijl_keys = ({k for l in self.windTurbines.function_inputs for k in l} &
                      {'dw_ijl', 'hcw_ijl', 'dh_ijl', 'cw_ijl'})
        if d_ijl_keys:
            d_ijl_dict = {k: lambda v=v: v for k, v in zip(['dw_ijl', 'hcw_ijl', 'dh_ijl'], self.site.distance(wd[na]))}
            d_ijl_dict['cw_ijl'] = lambda d_ijl_dict=d_ijl_dict: np.sqrt(
                d_ijl_dict['dw_ijl']**2 + d_ijl_dict['hcw_ijl']**2)
            wt_kwargs.update({k: d_ijl_dict[k]() for k in d_ijl_keys})

        wt_kwargs_keys = set(self.windTurbines.powerCtFunction.required_inputs +
                             self.windTurbines.powerCtFunction.optional_inputs)
        power_ilk = self.windTurbines.power(WS_eff_ilk, **{k: v for k, v in wt_kwargs.items() if k in wt_kwargs_keys})

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, lw, wt_kwargs

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

        self.site.distance.setup(wt_x_i, wt_y_i, wt_h_i, (x_j, y_j, h_j))

        for l in tqdm(range(L), disable=L <= 1 or not self.verbose, desc='Calculate flow map', unit='wd'):

            dw_ijl, hcw_ijl, dh_ijl = self.site.distance(wd_il=sim_res_data.WD.ilk((I, L, K))[:, l:l + 1, :].mean(2))

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
                         'yaw_ilk': get_ilk('yaw'),
                         'tilt_ilk': get_ilk('tilt'),
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
            arg_funcs.update({'cw_ijlk': lambda: np.hypot(dh_ijlk, hcw_ijlk),
                              'dw_ijlk': lambda: dw_ijlk, 'hcw_ijlk': lambda: hcw_ijlk, 'dh_ijlk': lambda: dh_ijlk})

            args = {k: arg_funcs[k]() for k in self.args4deficit if k != 'dw_ijlk'}
            arg_funcs['wake_radius_ijlk'] = lambda: self.wake_deficitModel.wake_radius(dw_ijlk=dw_ijlk, **args)
            if self.turbulenceModel:
                args.update({k: arg_funcs[k]() for k in self.turbulenceModel.args4addturb
                             if k not in self.args4deficit and k != 'dw_ijlk'})

            if I * J * K * 8 / 1024**2 > 10:
                # one wt at the time to avoid memory problems
                deficit_ijk = np.zeros((I, J, K))
                blockage_ijk = np.zeros((I, J, K))
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

                        deficit_ijk[i], blockage_ijk[i] = [v[0, :, 0]
                                                           for v in self._calc_deficit(dw_ijlk=dw_ijlk[i][na], **args_i)]

                    if self.turbulenceModel:
                        add_turb_ijk[i] = self.turbulenceModel.calc_added_turbulence(
                            dw_ijlk=dw_ijlk[i][na], **args_i)[0, :, 0]
            else:
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit, uc, sigma_sqr, blockage = self._calc_deficit_convection(dw_ijlk=dw_ijlk, **args)
                    deficit_ijk = deficit[:, :, 0]
                    blockage_ijk = blockage[:, :, 0]
                    uc_ijk = uc[:, :, 0]
                    sigma_sqr_ijk = sigma_sqr[:, :, 0]
                else:
                    deficit_ijk, blockage_ijk = self._calc_deficit(dw_ijlk=dw_ijlk, **args)
                    deficit_ijk, blockage_ijk = deficit_ijk[:, :, 0], blockage_ijk[:, :, 0]
                if self.turbulenceModel:
                    add_turb_ijk = self.turbulenceModel.calc_added_turbulence(dw_ijlk=dw_ijlk, **args)[:, :, 0]

            l_ = [l, 0][lw_j.WS_ilk.shape[1] == 1]
            if isinstance(self.superpositionModel, WeightedSum):
                cw_ijk = np.hypot(dh_ijl[..., na], hcw_ijlk)[:, :, 0]
                hcw_ijk, dh_ijk = hcw_ijlk[:, :, 0], dh_ijl[:, :, 0, na]
                WS_eff_jlk[:, l] = lw_j.WS_ilk[:, l_] - self.superpositionModel(
                    lw_j.WS_ilk[:, l_], deficit_ijk, uc_ijk, sigma_sqr_ijk, cw_ijk, hcw_ijk, dh_ijk)
                if self.blockage_deficitModel:
                    blockage_superpositionModel = self.blockage_deficitModel.superpositionModel or LinearSum()
                    WS_eff_jlk[:, l] -= blockage_superpositionModel(blockage_ijk)
            else:
                WS_eff_jlk[:, l] = lw_j.WS_ilk[:, l_] - self.superpositionModel(deficit_ijk)
                if self.blockage_deficitModel:
                    blockage_superpositionModel = self.blockage_deficitModel.superpositionModel or self.superpositionModel
                    WS_eff_jlk[:, l] -= blockage_superpositionModel(blockage_ijk)

            if self.turbulenceModel:
                l_ = [l, 0][lw_j.TI_ilk.shape[1] == 1]
                TI_eff_jlk[:, l] = self.turbulenceModel.calc_effective_TI(lw_j.TI_ilk[:, l_], add_turb_ijk)
        return lw_j, WS_eff_jlk, TI_eff_jlk

    def _validate_input(self, x_i, y_i):
        i1, i2 = np.where((np.abs(x_i[:, na] - x_i[na]) + np.abs(y_i[:, na] - y_i[na]) + np.eye(len(x_i))) == 0)
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
        return gradient_method(aep, True, argnum)

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
            return (gradient_method(aep, True, 0, **gradient_method_kwargs)(x, y, h, type, wd, ws, yaw_ilk),
                    gradient_method(aep, True, 1, **gradient_method_kwargs)(x, y, h, type, wd, ws, yaw_ilk))
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
                             x_i, y_i, h_i, D_i, yaw_ilk, tilt_ilk,
                             I, L, K, **kwargs):
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
            return np.broadcast_to(x_ilk.astype(float), (I, L, K)).reshape((I * L, K))

        indices = np.arange(I * I * L).reshape((I, I, L))

        TI_mk = ilk2mk(lw.TI_ilk)
        WS_mk = ilk2mk(lw.WS_ilk)
        WS_eff_mk = []
        TI_eff_mk = []
        yaw_mk = ilk2mk(yaw_ilk)
        tilt_mk = ilk2mk(tilt_ilk)
        ct_jlk = []

        if self.turbulenceModel:
            add_turb_nk = np.zeros((I * I * L, K))

        i_wd_l = np.arange(L)
        wd = mean_deg(lw.WD_ilk, (0, 2))
        dw_order_indices_dl = self.site.distance.dw_order_indices(wd)

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

                    WS_eff_lk = WS_mk[m] - self.superpositionModel(
                        WS_mk[m], deficit2WT, uc2WT, sigmasqr2WT, cw2WT, hcw2WT, dh2WT)
                else:
                    deficit2WT = np.array([d_nk2[i] for d_nk2, i in zip(deficit_nk, range(j)[::-1])])
                    WS_eff_lk = WS_mk[m] - self.superpositionModel(deficit2WT)

                WS_eff_mk.append(WS_eff_lk)
                if self.turbulenceModel:
                    TI_eff_mk.append(self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb_nk[n_uw]))

            # Calculate Power/CT
            def mask(k, v):
                if v is None or isinstance(v, (int, float)) or len(np.asarray(v).shape) == 0:
                    return v
                v = np.asarray(v)
                if len(v.shape) == 1 and len(v) == I:
                    return v[i_wt_l]
                elif v.shape[:2] == (I, L):
                    return v[i_wt_l, i_wd_l]
#                 elif v.shape == (L,):
#                     return v[i_wd_l]
#                 else:
#                     valid_shapes = f"(), ({I}), ({I},{L}), ({I},{L},{K}), ({L}), ({L},{K})"
#                     raise ValueError(
#                         f"Argument, {k}(shape={v.shape}), has unsupported shape. Valid shapes are {valid_shapes}")
            keys = self.windTurbines.powerCtFunction.required_inputs + self.windTurbines.powerCtFunction.optional_inputs
            _kwargs = {k: mask(k, v) for k, v in kwargs.items() if k in keys}
            if 'TI_eff' in _kwargs:
                _kwargs['TI_eff'] = TI_eff_mk[-1]
            ct_lk = self.windTurbines.ct(WS_eff_lk, **_kwargs)

            ct_jlk.append(ct_lk)

            if j < I - 1:

                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_eff_ilk': lambda: WS_eff_mk[-1][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[-1][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'tilt_ilk': lambda: tilt_mk[m][na],
                             'D_dst_ijl': lambda: D_i[dw_order_indices_dl[:, j + 1:]].T[na],
                             'h_il': lambda: h_i[i_wt_l][na],
                             'ct_ilk': lambda: ct_lk[na],
                             'wake_radius_ijlk': lambda: wake_radius_ijlk
                             }

                i_dw = dw_order_indices_dl[:, j + 1:]

                dw_jl, hcw_jl, dh_jl = self.site.distance(wd, src_idx=i_wt_l, dst_idx=i_dw.T)
                if self.wec != 1:
                    hcw_jl = hcw_jl / self.wec

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijl=dw_jl[na], hcw_ijl=hcw_jl[na], dh_ijl=dh_jl[na],
                        ** {k: arg_funcs[k]() for k in self.deflectionModel.args4deflection})
                else:
                    dw_ijlk, hcw_ijlk, dh_ijlk = [v[na, :, :, na] for v in [dw_jl, hcw_jl, dh_jl]]

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
                    deficit += blockage
                    uc_nk.append(uc[0])
                    sigma_sqr_nk.append(sigma_sqr[0])
                else:
                    deficit, _ = self._calc_deficit(dw_ijlk=dw_ijlk, **args)
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

        WS_eff_jlk, ct_jlk = np.array(WS_eff_mk), np.array(ct_jlk)

        dw_inv_indices = (np.argsort(dw_order_indices_dl, 1).T * L + np.arange(L)[na]).flatten()
        WS_eff_ilk = WS_eff_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))

        ct_ilk = ct_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_ilk = np.reshape(TI_eff_mk, (I * L, K))[dw_inv_indices].reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, ct_ilk


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
                             x_i, y_i, h_i, D_i, yaw_ilk, tilt_ilk,
                             I, L, K, **kwargs):
        lw = localWind
        WS_eff_ilk_last = WS_eff_ilk.copy()
        dw_iil, hcw_iil, dh_iil = self.site.distance(mean_deg(lw.WD_ilk, 2))

        ct_ilk = self.windTurbines.ct(lw.WS.ilk((I, L, K)), **kwargs)
        D_src_il = D_i[:, na]
        args = {'WS_ilk': lw.WS.ilk((I, L, K)),
                'TI_ilk': lw.TI.ilk((I, L, K)),
                'TI_eff_ilk': lw.TI.ilk((I, L, K)),
                'yaw_ilk': yaw_ilk,
                'tilt_ilk': tilt_ilk,
                'D_src_il': D_src_il,
                'D_dst_ijl': D_src_il[na],
                'dw_ijlk': dw_iil[..., na],
                'hcw_ijlk': hcw_iil[..., na],
                'cw_ijlk': np.sqrt(hcw_iil**2 + dh_iil**2)[..., na],
                'dh_ijlk': dh_iil[..., na],
                'h_il': h_i[:, na]
                }

        # Iterate until convergence
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose, desc="Calculate flow interaction", unit="wt"):

            ct_ilk = self.windTurbines.ct(np.maximum(WS_eff_ilk, 0), **kwargs)
            args['ct_ilk'] = ct_ilk
            args['WS_eff_ilk'] = WS_eff_ilk
            if self.deflectionModel:
                dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                    dw_ijl=dw_iil, hcw_ijl=hcw_iil, dh_ijl=dh_iil, **args)
                args.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                             'cw_ijlk': np.hypot(dh_iil[..., na], hcw_ijlk)})
                self._reset_deficit()
            elif j == 0:
                self._init_deficit(**args)

            if self.turbulenceModel:
                args['TI_eff_ilk'] = TI_eff_ilk
                if 'wake_radius_ijlk' in self.turbulenceModel.args4addturb:
                    args['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**args)

            # Calculate deficit
            if isinstance(self.superpositionModel, WeightedSum):
                deficit_iilk, uc_iilk, sigmasqr_iilk, blockage_iilk = self._calc_deficit_convection(**args)
            else:
                deficit_iilk, blockage_iilk = self._calc_deficit(**args)

            # Calculate effective wind speed
            if isinstance(self.superpositionModel, WeightedSum):
                WS_eff_ilk = lw.WS_ilk - self.superpositionModel(lw.WS_ilk, deficit_iilk,
                                                                 uc_iilk, sigmasqr_iilk,
                                                                 args['cw_ijlk'],
                                                                 args['hcw_ijlk'],
                                                                 dh_iil[..., na])
                # Add blockage as linear effect
                if self.blockage_deficitModel:
                    WS_eff_ilk -= (self.blockage_deficitModel.superpositionModel or LinearSum())(blockage_iilk)
            else:
                WS_eff_ilk = lw.WS_ilk.astype(float) - self.superpositionModel(deficit_iilk)
                if self.blockage_deficitModel:
                    WS_eff_ilk -= (self.blockage_deficitModel.superpositionModel or self.superpositionModel)(blockage_iilk)

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
        return WS_eff_ilk, TI_eff_ilk, ct_ilk


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
