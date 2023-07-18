from abc import abstractmethod
from numpy import newaxis as na
from py_wake import np
from py_wake.superposition_models import SuperpositionModel, LinearSum, WeightedSum
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.utils.gradients import cabs
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel, RotorCenter
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.deficit_models.deficit_model import ConvectionDeficitModel, BlockageDeficitModel, WakeDeficitModel
from tqdm import tqdm
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.utils.model_utils import check_model, fix_shape
from py_wake.utils.functions import mean_deg
from py_wake.utils.gradients import hypot
import warnings
from py_wake.input_modifier_models.input_modifier_model import InputModifierModel


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

    def __init__(self, site, windTurbines: WindTurbines, wake_deficitModel, superpositionModel, rotorAvgModel=None,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None, inputModifierModels=[]):

        WindFarmModel.__init__(self, site, windTurbines)
        if not isinstance(inputModifierModels, (list, tuple)):
            inputModifierModels = [inputModifierModels]
        for model, cls, name in ([(wake_deficitModel, WakeDeficitModel, 'wake_deficitModel'),
                                  (superpositionModel, SuperpositionModel, 'superpositionModel'),
                                  (blockage_deficitModel, BlockageDeficitModel, 'blockage_deficitModel'),
                                  (deflectionModel, DeflectionModel, 'deflectionModel'),
                                  (turbulenceModel, TurbulenceModel, 'turbulenceModel')] +
                                 [(imm, InputModifierModel, 'inputModificierModels') for imm in inputModifierModels]):
            check_model(model, cls, name)
            if model is not None:
                setattr(model, 'windFarmModel', self)
            setattr(self, name, model)
        self.inputModifierModels = inputModifierModels

        if isinstance(superpositionModel, WeightedSum):
            assert isinstance(wake_deficitModel, ConvectionDeficitModel)
            assert rotorAvgModel is None or isinstance(rotorAvgModel, RotorCenter), \
                "WeightedSum only works with RotorCenter"
        # TI_eff requires a turbulence model
        assert 'TI_eff_ilk' not in wake_deficitModel.args4deficit or turbulenceModel
        self.wake_deficitModel = wake_deficitModel
        if rotorAvgModel is not None:
            warnings.warn("""The rotorAvgModel-argument of WindFarmModel is ambiguous and therefore deprecated.
            Set the rotorAvgModel of the wake_deficitModel, the blockage_deficitModel and/or turbulenceModel instead.
            Until removed, the rotorAvgModel of WindFarmModel will apply the rotorAvgModel to the wake_deficitModel only
            if a rotorAvgModel has not already been specified for the wake_deficitModel""",
                          DeprecationWarning, stacklevel=2)
            check_model(rotorAvgModel, RotorAvgModel, 'rotorAvgModel')
            self.wake_deficitModel.rotorAvgModel = self.wake_deficitModel.rotorAvgModel or rotorAvgModel

        self.superpositionModel = superpositionModel
        self.blockage_deficitModel = blockage_deficitModel
        self.deflectionModel = deflectionModel
        self.turbulenceModel = turbulenceModel

        # wake expansion continuation (wake-width scale factor) see
        self.wec = 1
        # Thomas, J. J. and Ning, A., "A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,"
        # Journal of Physics: Conference Series, Vol. 1037, The Science of Making
        # Torque from Wind, Milano, Italy, jun 2018, p. 10.
        self.deficit_initalized = False

        self.args4deficit = self.wake_deficitModel.args4deficit
        # self.args4deficit = set(self.args4deficit) | {'yaw_ilk'}
        if self.blockage_deficitModel:
            self.args4deficit = set(self.args4deficit) | set(self.blockage_deficitModel.args4deficit)
        self.args4all = set(self.args4deficit)
        if self.turbulenceModel:
            self.args4all |= set(self.turbulenceModel.args4model)
        if self.deflectionModel:
            self.args4all |= set(self.deflectionModel.args4deflection)
        for input_modifier in self.inputModifierModels:
            self.args4all |= set(input_modifier.args4model)

    def __str__(self):
        def name(o):
            return o.__class__.__name__

        models = [self.__class__.__bases__[0].__name__, "%s-wake" % name(self.wake_deficitModel)]
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
        self.wake_deficitModel.calc_layout_terms(**kwargs)
        self.wake_deficitModel.deficit_initalized = True
        if self.blockage_deficitModel:
            if self.blockage_deficitModel != self.wake_deficitModel:
                self.blockage_deficitModel.calc_layout_terms(**kwargs)
            self.blockage_deficitModel.deficit_initalized = True

    def _reset_deficit(self):
        self.wake_deficitModel.deficit_initalized = False
        if self.blockage_deficitModel:
            self.blockage_deficitModel.deficit_initalized = False

    def _add_blockage(self, deficit, dw_ijlk, **kwargs):
        # the split line between wake and blockage is set slightly upstream to handle
        # numerical inaccuracy in the trigonometric functions that calculates dw_ijlk
        rotor_pos = -1e-10
        if self.blockage_deficitModel is None:
            deficit *= (dw_ijlk > rotor_pos)
            blockage = None
        elif (self.blockage_deficitModel != self.wake_deficitModel):
            blockage = self.blockage_deficitModel.calc_blockage_deficit(dw_ijlk=dw_ijlk, **kwargs)
            deficit *= (dw_ijlk > rotor_pos)
        else:
            # Same model for both wake and blockage
            # keep blockage in deficit and set blockage to zero
            blockage = np.zeros_like(deficit)
        return deficit, blockage

    def _calc_deficit(self, dw_ijlk, **kwargs):
        """Calculate wake (and blockage) deficit"""
        deficit = self.wake_deficitModel(dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit, blockage

    def _calc_deficit_convection(self, dw_ijlk, **kwargs):
        """Calculate wake convection deficit (and blockage)"""
        deficit, uc, sigma_sqr = self.wake_deficitModel.calc_deficit_convection(dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit, uc, sigma_sqr, blockage

    def _calc_wt_interaction_args(self, kwargs):
        """Used for parallel execution"""
        return self.calc_wt_interaction(**kwargs)

    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0, wd=None, ws=None, time=False,
                            WS_eff_ilk=None,
                            n_cpu=1, wd_chunks=None, ws_chunks=1,
                            **kwargs):
        """See WindFarmModel.calc_wt_interaction and additional parameters below

        Parameters
        ----------
        n_cpu : int or None, optional
            Number of CPUs to be used for execution.
            If 1 (default), the execution is not parallized
            If None, the available number of CPUs are used
        wd_chunks : int or None, optional
            If n_cpu>1, the wind directions are divided into <wd_chunks> chunks and executed in parallel.
            If wd_chunks is None, wd_chunks is set to the available number of CPUs
        ws_chunks : int or None, optional
            If n_cpu>1, the wind speeds are divided into <ws_chunks> chunks and executed in parallel.
            If ws_chunks is None, ws_chunks is set to 1
        """

        h_i, D_i = self.windTurbines.get_defaults(len(x_ilk), type_i, h_i)
        wd, ws = self.site.get_defaults(wd, ws)
        I, L, K, = len(x_ilk), len(wd), (1, len(ws))[time is False]
        kwargs.update(dict(x_ilk=x_ilk, y_ilk=y_ilk, h_ilk=h_i[:, na, na], wd=wd, ws=ws, time=time,
                           type_i=np.zeros_like(D_i) + type_i))

        for inputModifierModel in self.inputModifierModels:
            kwargs.update(inputModifierModel.setup(**kwargs))

        # Find local wind speed, wind direction, turbulence intensity and probability
        lw = self.site.local_wind(x=np.mean(x_ilk, (1, 2)), y=np.mean(y_ilk, (1, 2)), h=h_i,
                                  wd=kwargs['wd'], ws=kwargs['ws'], time=kwargs['time'])
        for k in ['WS_ilk', 'WD_ilk', 'TI_ilk']:
            if k in kwargs:
                lw.add_ilk(k, kwargs.pop(k))

        ri, oi = self.windTurbines.function_inputs

        if n_cpu != 1 or wd_chunks or ws_chunks > 1:
            # parallel execution
            map_func, arg_lst, wd_chunks, ws_chunks = self._multiprocessing_chunks(
                n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks,
                WS_eff_ilk=WS_eff_ilk, **kwargs)

            WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, _, kwargs = list(
                zip(*map_func(self._calc_wt_interaction_args, arg_lst)))

            def concatenate(v_ilk):
                v_ilk = [fix_shape(v, WS_eff.shape) for v, WS_eff in zip(v_ilk, WS_eff_ilk)]
                if kwargs[0]['time'] is False:
                    return np.concatenate([np.concatenate(v_ilk[i::ws_chunks], axis=1)
                                           for i in range(ws_chunks)], axis=2)
                else:
                    return np.concatenate(v_ilk, axis=1)

            return ([concatenate(v) for v in [WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk]] +
                    [lw,
                     {'type_i': kwargs[0]['type_i'],
                      **{k: concatenate([wt_i[k] for wt_i in kwargs]) for k in kwargs[0] if k.endswith('_ilk')}}])

        kwargs.update({'WD_ilk': lw.WD_ilk,
                       'WS_ilk': lw.WS_ilk,
                       'WS_eff_ilk': WS_eff_ilk,
                       'D_i': D_i,
                       'I': I, 'L': L, 'K': K,
                       **{k + '_ilk': self.site.interp(self.site.ds[k], lw) for k in ri + oi if k in self.site.ds},
                       })
        if hasattr(lw, 'TI_ilk'):
            kwargs['TI_ilk'] = lw.TI_ilk
            kwargs['TI_eff_ilk'] = lw.TI_ilk + 0.  # autograd-friendly copy

        self._check_input(kwargs)

        # Calculate down-wind and cross-wind distances
        self.site.distance.setup(kwargs['x_ilk'], kwargs['y_ilk'], kwargs['h_ilk'])

        WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs = self._calc_wt_interaction(**kwargs)

        power_ilk = self.windTurbines.power(ws=WS_eff_ilk, **self.get_wt_kwargs(TI_eff_ilk, kwargs))
        kwargs.update({'time': time, 'type_i': type_i})
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, lw, kwargs

    @abstractmethod
    def _calc_wt_interaction(self, **kwargs):
        """calculate WT interaction"""

    def get_map_args(self, x_j, y_j, h_j, sim_res_data, D_dst=0):
        wt_d_i = self.windTurbines.diameter(sim_res_data.type)
        wd, ws = [np.atleast_1d(sim_res_data[k].values) for k in ['wd', 'ws']]
        time = sim_res_data.get('time', False)
        wt_x_ilk = sim_res_data['x'].ilk()
        WD_il = sim_res_data.WD.ilk()

        lw_j = self.site.local_wind(x=x_j, y=y_j, h=h_j, wd=wd, ws=ws, time=time)
        I, J, L, K = [len(x) for x in [wt_x_ilk, x_j, wd, ws]]

        def get_ilk(k):
            v = sim_res_data[k].ilk()

            def wrap(l):
                l_ = [l, slice(0, 1)][v.shape[1] == 1]
                return v[:, l_]
            return wrap
        map_arg_funcs = {k.replace('CT', 'ct') + '_ilk': get_ilk(k)
                         for k in sim_res_data if k not in ['wd_bin_size', 'ws_l', 'ws_u']}
        map_arg_funcs.update({
            'D_src_il': lambda l: wt_d_i[:, na],
            'D_dst_ijl': lambda l: np.zeros((1, 1, 1)) + D_dst,
            'IJLK': lambda l=slice(None), I=I, J=J, L=L, K=K: (I, J, len(np.arange(L)[l]), K)})
        return map_arg_funcs, lw_j, wd, WD_il

    def _get_flow_l(self, model_kwargs, l, wt_x_ilk, wt_y_ilk, wt_h_ilk, lw_j, wd, WD_ilk):
        self.site.distance.setup(wt_x_ilk, wt_y_ilk, wt_h_ilk, (lw_j.x, lw_j.y, lw_j.h))
        dw_ijlk, hcw_ijlk, dh_ijlk = self.site.distance(wd_l=wd, WD_ilk=WD_ilk)

        WS_jlk = lw_j.WS_ilk[:, [l, slice(0, 1)][lw_j.WS_ilk.shape[1] == 1]]
        TI_jlk = lw_j.TI_ilk[:, [l, slice(0, 1)][lw_j.TI_ilk.shape[1] == 1]]

        if self.wec != 1:
            hcw_ijlk = hcw_ijlk / self.wec

        if self.deflectionModel:
            dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, dh_ijlk=dh_ijlk, z_ijlk=wt_h_ilk[:, na] + dh_ijlk,
                **model_kwargs)

        model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                             'x_ilk': wt_x_ilk, 'y_ilk': wt_y_ilk})

        if 'cw_ijlk' in self.args4all:
            model_kwargs['cw_ijlk'] = hypot(dh_ijlk, hcw_ijlk)

        if 'wake_radius_ijlk' in self.args4all:
            model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

        if 'wake_radius_ijl' in self.args4all:
            model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[..., 0]
        if 'z_ijlk' in self.args4all:
            model_kwargs['z_ijlk'] = wt_h_ilk[:, na] + dh_ijlk

        if isinstance(self.superpositionModel, WeightedSum):
            deficit_ijlk, uc_ijlk, sigma_sqr_ijlk, blockage_ijlk = self._calc_deficit_convection(**model_kwargs)
        else:
            deficit_ijlk, blockage_ijlk = self._calc_deficit(**model_kwargs)

        if self.turbulenceModel:
            add_turb_ijlk = self.turbulenceModel.calc_added_turbulence(**model_kwargs)

        if isinstance(self.superpositionModel, WeightedSum):
            cw_ijlk = hypot(dh_ijlk, hcw_ijlk)
            WS_eff_jlk = WS_jlk - self.superpositionModel(WS_jlk, deficit_ijlk, uc_ijlk,
                                                          sigma_sqr_ijlk, cw_ijlk, hcw_ijlk, dh_ijlk)
            if self.blockage_deficitModel:
                blockage_superpositionModel = self.blockage_deficitModel.superpositionModel or LinearSum()
                WS_eff_jlk -= blockage_superpositionModel(blockage_ijlk)
        else:
            WS_eff_jlk = WS_jlk - self.superpositionModel(deficit_ijlk)
            if self.blockage_deficitModel:
                blockage_superpositionModel = self.blockage_deficitModel.superpositionModel or self.superpositionModel
                WS_eff_jlk -= blockage_superpositionModel(blockage_ijlk)

        if self.turbulenceModel:
            TI_eff_jlk = self.turbulenceModel.calc_effective_TI(TI_jlk, add_turb_ijlk)
        else:
            TI_eff_jlk = None
        return WS_eff_jlk, TI_eff_jlk

    def _aep_map(self, x_j, y_j, h_j, type_j, sim_res_data):
        lw_j, WS_eff_jlk, _ = self._flow_map(x_j, y_j, h_j, sim_res_data)
        power_kwargs = {}
        if 'type' in (self.windTurbines.powerCtFunction.required_inputs +
                      self.windTurbines.powerCtFunction.optional_inputs):
            power_kwargs['type'] = type_j
        power_jlk = self.windTurbines.power(WS_eff_jlk, **power_kwargs)

        aep_j = (power_jlk * lw_j.P_ilk).sum((1, 2))
        return aep_j * 365 * 24 * 1e-9

    def _flow_map(self, x_j, y_j, h_j, sim_res_data, D_dst=0):
        """call this function via SimulationResult.flow_map"""
        arg_funcs, lw_j, wd, WD_il = self.get_map_args(x_j, y_j, h_j, sim_res_data, D_dst=D_dst)
        I, J, L, K = arg_funcs['IJLK']()
        if I == 0:
            return (lw_j, np.broadcast_to(lw_j.WS_ilk, (len(x_j), L, K)).astype(float),
                    np.broadcast_to(lw_j.TI_ilk, (len(x_j), L, K)).astype(float))

        size_gb = I * J * L * K * 8 / 1024**3
        wd_chunks = np.minimum(np.maximum(int(size_gb // 1), 1), L)
        wd_i = np.round(np.linspace(0, L, wd_chunks + 1)).astype(int)
        l_iter = tqdm([slice(i0, i1) for i0, i1 in zip(wd_i[:-1], wd_i[1:])], disable=L <= 1 or not self.verbose,
                      desc='Calculate flow map', unit='wd')
        wt_x_ilk, wt_y_ilk, wt_h_ilk = [sim_res_data[k].ilk() for k in ['x', 'y', 'h']]
        WS_eff_jlk, TI_eff_jlk = zip(*[self._get_flow_l(
            {k: arg_funcs[k](l) for k in arg_funcs},
            l,
            *[(v, v[:, l])[np.shape(v)[1] == L] for v in [wt_x_ilk, wt_y_ilk, wt_h_ilk]],
            lw_j, wd[l], WD_il[:, l])
            for l in l_iter])
        WS_eff_jlk = np.concatenate(WS_eff_jlk, 1)
        if self.turbulenceModel:
            TI_eff_jlk = np.concatenate(TI_eff_jlk, 1)
        else:
            TI_eff_jlk = np.zeros_like(WS_eff_jlk) + lw_j.TI_ilk
        return lw_j, WS_eff_jlk, TI_eff_jlk

    def _check_input(self, kwargs):
        x_ilk, y_ilk, h_ilk = kwargs['x_ilk'], kwargs['y_ilk'], kwargs['h_ilk']
        i1, i2, *_ = np.where((cabs(x_ilk[:, na] - x_ilk[na]) +
                               cabs(y_ilk[:, na] - y_ilk[na]) +
                               cabs(h_ilk[:, na] - h_ilk[na]) +
                               np.eye(len(x_ilk))[:, :, na, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" % (i1[i], i2[i]) for i in range(len(i1))])
            raise ValueError(msg)
        for k in self.args4all:
            if k.endswith('_ilk') and k not in ['ct_ilk'] and k not in kwargs:
                n = k.replace('_ilk', '')
                needed_by = str(self)
                for model in [self.wake_deficitModel, self.superpositionModel, self.blockage_deficitModel,
                              self.deflectionModel, self.turbulenceModel] + self.inputModifierModels:
                    if ((hasattr(model, 'args4model') and k in model.args4model) or
                            (hasattr(model, 'args4deficit') and k in model.args4deficit)):
                        needed_by = model.__class__.__name__
                        break
                raise ValueError(f"'{n}' needed by {needed_by} is missing")
        ri, oi = self.windTurbines.function_inputs
        for k in kwargs:
            n = k.replace('_ilk', '').replace('_i', '')
            if (n not in ri + oi and k not in self.args4all and
                    n not in {'x', 'y', 'h', 'wd', 'ws', 'time', 'type', 'D', 'WD', 'WS',
                              'WS_eff', 'TI', 'TI_eff', 'I', 'L', 'K'}):
                raise ValueError(f"WindFarmModel an got unexpected keyword argument: '{n}'")


class PropagateDownwind(EngineeringWindFarmModel):
    """Downstream wake deficits calculated and propagated in downstream direction.
    Very fast, but ignoring blockage effects
    """

    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None,
                 inputModifierModels=[]):
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
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel, rotorAvgModel,
                                          blockage_deficitModel=None, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, inputModifierModels=inputModifierModels)

    def _calc_wt_interaction(self, wd,
                             WS_eff_ilk, TI_eff_ilk,
                             D_i,
                             I, L, K, **kwargs):
        """
        Additional suffixes:

        - m: turbines and wind directions (il.flatten())
        - n: from_turbines, to_turbines and wind directions (iil.flatten())

        """

        deficit_nk = []
        uc_nk = []
        sigma_sqr_nk = []
        cw_nk = []
        hcw_nk = []
        dh_nk = []

        def ilk2mk(v_ilk):
            dtype = (float, np.complex128)[np.iscomplexobj(v_ilk)]
            _K = np.shape(v_ilk)[2]
            return np.broadcast_to(np.asarray(v_ilk).astype(dtype), (I, L, _K)).reshape((I * L, _K))

        WS_ilk, WD_ilk = [kwargs[k + '_ilk'] for k in ['WS', 'WD']]

        WS_mk, WD_mk, TI_mk, h_mk = [ilk2mk(kwargs[k + '_ilk']) for k in ['WS', 'WD', 'TI', 'h']]
        WS_eff_mk, TI_eff_mk = [], []
        yaw_mk = ilk2mk(kwargs.get('yaw_ilk', [[[0]]]))
        tilt_mk = ilk2mk(kwargs.get('tilt_ilk', [[[0]]]))
        modified_input_dict_mk = []

        ct_jlk = []

        if self.turbulenceModel:
            add_turb_nk = []

        i_wd_l = np.arange(L).astype(int)

        dw_order_indices_ld = self.site.distance.dw_order_indices(wd)[:, 0]

        wt_kwargs = self.get_wt_kwargs(TI_eff_ilk, kwargs)

        # Iterate over turbines in down wind order
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose, desc="Calculate flow interaction", unit="wt"):
            i_wt_l = dw_order_indices_ld[:, j]
            # current wt (j'th most upstream wts for all wdirs)
            m = i_wt_l * L + i_wd_l

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
                WS_eff_mk.append(WS_eff_lk)
                if self.turbulenceModel:
                    TI_eff_lk = TI_mk[m]
                    TI_eff_mk.append(np.broadcast_to(TI_eff_lk, (L, K)))
            else:  # 2..n most upstream turbines (wake)
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit2WT = np.array([d_nk2[i] for d_nk2, i in zip(deficit_nk, range(j)[::-1])])
                    uc2WT = np.array([d_nk2[i] for d_nk2, i in zip(uc_nk, range(j)[::-1])])
                    sigmasqr2WT = np.array([d_nk2[i] for d_nk2, i in zip(sigma_sqr_nk, range(j)[::-1])])
                    cw2WT = np.array([d_nk2[i] for d_nk2, i in zip(cw_nk, range(j)[::-1])])
                    hcw2WT = np.array([d_nk2[i] for d_nk2, i in zip(hcw_nk, range(j)[::-1])])
                    dh2WT = np.array([d_nk2[i] for d_nk2, i in zip(dh_nk, range(j)[::-1])])

                    WS_eff_lk = WS_mk[m] - self.superpositionModel(WS_mk[m],
                                                                   deficit2WT, uc2WT, sigmasqr2WT, cw2WT, hcw2WT, dh2WT)
                else:
                    deficit2WT = np.array([d_nk2[i] for d_nk2, i in zip(deficit_nk, range(j)[::-1])])
                    WS_eff_lk = WS_mk[m] - self.superpositionModel(deficit2WT)
                WS_eff_mk.append(WS_eff_lk)

                if self.turbulenceModel:
                    add_turb2WT = np.array([d_nk2[i] for d_nk2, i in zip(add_turb_nk, range(j)[::-1])])
                    TI_eff_lk = self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb2WT)
                    TI_eff_mk.append(TI_eff_lk)

            # Calculate Power/CT
            def mask(k, v):
                if len(np.squeeze(v).shape) == 0:
                    return np.squeeze(v)
                v = np.asarray(v)
                if v.shape[:2] == (I, L):
                    return v[i_wt_l, i_wd_l]
                elif v.shape[0] == I:
                    return v[i_wt_l].flatten()
                else:
                    assert v.shape[1] == L
                    return v[0, i_wd_l]

            _wt_kwargs = {k: mask(k, v) for k, v in wt_kwargs.items()}
            if 'TI_eff' in _wt_kwargs:
                _wt_kwargs['TI_eff'] = TI_eff_mk[-1]

            ct_lk = self.windTurbines.ct(ws=WS_eff_mk[-1], **_wt_kwargs)

            ct_jlk.append(ct_lk)

            if j < I - 1 or len(self.inputModifierModels):
                i_dw = dw_order_indices_ld[:, j + 1:]

                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_jlk': lambda: np.moveaxis([WS_ilk[(slice(0, 1), j)[WS_ilk.shape[0] > 1],
                                                                   (0, l)[WS_ilk.shape[1] > 1]]
                                                            for j, l in zip(i_dw, i_wd_l)], 0, 1),
                             'WS_eff_ilk': lambda: WS_eff_mk[-1][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[-1][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'tilt_ilk': lambda: tilt_mk[m][na],
                             'D_dst_ijl': lambda: D_i[dw_order_indices_ld[:, j + 1:]].T[na],
                             'h_ilk': lambda: h_mk[m][na],
                             'ct_ilk': lambda: ct_lk[na],
                             'IJLK': lambda: (1, i_dw.shape[1], L, K),
                             'WD_ilk': lambda: ilk2mk(WD_ilk)[m][na],
                             **{k + '_ilk': lambda k=k: ilk2mk(kwargs[k + '_ilk'])[m][na] for k in 'xyh'},
                             'type_il': lambda: kwargs['type_i'][i_wt_l][na]

                             }
                model_kwargs = {k: arg_funcs[k]() for k in self.args4all if k in arg_funcs}

                # custom model arguments
                custom_args = (set([k for k in self.args4all if k.endswith('_ilk')]) - set(model_kwargs)) & set(kwargs)
                model_kwargs.update({k: ilk2mk(kwargs[k])[m][na] for k in custom_args})

                dw_ijlk, hcw_ijlk, dh_ijlk = self.site.distance(
                    wd_l=wd, WD_ilk=WD_mk[m][na], src_idx=i_wt_l, dst_idx=i_dw.T)

                for inputModidifierModel in self.inputModifierModels:
                    modified_input_dict = inputModidifierModel(**model_kwargs)
                    modified_input_dict_mk.append(modified_input_dict)
                    model_kwargs.update(modified_input_dict)

                if self.wec != 1:
                    hcw_ijlk = hcw_ijlk / self.wec

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, dh_ijlk=dh_ijlk, **model_kwargs)

                model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk})
                if 'z_ijlk' in self.args4all:
                    model_kwargs['z_ijlk'] = h_mk[m][na, na] + dh_ijlk

                hcw_nk.append(hcw_ijlk[0])
                dh_nk.append(dh_ijlk[0])

                if 'cw_ijlk' in self.args4all:
                    # sqrt(a**2+b**2) as hypot does not support complex numbers
                    model_kwargs['cw_ijlk'] = np.sqrt(dh_ijlk**2 + hcw_ijlk**2)
                    cw_nk.append(model_kwargs['cw_ijlk'][0])

                if 'wake_radius_ijl' in self.args4all:
                    model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[..., 0]

                if 'wake_radius_ijlk' in self.args4all:
                    model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

                # Calculate deficit
                if isinstance(self.superpositionModel, WeightedSum):
                    deficit, uc, sigma_sqr, _ = self._calc_deficit_convection(**model_kwargs)
                    uc_nk.append(uc[0])
                    sigma_sqr_nk.append(sigma_sqr[0])
                else:
                    deficit, _ = self._calc_deficit(**model_kwargs)
                deficit_nk.append(deficit[0])

                if self.turbulenceModel:

                    # Calculate added turbulence
                    add_turb_nk.append(self.turbulenceModel(**model_kwargs)[0])

        WS_eff_jlk, ct_jlk = np.array(WS_eff_mk), np.array(ct_jlk)

        dw_inv_indices = (np.argsort(dw_order_indices_ld, 1).T * L + np.arange(L).astype(int)[na]).flatten()
        WS_eff_ilk = WS_eff_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))

        ct_ilk = ct_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_jlk = np.array(TI_eff_mk)
            TI_eff_ilk = TI_eff_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))

        if len(self.inputModifierModels):
            for k in modified_input_dict_mk[0].keys():
                mi_jlk = np.array([mi_dict[k] for mi_dict in modified_input_dict_mk])
                kwargs[k] = mi_jlk.reshape((I * L, K))[dw_inv_indices].reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs


class All2AllIterative(EngineeringWindFarmModel):
    """Wake and blockage deficits calculated from all wt to all points of interest (wt/map points).
    The calculations are iteratively repeated until convergence (change of effective wind speed < convergence_tolerance)"""

    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 convergence_tolerance=1e-6, rotorAvgModel=None, inputModifierModels=[]):
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
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        convergence_tolerance : float or None
            if float: maximum accepted change in WS_eff_ilk [m/s]
            if None: return after first iteration. This only makes sense for benchmark studies where CT,
            wakes and blockage are independent of effective wind speed WS_eff_ilk

        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel, rotorAvgModel,
                                          blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, inputModifierModels=inputModifierModels)
        self.convergence_tolerance = convergence_tolerance

    def _calc_wt_interaction(self, ws, wd, WD_ilk, WS_ilk, TI_ilk,
                             WS_eff_ilk, TI_eff_ilk,
                             D_i, time,
                             I, L, K, **kwargs):
        if any([np.iscomplexobj(v) for v in ([kwargs.get(k, 0)
                                              for k in ['x_ilk', 'y_ilk', 'h_ilk', 'D_i', 'yaw_ilk', 'tilt_ilk']] +
                                             [ws, wd])]):
            dtype = np.complex128
        else:
            dtype = float
        WS_ILK = np.broadcast_to(WS_ilk, (I, L, K))
        # calculate WS_eff without blockage as a first guess
        if WS_eff_ilk is None:
            # Initialize with PropagateDownwind
            blockage_deficitModel = self.blockage_deficitModel
            self.blockage_deficitModel = None
            WS_eff_ilk = PropagateDownwind._calc_wt_interaction(
                self, wd=wd, WD_ilk=WD_ilk, WS_ilk=WS_ilk, TI_ilk=TI_ilk, WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
                D_i=D_i, I=I, L=L, K=K, **kwargs)[0]
            self.blockage_deficitModel = blockage_deficitModel
        elif WS_eff_ilk == 0:
            WS_eff_ilk = WS_ILK + 0.

        WS_eff_ilk = WS_eff_ilk.astype(dtype)
        WS_eff_ilk_last = WS_eff_ilk + 0  # fast autograd-friendly copy
        diff_lk = np.zeros((L, K))
        diff_lk_last = None
        dw_iilk, hcw_iilk, dh_iilk = self.site.distance(wd_l=wd, WD_ilk=WD_ilk)

        wt_kwargs = self.get_wt_kwargs(TI_eff_ilk, kwargs)
        ct_ilk = self.windTurbines.ct(ws=WS_ILK, **wt_kwargs)
        ct_ilk_idle = self.windTurbines.ct(ws=0.1 * np.ones_like(WS_ILK), **wt_kwargs)
        unstable_lk = np.zeros((L, K), dtype=bool)
        ioff = np.broadcast_to(ct_ilk, (I, L, K)) < -1  # index of off/idling turbines
        D_src_il = D_i[:, na]
        model_kwargs = {'WS_ilk': WS_ilk,
                        'WS_eff_ilk': WS_eff_ilk,
                        'WD_ilk': WD_ilk,
                        'TI_ilk': TI_ilk,
                        'TI_eff_ilk': TI_eff_ilk,
                        'D_src_il': D_src_il,
                        'D_dst_ijl': D_src_il[na],
                        'dw_ijlk': dw_iilk,
                        'hcw_ijlk': hcw_iilk,
                        'cw_ijlk': np.sqrt(hcw_iilk**2 + dh_iilk**2),
                        'dh_ijlk': dh_iilk,
                        'z_ijlk': kwargs['h_ilk'][:, na] + dh_iilk,
                        'IJLK': (I, I, L, K),
                        'type_il': kwargs['type_i'][:, na],
                        ** kwargs,
                        }
        if 'wake_radius_ijl' in self.args4all:
            model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[:, :, :, 0]

        if not self.deflectionModel:
            self._init_deficit(**model_kwargs)

        cw_iilk = np.sqrt(hcw_iilk**2 + dh_iilk**2)

        i2i_zero = ~np.eye(I).astype(bool)[:, :, na, na]

        # Iterate until convergence
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose,
                      desc="Calculate flow interaction", unit="Iteration"):

            ct_ilk = self.windTurbines.ct(np.maximum(WS_eff_ilk, 0), **wt_kwargs)
            ioff |= (unstable_lk)[na] & (ct_ilk <= ct_ilk_idle)

            model_kwargs.update(dict(ct_ilk=ct_ilk, WS_eff_ilk=WS_eff_ilk))
            if self.inputModifierModels:
                # x_ilk, y_ilk and h_ilk is may be updated by an inputModifierModel and
                # must be reset in every iterations
                model_kwargs.update(dict(x_ilk=kwargs['x_ilk'], y_ilk=kwargs['y_ilk'], h_ilk=kwargs['h_ilk']))

            if self.deflectionModel:
                model_kwargs.update(dict(
                    # dw_ijlk, hcw_ijlk and dh_ijlk is updated by deflection model and must be reset in every iterations
                    dw_ijlk=dw_iilk,
                    hcw_ijlk=hcw_iilk,
                    cw_ijlk=cw_iilk,
                    dh_ijlk=dh_iilk,
                    z_ijlk=kwargs['h_ilk'][:, na] + dh_iilk))

            for inputModidifierModel in self.inputModifierModels:
                modified_input_dict = inputModidifierModel(**model_kwargs)
                model_kwargs.update(modified_input_dict)
                if any([k in modified_input_dict for k in ['x_ilk', 'y_ilk']]):
                    self.site.distance.setup(model_kwargs['x_ilk'], model_kwargs['y_ilk'], model_kwargs['h_ilk'])
                    model_kwargs.update({k: v for k, v in zip(['dw_ijlk', 'hcw_ijlk', 'dh_ijlk'],
                                                              self.site.distance(wd_l=wd, WD_ilk=WD_ilk))})
                    model_kwargs['cw_ijlk'] = hypot(model_kwargs['dh_ijlk'], model_kwargs['hcw_ijlk'])
                    if not self.deflectionModel:
                        self._init_deficit(**model_kwargs)

            if self.deflectionModel:
                dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(**model_kwargs)
                model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                                     'cw_ijlk': hypot(dh_ijlk, hcw_ijlk)})
                self._reset_deficit()
            if 'wake_radius_ijlk' in self.args4all:
                model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

            if self.turbulenceModel:
                model_kwargs['TI_eff_ilk'] = TI_eff_ilk

            # Calculate deficit
            if isinstance(self.superpositionModel, WeightedSum):
                deficit_iilk, uc_iilk, sigmasqr_iilk, blockage_iilk = self._calc_deficit_convection(**model_kwargs)
            else:
                deficit_iilk, blockage_iilk = self._calc_deficit(**model_kwargs)

            # set own deficit to 0
            deficit_iilk *= i2i_zero
            if blockage_iilk is not None:
                blockage_iilk *= i2i_zero

            # Calculate effective wind speed
            if isinstance(self.superpositionModel, WeightedSum):
                WS_eff_ilk = WS_ilk - self.superpositionModel(WS_ilk, deficit_iilk,
                                                              uc_iilk, sigmasqr_iilk,
                                                              model_kwargs['cw_ijlk'],
                                                              model_kwargs['hcw_ijlk'],
                                                              dh_iilk)
                # Add blockage as linear effect
                if self.blockage_deficitModel:
                    WS_eff_ilk -= (self.blockage_deficitModel.superpositionModel or LinearSum())(blockage_iilk)
            else:
                WS_eff_ilk = WS_ilk.astype(dtype) - self.superpositionModel(deficit_iilk)
                if self.blockage_deficitModel:
                    WS_eff_ilk -= (self.blockage_deficitModel.superpositionModel or self.superpositionModel)(blockage_iilk)

            # ensure idling wt in unstable flow cases do not cutin even if ws increases due to speedup
            # this helps to converge
            # WS_eff_ilk[ioff] = np.minimum(WS_eff_ilk[ioff], WS_eff_ilk_last[ioff])
            WS_eff_ilk = np.minimum(WS_eff_ilk, WS_eff_ilk_last, out=WS_eff_ilk, where=ioff)

            if self.turbulenceModel:
                add_turb_ijlk = self.turbulenceModel(**model_kwargs)
                add_turb_ijlk *= i2i_zero
                TI_eff_ilk = self.turbulenceModel.calc_effective_TI(TI_ilk, add_turb_ijlk)

            # Check if converged
            diff_ilk = cabs(WS_eff_ilk_last - WS_eff_ilk)
            diff_lk = diff_ilk.mean(0)
            max_diff = np.max(diff_ilk.max(0))

            if (self.convergence_tolerance is None or
                    (self.convergence_tolerance and max_diff < self.convergence_tolerance)):
                break
            # i_, l_, k_ = list(zip(*np.where(diff_ilk == max_diff)))[0]
            # wsi, wsl, wsk = WS_ilk.shape

            # wsi, wsl, wsk = WS_ilk.shape

            # print("Iteration: %d, max diff_ilk: %.8f, WT: %d, WD: %d, WS: %f, WS_eff: %f" %
            #       (j, max_diff, i_, wd[l_],
            #        WS_ilk[min(i_, wsi - 1), min(l_, wsl - 1), min(k_, wsk - 1)],
            #        WS_eff_ilk[i_, l_, k_]))
            # print(j, diff_ilk.mean(0), WS_eff_ilk.squeeze())

            # assume flow case to be unstable if mean difference of two iterations increases
            if j > 1:
                unstable_lk |= diff_lk_last < diff_lk

            WS_eff_ilk_last = WS_eff_ilk + 0  # fast autograd-friendly copy
            diff_lk_last = diff_lk

        # print("All2AllIterative converge after %d iterations" % (j + 1))
        self.iterations = j + 1
        self.WS_eff_ilk_last = getattr(WS_eff_ilk, '_value', WS_eff_ilk)
        self._reset_deficit()

        if len(self.inputModifierModels):
            kwargs.update({k: modified_input_dict[k] for k in modified_input_dict})
        return WS_eff_ilk, np.broadcast_to(TI_eff_ilk, (I, L, K)), ct_ilk, kwargs


class All2All(All2AllIterative):
    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 rotorAvgModel=None):
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel=superpositionModel,
                                  blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=None,
                                  rotorAvgModel=rotorAvgModel)

    def _calc_wt_interaction(self, WS_eff_ilk, **kwargs):
        return All2AllIterative._calc_wt_interaction(self, WS_eff_ilk=0, **kwargs)


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
