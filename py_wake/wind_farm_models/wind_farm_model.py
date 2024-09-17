from abc import abstractmethod, ABC
from py_wake.site._site import Site, LocalWind
from py_wake.wind_turbines import WindTurbines
from py_wake import np
from py_wake.flow_map import FlowMap, HorizontalGrid, FlowBox, Grid
import xarray as xr
from py_wake.utils import xarray_utils, weibull  # register ilk function @UnusedImport
from numpy import newaxis as na
from py_wake.utils.model_utils import check_model, fix_shape
import multiprocessing
from py_wake.utils.parallelization import get_pool_map, get_pool_starmap
from py_wake.utils.functions import arg2ilk
from py_wake.utils.gradients import autograd
from py_wake.noise_models.iso import ISONoiseModel


class WindFarmModel(ABC):
    """Base class for RANS and engineering flow models"""
    verbose = True

    def __init__(self, site, windTurbines):
        check_model(site, Site, 'site')
        check_model(windTurbines, WindTurbines, 'windTurbines')
        self.site = site
        self.windTurbines = windTurbines

    def get_wt_kwargs(self, TI_eff_ilk, kwargs):
        wt_kwargs = {}

        def add_arg(name, optional):
            if name in wt_kwargs:  # custom WindFarmModel.__call__ arguments
                return
            elif name + '_ilk' in kwargs:
                wt_kwargs[name] = kwargs[k + '_ilk']
            elif name + '_i' in kwargs:
                wt_kwargs[name] = kwargs[k + '_i']
            # elif name in self.site.ds:
            #    wt_kwargs[name] = self.site.interp(self.site.ds[name], lw)
            elif name in ['TI_eff']:
                if self.turbulenceModel:
                    wt_kwargs['TI_eff'] = TI_eff_ilk
                elif optional is False:
                    raise KeyError("Argument, TI_eff, needed to calculate power and ct requires a TurbulenceModel")
            elif name in ['dw_ijlk', 'cw_ijlk', 'hcw_ijlk']:
                pass
            elif optional:
                pass
            else:
                raise KeyError("Argument, %s, required to calculate power and ct not found" % name)
        for opt, lst in zip([False, True], self.windTurbines.function_inputs):
            for k in lst:
                add_arg(k, opt)
        return wt_kwargs

    def _run(self, x, y, h=None, type=0, wd=None, ws=None, time=False, verbose=False,  # @ReservedAssignment
             n_cpu=1, wd_chunks=None, ws_chunks=1, **kwargs):
        if time is False and np.ndim(wd):
            wd = np.sort(wd)
        assert len(x) == len(y)
        self.verbose = verbose
        h, _ = self.windTurbines.get_defaults(len(x), type, h)
        wd, ws = self.site.get_defaults(wd, ws)
        I, L, K, = len(x), len(np.atleast_1d(wd)), (1, len(np.atleast_1d(ws)))[time is False]
        if len([k for k in kwargs if 'yaw' in k.lower() and k != 'yaw' and not k.startswith('yawc_')]):
            raise ValueError(
                'Custom *yaw*-keyword arguments not allowed to avoid confusion with the default "yaw" keyword')
        kwargs.update(dict(x=x, y=y, h=h))
        kwargs_ilk = {k + '_ilk': arg2ilk(k, v, I, L, K) for k, v in kwargs.items()}

        return self.calc_wt_interaction(h_i=h, type_i=type,
                                        wd=wd, ws=ws, time=time,
                                        n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks,
                                        **kwargs_ilk)

    def __call__(self, x, y, h=None, type=0, wd=None, ws=None, time=False, verbose=False,  # @ReservedAssignment
                 n_cpu=1, wd_chunks=None, ws_chunks=1, return_simulationResult=True, **kwargs):
        """Run the wind farm simulation

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        h : array_like, optional
            Wind turbine hub heights
        type : int or array_like, optional
            Wind turbine type, default is 0
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        yaw : int, float, array_like or None, optional
            Yaw misalignement, Positive is counter-clockwise when seen from above.
            May be
            - constant for all wt and flow cases or dependent on
            - wind turbine(i),
            - wind turbine and wind direction(il) or
            - wind turbine, wind direction and wind speed (ilk)
        tilt : array_like or None, optional
            Tilt angle of rotor shaft. Normal tilt (rotor center above tower top) is positivie
            May be
            - constant for all wt and flow cases or dependent on
            - wind turbine(i),
            - wind turbine and wind direction(il) or
            - wind turbine, wind direction and wind speed (ilk)
        time : boolean or array_like
            If False (default), the simulation will be computed for the full wd x ws matrix
            If True, the wd and ws will be considered as a time series of flow conditions with time stamp 0,1,..,n
            If array_like: same as True, but the time array is used as flow case time stamp
        n_cpu : int or None, optional
            Number of CPUs to be used for execution.
            If 1 (default), the execution is not parallized
            If None, the available number of CPUs are used
        wd_chunks : int or None, optional
            The wind directions are divided into <wd_chunks> chunks. More chunks reduces the memory usage
            and allows parallel execution if n_cpu>1.
            If wd_chunks is None, wd_chunks is set to the number of CPUs used, i.e. 1 if n_cpu is not specified
        ws_chunks : int, optional
            The wind speeds are divided into <ws_chunks> chunks. More chunks reduces the memory usage
            and allows parallel execution if n_cpu>1.
        return_simulationResult : boolean, optional
            see Returns

        Returns
        -------
        If return_simulationResult is True a SimulationResult (xarray Dataset) is returned
        If return_simulationResult is False the functino returns a tuple of:
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk
        """

        res = self._run(x, y, h=h, type=type, wd=wd, ws=ws, time=time, verbose=verbose,
                        n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks, **kwargs)
        if return_simulationResult:
            WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk = res

            return SimulationResult(self, localWind=localWind,
                                    WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
                                    power_ilk=power_ilk, ct_ilk=ct_ilk, **kwargs_ilk)
        else:
            return res

    def aep(self, x, y, h=None, type=0, wd=None, ws=None,  # @ReservedAssignment
            normalize_probabilities=False, with_wake_loss=True,
            n_cpu=1, wd_chunks=None, ws_chunks=1, **kwargs):
        """Anual Energy Production (sum of all wind turbines, directions and speeds) in GWh.

        the typical use is:
        >> sim_res = windFarmModel(x,y,...)
        >> sim_res.aep()

        This function bypasses the simulation result and returns only the total AEP,
        which makes it slightly faster for small problems.
        >> windFarmModel.aep(x,y,...)

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        h : array_like, optional
            Wind turbine hub heights
        type : int or array_like, optional
            Wind turbine type, default is 0
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        yaw : int, float, array_like or None, optional
            Yaw misalignement, Positive is counter-clockwise when seen from above.
            May be
            - constant for all wt and flow cases or dependent on
            - wind turbine(i),
            - wind turbine and wind direction(il) or
            - wind turbine, wind direction and wind speed (ilk)
        tilt : array_like or None, optional
            Tilt angle of rotor shaft. Normal tilt (rotor center above tower top) is positivie
            May be
            - constant for all wt and flow cases or dependent on
            - wind turbine(i),
            - wind turbine and wind direction(il) or
            - wind turbine, wind direction and wind speed (ilk)
        n_cpu : int or None, optional
            Number of CPUs to be used for execution.
            If 1 (default), the execution is not parallized
            If None, the available number of CPUs are used
        wd_chunks : int or None, optional
            The wind directions are divided into <wd_chunks> chunks. More chunks reduces the memory usage
            and allows parallel execution if n_cpu>1.
            If wd_chunks is None, wd_chunks is set to the number of CPUs used, i.e. 1 if n_cpu is not specified
        ws_chunks : int, optional
            The wind speeds are divided into <ws_chunks> chunks. More chunks reduces the memory usage
            and allows parallel execution if n_cpu>1.

        Returns
        -------
        AEP in GWh

        """
        res = self._run(x, y, h=h, type=type, wd=wd, ws=ws,
                        n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks, **kwargs)
        _, _, power_ilk, _, localWind, kwargs_ilk = res
        P_ilk = localWind.P_ilk
        if normalize_probabilities:
            norm = P_ilk.sum((1, 2))[:, na, na]
        else:
            norm = 1

        if with_wake_loss is False:
            wd, ws = self.site.get_defaults(wd, ws)
            I, L, K, = len(x), len(np.atleast_1d(wd)), len(np.atleast_1d(ws))
            power_ilk = np.broadcast_to(self.windTurbines.power(ws=localWind.WS_ilk,
                                                                **self.get_wt_kwargs(localWind.TI.ilk(), kwargs_ilk)), (I, L, K))
        return (power_ilk * P_ilk / norm * 24 * 365 * 1e-9).sum()

    @abstractmethod
    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0,
                            wd=None, ws=None, time=False,
                            n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):
        """Calculate effective wind speed, turbulence intensity,
        power and thrust coefficient, and local site parameters

        Typical users should not call this function directly, but by calling the
        windFarmModel object (invokes the __call__() function above)
        which returns a nice SimulationResult object

        Parameters
        ----------
        x_ilk : array_like
            X position of wind turbines
        y_ilk : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i : array_like or None, optional
            Wind turbine types\n
            If None, default, the first type is used (type=0)
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws
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


        Returns
        -------
        WS_eff_ilk : array_like
            Effective wind speeds [m/s]
        TI_eff_ilk : array_like
            Effective turbulence intensities [-]
        power_ilk : array_like
            Power productions [w]
        ct_ilk : array_like
            Thrust coefficients
        localWind : LocalWind
            Local free-flow wind
        """

    def _multiprocessing_chunks(self, wd, ws, time,
                                n_cpu, wd_chunks, ws_chunks, **kwargs):
        n_cpu = n_cpu or multiprocessing.cpu_count()
        wd_chunks = int(np.minimum(wd_chunks or n_cpu, len(wd)))
        ws_chunks = int(np.minimum(ws_chunks or 1, len(ws)))

        if time is not False:
            wd_chunks = ws_chunks = int(np.maximum(ws_chunks, wd_chunks))

        wd_i = np.linspace(0, len(wd) + 1, wd_chunks + 1).astype(int)
        ws_i = np.linspace(0, len(ws) + 1, ws_chunks + 1).astype(int)
        if n_cpu > 1:
            map_func = get_pool_map(n_cpu)
        else:
            from tqdm import tqdm

            def map_func(f, iter):
                return tqdm(map(f, iter), total=len(iter), disable=not self.verbose)

        if time is False:
            # (wd x ws) matrix
            slice_lst = [(slice(wd_i0, wd_i1), slice(ws_i0, ws_i1))
                         for wd_i0, wd_i1 in zip(wd_i[:-1], wd_i[1:])
                         for ws_i0, ws_i1 in zip(ws_i[:-1], ws_i[1:])]
        else:
            # (wd, ws) vector
            if time is True:
                time = np.arange(len(wd))
            slice_lst = [(slice(wd_i0, wd_i1), slice(wd_i0, wd_i1))
                         for wd_i0, wd_i1 in zip(wd_i[:-1], wd_i[1:])]

        I, L, K = len(kwargs.get('x_ilk', kwargs.get('x'))), len(wd), len(ws)

        def get_subtask_arg(k, arg, wd_slice, ws_slice):
            if (isinstance(arg, (None.__class__, bool, int, float)) or
                    k in {'gradient_method', 'wrt_arg'}):
                return arg
            s = np.shape(arg)
            if s in {(), (I,), (I, 1, 1), (1, 1, 1)}:
                return arg
            elif s == (I, L) or s == (I, L, 1) or s == (1, L, 1):
                return arg[:, wd_slice]
            elif s in {(I, L, K), (1, L, K)}:
                return arg[:, wd_slice][:, :, ws_slice]
            elif s in {(I, 1, K), (1, 1, K)}:
                return arg[:, :, ws_slice]
            elif s == (L,):
                return arg[wd_slice]
            # elif s == (L, K):
            #     return arg[wd_slice][:, ws_slice]
            else:  # pragma: no cover
                raise ValueError(f'Shape, {s}, of argument {k} is invalid')

        arg_lst = [{'wd': wd[wd_slice], 'ws': ws[ws_slice], 'time':get_subtask_arg('time', time, wd_slice, ws_slice),
                    ** {k: get_subtask_arg(k, v, wd_slice, ws_slice) for k, v in kwargs.items()}} for wd_slice, ws_slice in slice_lst]

        return map_func, arg_lst, wd_chunks, ws_chunks

    def _aep_chunk_wrapper(self, aep_function,
                           x, y, h=None, type=0, wd=None, ws=None,   # @ReservedAssignment
                           normalize_probabilities=False, with_wake_loss=True,
                           n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):
        wd, ws = self.site.get_defaults(wd, ws)
        wd_bin_size = self.site.wd_bin_size(wd)

        map_func, kwargs_lst, wd_chunks, ws_chunks = self._multiprocessing_chunks(
            wd=wd, ws=ws, time=False, n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks,
            x=x, y=y, h=h, type=type, **kwargs)

        return np.sum([np.array(aep) / self.site.wd_bin_size(args['wd']) * wd_bin_size
                       for args, aep in zip(kwargs_lst, map_func(aep_function, kwargs_lst))], 0)

    def aep_gradients(self, gradient_method=autograd, wrt_arg=['x', 'y'], gradient_method_kwargs={},
                      n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):
        """Method to compute the gradients of the AEP with respect to wrt_arg using the gradient_method

        Note, this method has two behaviours:
        1) Without specifying additional key-word arguments, kwargs, the method returns the function to
        compute the gradients of the aep:
        gradient_function = wfm.aep_gradietns(autograd, ['x','y'])
        gradients = gradient_function(x,y)
        This behaiour only works when wrt_arg is one or more of ['x','y','h','wd', 'ws']

        2) With additional key-word arguments, kwargs, the method returns the gradients of the aep:
        gradients = wfm.aep_gradients(autograd,['x','y'],x=x,y=y)
        This behaviour also works when wrt_arg is a keyword argument, e.g. yaw

        Parameters
        ----------
        gradient_method : gradient function, {fd, cs, autograd}
            gradient function
        wrt_arg : {'x', 'y', 'h', 'wd', 'ws', 'yaw','tilt'} or list of these arguments, e.g. ['x','y']
            argument to compute gradients of AEP with respect to
        gradient_method_kwargs : dict, optional
            additional arguments for the gradient method, e.g. step size
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
        if n_cpu != 1 or wd_chunks or ws_chunks:
            return self._aep_chunk_wrapper(
                self._aep_gradients_kwargs, gradient_method=gradient_method, wrt_arg=wrt_arg,
                gradient_method_kwargs=gradient_method_kwargs,
                n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks, **kwargs)

        if kwargs:
            wrt_arg = np.atleast_1d(wrt_arg)

            def wrap_aep(*args, **kwargs):
                kwargs.update({n: v for n, v in zip(wrt_arg, args)})
                return self.aep(**kwargs)

            f = gradient_method(wrap_aep, True, tuple(range(len(wrt_arg))), **gradient_method_kwargs)
            return np.array(f(*[kwargs.pop(n) for n in wrt_arg], **kwargs))
        else:
            argnum = [['x', 'y', 'h', 'type', 'wd', 'ws'].index(a) for a in np.atleast_1d(wrt_arg)]
            f = gradient_method(self.aep, True, argnum, **gradient_method_kwargs)
            return f

    def _aep_gradients_kwargs(self, kwargs):
        return self.aep_gradients(**kwargs)


class SimulationResult(xr.Dataset):
    """Simulation result returned when calling a WindFarmModel object"""
    __slots__ = ('windFarmModel', 'localWind')

    def __init__(self, windFarmModel, localWind, type_i,
                 WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, **kwargs):
        self.windFarmModel = windFarmModel
        lw = localWind
        self.localWind = localWind
        n_wt = len(lw.i)

        coords = {k: (dep, v, {'Description': d}) for k, dep, v, d in [
            ('wt', 'wt', np.arange(n_wt), 'Wind turbine number'),
            ('wd', ('wd', 'time')['time' in lw], lw.wd, 'Ambient reference wind direction [deg]'),
            ('ws', ('ws', 'time')['time' in lw], lw.ws, 'Ambient reference wind speed [m/s]'),
            ('type', 'wt', np.zeros(n_wt) + type_i, 'Wind turbine type')]}
        if 'time' in lw:
            coords['time'] = ('time', lw.time)

        ilk_dims = np.array((['wt', 'wd', 'ws'], ['wt', 'time'])['time' in lw], dtype=np.str_)
        data_vars = {k: (ilk_dims, (v, v[:, :, 0])['time' in lw], {'Description': d})
                     for k, v, d in [('WS_eff', WS_eff_ilk, 'Effective local wind speed [m/s]'),
                                     ('TI_eff', np.zeros_like(WS_eff_ilk) + TI_eff_ilk,
                                      'Effective local turbulence intensity'),
                                     ('Power', power_ilk, 'Power [W]'),
                                     ('CT', ct_ilk, 'Thrust coefficient'),
                                     ]}
        description = dict([('x', 'Wind turbine x coordinate [m]'),
                            ('y', 'Wind turbine y coordinate [m]'),
                            ('h', 'Wind turbine hub height [m]'),
                            ('yaw', 'Yaw misalignment [deg]'),
                            ('tilt', 'Rotor tilt [deg]')])
        for k in kwargs:
            if k.endswith('_ilk'):
                v = kwargs[k]
                if k in {'x_ilk', 'y_ilk'}:
                    dims = ['wt'] + [d for d, s in zip(ilk_dims[1:], np.shape(v)[1:]) if s > 1]
                    v = v.squeeze(tuple([i for i, d in enumerate(v.shape[1:], 1) if d == 1]))
                else:
                    dims = [d for d, s in zip(ilk_dims, np.shape(v)) if s != 1]
                    v = v.squeeze(tuple([i for i, d in enumerate(v.shape) if d == 1]))
                v = np.broadcast_to(v, [len(coords[k][1]) for k in dims])
                n = k.replace("_ilk", '')
                data_vars[n] = (dims, v,
                                {'Description': description.get(n, '')})

        for n in localWind:
            if n[-4:] == '_ilk':
                if n[:-4] not in data_vars:
                    data_vars[n[:-4]] = getattr(localWind, n[:-4])
                    if 'i' in data_vars[n[:-4]].dims:
                        data_vars[n[:-4]] = data_vars[n[:-4]].rename(i='wt')
            elif n in ['ws_lower', 'ws_upper']:
                if 'time' not in lw:
                    v = localWind[n]
                    dims = [n for n, d in zip(('wt', 'wd', 'ws'), v.shape) if d > 1 or d == 0]
                    data_vars[n[:-4]] = (dims, v.squeeze())
            else:
                data_vars[n] = localWind[n]

        xr.Dataset.__init__(self,
                            data_vars=data_vars,
                            coords=coords)

        # for backward compatibility
        for k in ['WD', 'WS', 'TI', 'P', 'WS_eff', 'TI_eff']:
            setattr(self.__class__, "%s_ilk" % k, property(lambda self, k=k: self[k].ilk()))
        setattr(self.__class__, "ct_ilk", property(lambda self: self.CT.ilk()))
        setattr(self.__class__, "power_ilk", property(lambda self: self.Power.ilk()))

    def aep_ilk(self, normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production of all turbines (i), wind directions (l) and wind speeds (k) in  in GWh

        Parameters
        ----------
        normalize_propabilities : Optional bool, defaults to False
            In case only a subset of all wind speeds and/or wind directions is simulated,
            this parameter determines whether the returned AEP represents the energy produced in the fraction
            of a year where these flow cases occur or a whole year of only these cases.
            If for example, wd=[0], then
            - False means that the AEP only includes energy from the faction of year\n
            with northern wind (359.5-0.5deg), i.e. no power is produced the rest of the year.
            - True means that the AEP represents a whole year of northen wind.
        with_wake_loss : Optional bool, defaults to True
            If True, wake loss is included, i.e. power is calculated using local effective wind speed\n
            If False, wake loss is neglected, i.e. power is calculated using local free flow wind speed
         """
        return self.aep(normalize_probabilities=normalize_probabilities, with_wake_loss=with_wake_loss).ilk()

    def aep(self, normalize_probabilities=False, with_wake_loss=True,
            hours_pr_year=24 * 365, linear_power_segments=False):
        """Anual Energy Production (sum of all wind turbines, directions and speeds) in GWh.

        See aep_ilk
        """
        if normalize_probabilities:
            norm = self.P.ilk().sum((1, 2))[:, na, na]
        else:
            norm = 1
        if with_wake_loss:
            power_ilk = self.Power.ilk()
        else:
            power_ilk = self.windFarmModel.windTurbines.power(
                self.WS.ilk(self.Power.ilk().shape), **self.wt_kwargs)

        if linear_power_segments:
            s = "The linear_power_segments method "
            assert all([n in self for n in ['Weibull_A', 'Weibull_k', 'Sector_frequency']]),\
                s + "requires a site with weibull information"
            assert normalize_probabilities is False, \
                s + "cannot be combined with normalize_probabilities"
            assert np.all(self.Power.isel(ws=0) == 0) and np.all(self.Power.isel(ws=-1) == 0),\
                s + "requires first wind speed to have no power (just below cut-in)"
            assert np.all(self.Power.isel(ws=-1) == 0),\
                s + "requires last wind speed to have no power (just above cut-out)"
            weighted_power = weibull.WeightedPower(
                self.ws.values,
                self.Power.ilk(),
                self.Weibull_A.ilk(),
                self.Weibull_k.ilk())
            aep = weighted_power * self.Sector_frequency.ilk() * hours_pr_year * 1e-9
            ws = (self.ws.values[1:] + self.ws.values[:-1]) / 2
            return xr.DataArray(aep, [('wt', self.wt.values), ('wd', self.wd.values), ('ws', ws)])
        else:
            weighted_power = power_ilk * self.P.ilk() / norm
        if 'time' in self.dims and weighted_power.shape[2] == 1:
            weighted_power = weighted_power[:, :, 0]

        return xr.DataArray(weighted_power * hours_pr_year * 1e-9,
                            self.Power.coords,
                            name='AEP [GWh]',
                            attrs={'Description': 'Annual energy production [GWh]'})

    def loads(self, method, lifetime_years=20, n_eq_lifetime=1e7, normalize_probabilities=False, softmax_base=None):
        assert method in ['TwoWT', 'OneWT_WDAvg', 'OneWT']
        wt = self.windFarmModel.windTurbines

        P_ilk = self.P_ilk
        if normalize_probabilities:
            P_ilk /= P_ilk.sum((1, 2))[:, na, na]
        WS_eff_ilk = self.WS_eff_ilk
        TI_eff_ilk = self.TI_eff_ilk

        kwargs = self.wt_kwargs

        if method == 'OneWT_WDAvg':  # average over wd
            p_wd_ilk = P_ilk.sum((0, 2))[na, :, na]
            ws_ik = (WS_eff_ilk * p_wd_ilk).sum(1)
            kwargs_ik = {k: (fix_shape(v, WS_eff_ilk) * p_wd_ilk).sum(1) for k, v in kwargs.items()
                         if k != 'TI_eff' and v is not None}
            kwargs_ik.update({k: v for k, v in kwargs.items() if v is None})

            loads, i_lst = [], []
            m_lst = np.asarray(wt.loadFunction.wohler_exponents)
            for m in np.unique(m_lst):
                i = np.where(m_lst == m)[0]
                if 'TI_eff' in kwargs:
                    kwargs_ik['TI_eff'] = ((p_wd_ilk * TI_eff_ilk ** m).sum(1)) ** (1 / m)
                loads.extend(wt.loads(ws_ik, run_only=i, **kwargs_ik))
                i_lst.extend(i)
            loads = [loads[i] for i in np.argsort(i_lst)]  # reorder

            ds = xr.DataArray(
                loads,
                dims=['sensor', 'wt', 'ws'],
                coords={'sensor': wt.loadFunction.output_keys,
                        'm': ('sensor', wt.loadFunction.wohler_exponents),
                        'wt': self.wt, 'ws': self.ws},
                attrs={'description': '1Hz Damage Equivalent Load'}).to_dataset(name='DEL')
            if 'wd' in self.P.dims:
                ds['P'] = self.P.sum('wd')
            else:
                ds['P'] = self.P
            t_flowcase = ds.P * lifetime_years * 365 * 24 * 3600
            f = ds.DEL.mean()  # factor used to reduce numerical errors in power
            ds['LDEL'] = ((t_flowcase * (ds.DEL / f)**ds.m).sum('ws') / n_eq_lifetime)**(1 / ds.m) * f
            ds.LDEL.attrs['description'] = "Lifetime (%d years) equivalent loads, n_eq_L=%d" % (
                lifetime_years, n_eq_lifetime)
        elif method == 'OneWT' or method == 'TwoWT':
            if method == 'OneWT':
                loads_silk = wt.loads(WS_eff_ilk, **kwargs)
            else:  # method == 'TwoWT':
                I, L, K = WS_eff_ilk.shape
                ws_iilk = np.broadcast_to(WS_eff_ilk[na], (I, I, L, K))

                def _fix_shape(k, v):
                    if k[-4:] == 'ijlk':
                        return fix_shape(v, ws_iilk)
                    else:
                        return np.broadcast_to(fix_shape(v, WS_eff_ilk)[na], (I, I, L, K))
                kwargs_iilk = {k: _fix_shape(k, v)
                               for k, v in kwargs.items()
                               if k in wt.loadFunction.required_inputs + wt.loadFunction.optional_inputs}

                loads_siilk = np.array(wt.loads(ws_iilk, **kwargs_iilk))
                if softmax_base is None:
                    loads_silk = loads_siilk.max(1)
                else:
                    # factor used to reduce numerical errors in power
                    f = loads_siilk.mean((1, 2, 3, 4)) / 10
                    loads_silk = (np.log((softmax_base**(loads_siilk / f[:, na, na, na, na])).sum(1)) /
                                  np.log(softmax_base) * f[:, na, na, na])

            if 'time' in self.dims:
                ds = xr.DataArray(
                    np.array(loads_silk)[..., 0],
                    dims=['sensor', 'wt', 'time'],
                    coords={'sensor': wt.loadFunction.output_keys,
                            'm': ('sensor', wt.loadFunction.wohler_exponents, {'description': 'Wohler exponents'}),
                            'wt': self.wt, 'time': self.time, 'wd': self.wd, 'ws': self.ws},
                    attrs={'description': '1Hz Damage Equivalent Load'}).to_dataset(name='DEL')
            else:
                ds = xr.DataArray(
                    loads_silk,
                    dims=['sensor', 'wt', 'wd', 'ws'],
                    coords={'sensor': wt.loadFunction.output_keys,
                            'm': ('sensor', wt.loadFunction.wohler_exponents, {'description': 'Wohler exponents'}),
                            'wt': self.wt, 'wd': self.wd, 'ws': self.ws},
                    attrs={'description': '1Hz Damage Equivalent Load'}).to_dataset(name='DEL')
            f = ds.DEL.mean()   # factor used to reduce numerical errors in power
            if 'time' in self.dims:
                assert 'duration' in self, "Simulation must contain a dataarray 'duration' with length of time steps in seconds"
                t_flowcase = self.duration
                ds['LDEL'] = ((t_flowcase * (ds.DEL / f)**ds.m).sum(('time')) / n_eq_lifetime)**(1 / ds.m) * f
            else:
                ds['P'] = self.P
                t_flowcase = ds.P * 3600 * 24 * 365 * lifetime_years
                ds['LDEL'] = ((t_flowcase * (ds.DEL / f)**ds.m).sum(('wd', 'ws')) / n_eq_lifetime)**(1 / ds.m) * f
            ds.LDEL.attrs['description'] = "Lifetime (%d years) equivalent loads, n_eq_L=%d" % (
                lifetime_years, n_eq_lifetime)

        return ds

    def noise_model(self, noiseModel=ISONoiseModel):
        WS_eff_ilk = self.WS_eff_ilk
        freqs, sound_power_level = self.windFarmModel.windTurbines.sound_power_level(WS_eff_ilk, **self.wt_kwargs)
        return noiseModel(src_x=self.x.values, src_y=self.y.values, src_h=np.array(self.h.values),
                          freqs=freqs, sound_power_level=sound_power_level,
                          elevation_function=self.windFarmModel.site.elevation)

    def noise_map(self, noiseModel=ISONoiseModel, grid=None, ground_type=0, temperature=20, relative_humidity=80):
        if grid is None:
            grid = HorizontalGrid(h=2)
        nm = self.noise_model(noiseModel)
        X, Y, x_j, y_j, h_j, _ = self._get_grid(grid)
        spl_jlk, spl_jlkf = nm(x_j, y_j, h_j, temperature, relative_humidity, ground_type=ground_type)
        return xr.Dataset({'Total sound pressure level': (('y', 'x', 'wd', 'ws'),
                                                          spl_jlk.reshape(X.shape + (spl_jlk.shape[1:]))),
                           'Sound pressure level': (('y', 'x', 'wd', 'ws', 'freq'),
                                                    spl_jlkf.reshape(X.shape + (spl_jlkf.shape[1:])))},
                          coords={'x': X[0], 'y': Y[:, 0], 'wd': self.wd, 'ws': self.ws, 'freq': nm.freqs})

    def flow_box(self, x, y, h, wd=None, ws=None, time=None):
        X, Y, H = np.meshgrid(x, y, h)
        x_j, y_j, h_j = X.flatten(), Y.flatten(), H.flatten()

        if time is not None:

            lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(x_j, y_j, h_j, self.sel(time=time))
        else:
            wd, ws = self._wd_ws(wd, ws)
            lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(x_j, y_j, h_j, self.sel(wd=wd, ws=ws))

        return FlowBox(self, X, Y, H, lw_j, WS_eff_jlk, TI_eff_jlk)

    def _get_grid(self, grid):
        if grid is None:
            grid = HorizontalGrid()
        if isinstance(grid, Grid):
            plane = grid.plane
            if 'h' not in self.dims or len(np.atleast_1d(self.h.values)) == 0:
                h = self.windFarmModel.windTurbines.hub_height()

            grid = grid(x_i=self.x.values, y_i=self.y.values, h_i=h,
                        d_i=self.windFarmModel.windTurbines.diameter(self.type))
        else:
            plane = (None,)
        return grid + (plane, )

    @property
    def wt_kwargs(self):
        wt_kwargs = {}
        for opt, lst in zip([False, True], self.windFarmModel.windTurbines.function_inputs):
            for k in lst:
                if k not in wt_kwargs:
                    if k in self:
                        wt_kwargs[k] = self[k].ilk()
                    elif k in {'dw_ijlk', 'hcw_ijlk', 'cw_ijlk', 'dh_ijlk'}:
                        z_ilk = self.windFarmModel.site.elevation(self.x.ilk(), self.y.ilk())
                        self.windFarmModel.site.distance.setup(self.x.ilk(), self.y.ilk(), self.h.ilk(), z_ilk)
                        dist = {k: v for k, v in zip(['dw_ijlk', 'hcw_ijlk', 'cw_ijlk'],
                                                     self.windFarmModel.site.distance(WD_ilk=self.WD.ilk(),
                                                                                      wd_l=self.wd.values))}
                        wt_kwargs.update({k: v for k, v in dist.items() if k in lst})
                        if k == 'cw_ijlk':  # pragma: no cover
                            raise NotImplementedError()
                    elif not opt:  # pragma: no cover
                        # should never come here
                        raise KeyError(f"Argument, {k}, required to calculate power and ct not found")
        return wt_kwargs

    def aep_map(self, grid=None, wd=None, ws=None, type=0, normalize_probabilities=False, n_cpu=1, wd_chunks=None):  # @ReservedAssignment
        X, Y, x_j, y_j, h_j, plane = self._get_grid(grid)
        wd, ws = self._wd_ws(wd, ws)
        sim_res = self.sel(wd=wd, ws=ws)
        for k in self.__slots__:
            setattr(sim_res, k, getattr(self, k))
        n_cpu = n_cpu or multiprocessing.cpu_count()
        wd_chunks = np.minimum(wd_chunks or n_cpu, len(wd))
        if n_cpu != 1:
            n_cpu = n_cpu or multiprocessing.cpu_count()
            map = get_pool_starmap(n_cpu)  # @ReservedAssignment
            if len(wd) >= n_cpu:
                # chunkification more efficient on wd than j
                wd_i = np.linspace(0, len(wd), n_cpu + 1).astype(int)
                args_lst = [[x_j, y_j, h_j, type, sim_res.sel(wd=wd[i0:i1])] for i0, i1 in zip(wd_i[:-1], wd_i[1:])]
                aep_lst = map(self.windFarmModel._aep_map, args_lst)
                aep_j = np.sum(aep_lst, 0)
            else:
                j_i = np.linspace(0, len(x_j), n_cpu + 1).astype(int)
                args_lst = [[xyh_j[i0:i1] for xyh_j in [x_j, y_j, h_j]] + [type, sim_res]
                            for i0, i1 in zip(j_i[:-1], j_i[1:])]
                aep_lst = map(self.windFarmModel._aep_map, args_lst)
                aep_j = np.concatenate(aep_lst)
        else:
            aep_j = self.windFarmModel._aep_map(x_j, y_j, h_j, type, sim_res)
        if normalize_probabilities:
            lw_j = self.windFarmModel.site.local_wind(x=x_j, y=y_j, h=h_j, wd=wd, ws=ws)
            aep_j /= lw_j.P_ilk.sum((1, 2))

        if plane[0] == 'XY':
            coords = {'x': X[0], 'y': Y[:, 0]}
            return xr.DataArray(aep_j.reshape(X.shape), name='AEP', attrs={
                                'units': 'GWh'}, coords=coords, dims=['y', 'x'])
        elif plane[0] == 'xyz':
            return xr.DataArray(aep_j, name='AEP', attrs={'units': 'GWh'}, coords={
                                'x': ('i', grid.x), 'y': ('i', grid.y)}, dims=['i'])
        else:  # pragma: no cover
            raise NotImplementedError()

    def flow_map(self, grid=None, wd=None, ws=None, time=None, D_dst=0):
        """Return a FlowMap object with WS_eff and TI_eff of all grid points

        Parameters
        ----------
        grid : Grid or tuple(X, Y, x, y, h)
            Grid, e.g. HorizontalGrid or\n
            tuple(X, Y, x, y, h) where X, Y is the meshgrid for visualizing data\n
            and x, y, h are the flattened grid points
        wd : int, float, array_like or None
            Wind directions to include in the flow map (if more than one, an weighted average will be computed)
            The simulation result must include the requested wind directions.
            If None, an weighted average of all wind directions from the simulation results will be computed.
            Note, computing a flow map with multiple wind directions may be slow
        ws : int, float, array_like or None
            Same as "wd", but for wind speed
        ws : int, array_like or None
            Same as "wd", but for time
        D_dst : int, float or None
            In combination with a rotor average model, D_dst defines the downstream rotor diameter
            at which the deficits will be averaged

        See Also
        --------
        pywake.wind_farm_models.flow_map.FlowMap
        """
        X, Y, x_j, y_j, h_j, plane = self._get_grid(grid)
        wd, ws = self._wd_ws(wd, ws)
        if 'time' in self:
            sim_res = self.sel(time=(time, slice(time))[time is None])
        else:
            sim_res = self.sel(wd=wd, ws=ws)
        for k in self.__slots__:
            setattr(sim_res, k, getattr(self, k))

        lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(x_j, y_j, h_j, sim_res, D_dst=D_dst)
        return FlowMap(sim_res, X, Y, lw_j, WS_eff_jlk, TI_eff_jlk, plane=plane)

    def _wd_ws(self, wd, ws):
        if wd is None:
            wd = self.wd
        else:
            assert np.all(np.isin(wd, self.wd)), "All wd=%s not in simulation result" % wd
        if ws is None:
            ws = self.ws
        else:
            assert np.all(np.isin(ws, self.ws)), "All ws=%s not in simulation result (ws=%s)" % (ws, self.ws)
        return np.atleast_1d(wd), np.atleast_1d(ws)

    def save(self, filename):
        self.to_netcdf(filename)

    @staticmethod
    def load(filename, wfm):
        ds = xr.load_dataset(filename)
        if 'time' in ds:
            time = ds.time.data
        else:
            time = False
        lw = LocalWind(ds.x.data, ds.y.data, ds.h.data, ds.wd.data, ds.ws.data, time,
                       wd_bin_size=ds['wd_bin_size'],
                       WD=ds.WD, WS=ds.WS, TI=ds.TI, P=ds.P)

        sim_res = SimulationResult(wfm, lw, type_i=ds.type.values,
                                   WS_eff_ilk=ds.WS_eff.ilk(), TI_eff_ilk=ds.TI_eff.ilk(), power_ilk=ds.Power.ilk(), ct_ilk=ds.CT.ilk(),
                                   **{k: v.ilk() for k, v in ds.items()
                                      if k not in {'wd_bin_size', 'ws_l', 'ws_u', 'WS_eff', 'TI_eff', 'Power', 'CT'}})
        for k, v in ds.items():
            if k not in sim_res:
                sim_res[k] = v

        return sim_res

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        res = xr.Dataset.sel(self, indexers=indexers, method=method, tolerance=tolerance, drop=drop, **indexers_kwargs)
        for n in self.__slots__:
            setattr(res, n, getattr(self, n, None))
        return res


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
        from py_wake import IEA37SimpleBastankhahGaussian

        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
        simulation_result = wind_farm_model(x, y)
        fm = simulation_result.flow_map(wd=30)
        fm.plot_wake_map()

        plt.figure()
        fm.plot(fm.power_xylk().sum(['wd', 'ws']) * 1e-3, "Power [kW]")

        fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=50))
        plt.figure()
        fm.plot(fm.aep_xy(), "AEP [GWh]")
        plt.show()


main()
