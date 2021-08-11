from abc import abstractmethod, ABC
from py_wake.site._site import Site, UniformSite, UniformWeibullSite, LocalWind
from py_wake.wind_turbines import WindTurbines
import numpy as np
from py_wake.flow_map import FlowMap, HorizontalGrid, FlowBox, YZGrid, Grid, Points
import xarray as xr
from py_wake.utils import xarray_utils, weibull  # register ilk function @UnusedImport
from numpy import newaxis as na
from py_wake.utils.model_utils import check_model, fix_shape
from py_wake.utils.xarray_utils import da2py


class WindFarmModel(ABC):
    """Base class for RANS and engineering flow models"""
    verbose = True

    def __init__(self, site, windTurbines):
        check_model(site, Site, 'site')
        check_model(windTurbines, WindTurbines, 'windTurbines')
        self.site = site
        self.windTurbines = windTurbines

    def __call__(self, x, y, h=None, type=0, wd=None, ws=None, yaw=None,
                 tilt=None, time=False, verbose=False, **kwargs):
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
        yaw_ilk : array_like or None, optional
            Yaw misalignement of turbine(i) for wind direction(l) and wind speed (k)\n
            Positive is counter-clockwise when seen from above

        Returns
        -------
        SimulationResult
        """
        if time is False and np.ndim(wd):
            wd = np.sort(wd)
        assert len(x) == len(y)
        self.verbose = verbose
        h, _ = self.windTurbines.get_defaults(len(x), type, h)
        I, L, K, = len(x), len(np.atleast_1d(wd)), (1, len(np.atleast_1d(ws)))[time is False]
        if len([k for k in kwargs if 'yaw' in k.lower() and k != 'yaw' and not k.startswith('yawc_')]):
            raise ValueError(
                'Custom *yaw*-keyword arguments not allowed to avoid confusion with the default "yaw" keyword')
        yaw_ilk = fix_shape(yaw, (I, L, K), allow_None=True)
        tilt_ilk = fix_shape(tilt, (I, L, K), allow_None=True)

        if len(x) == 0:
            lw = UniformSite([1], 0.1).local_wind(x_i=[], y_i=[], h_i=[], wd=wd, ws=ws)
            z = xr.DataArray(np.zeros((0, len(lw.wd), len(lw.ws))), coords=[('wt', []), ('wd', da2py(lw.wd)),
                                                                            ('ws', da2py(lw.ws))])
            return SimulationResult(self, lw, [], yaw, tilt, z, z, z, z, kwargs)
        res = self.calc_wt_interaction(x_i=np.asarray(x), y_i=np.asarray(y), h_i=h, type_i=type,
                                       yaw_ilk=yaw_ilk, tilt_ilk=tilt_ilk,
                                       wd=wd, ws=ws, time=time, **kwargs)
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, wt_inputs = res

        return SimulationResult(self, localWind=localWind,
                                type_i=np.zeros(len(x), dtype=int) + type,
                                yaw_ilk=yaw_ilk, tilt_ilk=tilt_ilk,
                                WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
                                power_ilk=power_ilk, ct_ilk=ct_ilk, wt_inputs=wt_inputs)

    def aep(self, x, y, h=None, type=0, wd=None, ws=None, yaw_ilk=None,  # @ReservedAssignment
            normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production (sum of all wind turbines, directions and speeds) in GWh.

        the typical use is:
        >> sim_res = windFarmModel(x,y,...)
        >> sim_res.aep()

        This function bypasses the simulation result and returns only the total AEP,
        which makes it slightly faster for small problems.
        >> windFarmModel.aep(x,y,...)

        """
        _, _, power_ilk, _, localWind, power_ct_inputs = self.calc_wt_interaction(
            x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk, wd=wd, ws=ws)
        P_ilk = localWind.P_ilk
        if normalize_probabilities:
            norm = P_ilk.sum((1, 2))[:, na, na]
        else:
            norm = 1

        if with_wake_loss is False:
            power_ilk = self.windTurbines.power(localWind.WS_ilk, **power_ct_inputs)
        return (power_ilk * P_ilk / norm * 24 * 365 * 1e-9).sum()

    @abstractmethod
    def calc_wt_interaction(self, x_i, y_i, h_i=None, type_i=None, yaw_ilk=None,
                            wd=None, ws=None, time=False, **kwargs):
        """Calculate effective wind speed, turbulence intensity,
        power and thrust coefficient, and local site parameters

        Typical users should not call this function directly, but by calling the
        windFarmModel object (invokes the __call__() function above)
        which returns a nice SimulationResult object

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i : array_like or None, optional
            Wind turbine types\n
            If None, default, the first type is used (type=0)
        yaw_ilk : array_like or None, optional
            Yaw misalignement [deg] of turbine(i) for wind direction(l) and wind speed (k)\n
            Positive is counter-clockwise when seen from above
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws


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


class SimulationResult(xr.Dataset):
    """Simulation result returned when calling a WindFarmModel object"""
    __slots__ = ('windFarmModel', 'localWind', 'wt_inputs')

    def __init__(self, windFarmModel, localWind, type_i, yaw_ilk, tilt_ilk,
                 WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, wt_inputs):
        self.windFarmModel = windFarmModel
        lw = localWind
        self.localWind = localWind
        self.wt_inputs = wt_inputs
        n_wt = len(lw.i)

        coords = {k: (dep, v, {'Description': d}) for k, dep, v, d in [
            ('wt', 'wt', np.arange(n_wt), 'Wind turbine number'),
            ('wd', ('wd', 'time')['time' in lw], lw.wd.values, 'Ambient reference wind direction [deg]'),
            ('ws', ('ws', 'time')['time' in lw], lw.ws.values, 'Ambient reference wind speed [m/s]'),
            ('x', 'wt', lw.x.values, 'Wind turbine x coordinate [m]'),
            ('y', 'wt', lw.y.values, 'Wind turbine y coordinate [m]'),
            ('h', 'wt', lw.h.values, 'Wind turbine hub height [m]'),
            ('type', 'wt', type_i, 'Wind turbine type')]}

        ilk_dims = (['wt', 'wd', 'ws'], ['wt', 'time'])['time' in lw]
        xr.Dataset.__init__(self,
                            data_vars={k: (ilk_dims, da2py((v, v[:, :, 0])['time' in lw]),
                                           {'Description': d})
                                       for k, v, d in [('WS_eff', WS_eff_ilk, 'Effective local wind speed [m/s]'),
                                                       ('TI_eff', np.zeros_like(WS_eff_ilk) + TI_eff_ilk,
                                                        'Effective local turbulence intensity'),
                                                       ('Power', power_ilk, 'Power [W]'),
                                                       ('CT', ct_ilk, 'Thrust coefficient'),
                                                       ]},
                            coords=coords)
        for n in localWind:
            self[n] = localWind[n]
        self.attrs.update(localWind.attrs)
        for n in set(wt_inputs) - {'type', 'TI_eff', 'yaw'}:
            if '_ijl' in n:
                self.add_ijlk(n, wt_inputs[n])
            else:
                self.add_ilk(n, wt_inputs[n])

        if yaw_ilk is None:
            self['yaw'] = self.Power * 0
        else:
            self.add_ilk('yaw', yaw_ilk)
        self['yaw'].attrs['Description'] = 'Yaw misalignment [deg]'
        if tilt_ilk is None:
            self['tilt'] = self.Power * 0
        else:
            self.add_ilk('tilt', tilt_ilk)
        self['tilt'].attrs['Description'] = 'Rotor tilt [deg]'

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
            power_ilk = self.windFarmModel.windTurbines.power(self.WS.ilk(self.Power.shape), **self.wt_inputs)

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
        if 'time' in self and weighted_power.shape[2] == 1:
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

        kwargs = self.wt_inputs

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
            ds['P'] = self.P.sum('wd')
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
                    if k[-3:] == 'ijl':
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

    def flow_box(self, x, y, h, wd=None, ws=None):
        X, Y, H = np.meshgrid(x, y, h)
        x_j, y_j, h_j = X.flatten(), Y.flatten(), H.flatten()

        wd, ws = self._wd_ws(wd, ws)
        lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(
            x_j, y_j, h_j,
            self.sel(wd=wd, ws=ws)
        )

        return FlowBox(self, X, Y, H, lw_j, WS_eff_jlk, TI_eff_jlk)

    def flow_map(self, grid=None, wd=None, ws=None):
        """Return a FlowMap object with WS_eff and TI_eff of all grid points

        Parameters
        ----------
        grid : Grid or tuple(X, Y, x, y, h)
            Grid, e.g. HorizontalGrid or\n
            tuple(X, Y, x, y, h) where X, Y is the meshgrid for visualizing data\n
            and x, y, h are the flattened grid points

        See Also
        --------
        pywake.wind_farm_models.flow_map.FlowMap
        """

        if grid is None:
            grid = HorizontalGrid()
        if isinstance(grid, Grid):
            if isinstance(grid, HorizontalGrid):
                plane = "XY", self.h
            if isinstance(grid, YZGrid):
                plane = 'YZ', grid.x
            if isinstance(grid, Points):
                plane = 'xyz', None
            grid = grid(x_i=self.x, y_i=self.y, h_i=self.h,
                        d_i=self.windFarmModel.windTurbines.diameter(self.type))
        else:
            plane = (None,)

        wd, ws = self._wd_ws(wd, ws)
        X, Y, x_j, y_j, h_j = grid

        lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(
            x_j, y_j, h_j,
            self.sel(wd=wd, ws=ws)
        )

        return FlowMap(self, X, Y, lw_j, WS_eff_jlk, TI_eff_jlk, plane=plane)

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
        lw = LocalWind(ds.x, ds.y, ds.h, ds.wd, ds.ws, time=False, wd_bin_size=ds.attrs['wd_bin_size'],
                       WD=ds.WD, WS=ds.WS, TI=ds.TI, P=ds.P)
        sim_res = SimulationResult(wfm, lw, type_i=ds.type.values, yaw_ilk=ds.yaw, tilt_ilk=ds.tilt,
                                   WS_eff_ilk=ds.WS_eff.ilk(), TI_eff_ilk=ds.TI_eff.ilk(), power_ilk=ds.Power, ct_ilk=ds.CT,
                                   wt_inputs={})

        return sim_res


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
