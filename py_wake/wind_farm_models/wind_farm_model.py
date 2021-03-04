from abc import abstractmethod, ABC
from py_wake.site._site import Site, UniformSite, UniformWeibullSite
from py_wake.wind_turbines import WindTurbines
import numpy as np
from py_wake.flow_map import FlowMap, HorizontalGrid, FlowBox, YZGrid, Grid, Points
import xarray as xr
from py_wake.utils import xarray_utils, weibull  # register ilk function @UnusedImport
from numpy import newaxis as na


class WindFarmModel(ABC):
    """Base class for RANS and engineering flow models"""
    verbose = True

    def __init__(self, site, windTurbines):
        assert isinstance(site, Site)
        assert isinstance(windTurbines, WindTurbines)
        self.site = site
        self.windTurbines = windTurbines

    def __call__(self, x, y, h=None, type=0, wd=None, ws=None, yaw_ilk=None, verbose=False, **kwargs):
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
        assert len(x) == len(y)
        self.verbose = verbose
        h, _ = self.windTurbines.get_defaults(len(x), type, h)

        if len(x) == 0:
            lw = UniformSite([1], 0.1).local_wind(x_i=[], y_i=[], h_i=[], wd=wd, ws=ws)
            z = xr.DataArray(np.zeros((0, len(lw.wd), len(lw.ws))), coords=[('wt', []), ('wd', lw.wd), ('ws', lw.ws)])
            return SimulationResult(self, lw, [], yaw_ilk, z, z, z, z, kwargs)
        res = self.calc_wt_interaction(x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk, wd=wd, ws=ws, **kwargs)
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, power_ct_inputs = res
        return SimulationResult(self, localWind=localWind,
                                type_i=np.zeros(len(x), dtype=int) + type, yaw_ilk=yaw_ilk,

                                WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
                                power_ilk=power_ilk, ct_ilk=ct_ilk, power_ct_inputs=power_ct_inputs)

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
    def calc_wt_interaction(self, x_i, y_i, h_i=None, type_i=None, yaw_ilk=None, wd=None, ws=None, **kwargs):
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
            Turbulence intensities. Should be effective, but not implemented yet
        power_ilk : array_like
            Power productions [w]
        ct_ilk : array_like
            Thrust coefficients
        localWind : LocalWind
            Local free-flow wind
        """


class SimulationResult(xr.Dataset):
    """Simulation result returned when calling a WindFarmModel object"""
    __slots__ = ('windFarmModel', 'localWind', 'power_ct_inputs')

    def __init__(self, windFarmModel, localWind, type_i, yaw_ilk,
                 WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, power_ct_inputs):
        self.windFarmModel = windFarmModel
        lw = localWind
        self.localWind = localWind
        self.power_ct_inputs = power_ct_inputs
        n_wt = len(lw.i)

        coords = {k: (k, v, {'Description': d}) for k, v, d in [
            ('wt', np.arange(n_wt), 'Wind turbine number'),
            ('wd', lw.wd, 'Ambient reference wind direction [deg]'),
            ('ws', lw.ws, 'Ambient reference wind speed [m/s]')]}
        coords.update({k: ('wt', v, {'Description': d}) for k, v, d in [
            ('x', lw.x, 'Wind turbine x coordinate [m]'),
            ('y', lw.y, 'Wind turbine y coordinate [m]'),
            ('h', lw.h, 'Wind turbine hub height [m]'),
            ('type', type_i, 'Wind turbine type')]})
        xr.Dataset.__init__(self,
                            data_vars={k: (['wt', 'wd', 'ws'], v, {'Description': d}) for k, v, d in [
                                ('WS_eff', WS_eff_ilk, 'Effective local wind speed [m/s]'),
                                ('TI_eff', np.zeros_like(WS_eff_ilk) + TI_eff_ilk,
                                 'Effective local turbulence intensity'),
                                ('Power', power_ilk, 'Power [W]'),
                                ('CT', ct_ilk, 'Thrust coefficient'),
                            ]},
                            coords=coords)
        for n in localWind:
            self[n] = localWind[n]
        self.attrs.update(localWind.attrs)

        if yaw_ilk is None:
            self['Yaw'] = self.Power * 0
        else:
            self['Yaw'] = xr.DataArray(yaw_ilk, dims=['wt', 'wd', 'ws'])
        self['Yaw'].attrs['Description'] = 'Yaw misalignment [deg]'

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
            power_ilk = self.windFarmModel.windTurbines.power(self.WS.ilk(self.Power.shape), **self.power_ct_inputs)

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
            return xr.DataArray(aep, [('wt', self.wt), ('wd', self.wd), ('ws', ws)])
        else:
            weighted_power = power_ilk * self.P.ilk() / norm

        return xr.DataArray(weighted_power * hours_pr_year * 1e-9,
                            self.Power.coords,
                            name='AEP [GWh]',
                            attrs={'Description': 'Annual energy production [GWh]'})

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
