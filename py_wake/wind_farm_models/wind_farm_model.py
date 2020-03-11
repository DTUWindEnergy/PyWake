from abc import abstractmethod, ABC
from py_wake.site._site import Site, LocalWind
from py_wake.wind_turbines import WindTurbines
import numpy as np
from py_wake.flow_map import FlowMap, HorizontalGrid


class WindFarmModel(ABC):
    """Base class for RANS and engineering flow models"""

    def __init__(self, site, windTurbines):
        assert isinstance(site, Site)
        assert isinstance(windTurbines, WindTurbines)
        self.site = site
        self.windTurbines = windTurbines

    def __call__(self, x, y, h=None, type=0, wd=None, ws=None, yaw_ilk=None):
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

        Returns
        -------
        SimulationResult
        """
        assert len(x) == len(y)
        type, h, _ = self.windTurbines.get_defaults(len(x), type, h)
        wd, ws = self.site.get_defaults(wd, ws)

        if len(x) == 0:
            wd, ws = np.atleast_1d(wd), np.atleast_1d(ws)
            z = np.zeros((0, len(wd), len(ws)))
            localWind = LocalWind(z, z, z, z)
            return SimulationResult(self, localWind=localWind,
                                    x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk,
                                    wd=wd, ws=ws,
                                    WS_eff_ilk=z, TI_eff_ilk=z,
                                    power_ilk=z, ct_ilk=z)
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind = self.calc_wt_interaction(
            x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk, wd=wd, ws=ws)
        return SimulationResult(self, localWind=localWind,
                                x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk,
                                wd=wd, ws=ws,
                                WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
                                power_ilk=power_ilk, ct_ilk=ct_ilk)

    @abstractmethod
    def calc_wt_interaction(self, x_i, y_i, h_i=None, type_i=None, yaw_ilk=None, wd=None, ws=None):
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


class SimulationResult():
    """Simulation result returned when calling a WindFarmModel object"""

    def __init__(self, windFarmModel, localWind, x_i, y_i, h_i, type_i, yaw_ilk,
                 wd, ws, WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk):
        self.windFarmModel = windFarmModel
        self.localWind = localWind
        self.x_i = x_i
        self.y_i = y_i
        self.h_i = h_i
        self.type_i = type_i
        self.yaw_ilk = yaw_ilk
        self.WS_eff_ilk = WS_eff_ilk
        self.TI_eff_ilk = TI_eff_ilk
        self.power_ilk = power_ilk
        self.ct_ilk = ct_ilk
        self.wd = wd
        self.ws = ws

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
        P_ilk = self.localWind.P_ilk
        if normalize_probabilities:
            P_ilk /= P_ilk.sum()
        if with_wake_loss:
            return self.power_ilk * P_ilk * 24 * 365 * 1e-9
        else:
            power_ilk = self.windFarmModel.windTurbines.power(self.localWind.WS_ilk, self.type_i)
            return power_ilk * P_ilk * 24 * 365 * 1e-9

    def aep(self, normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production (sum of all wind turbines, directions and speeds) in GWh.

        See aep_ilk
        """
        return self.aep_ilk(normalize_probabilities, with_wake_loss).sum()

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
        if isinstance(grid, HorizontalGrid):
            grid = grid(self.x_i, self.y_i, self.h_i)
        if wd is None:
            wd = self.wd
        else:
            assert np.all(np.isin(wd, self.wd)), "All wd=%s not in simulation result" % wd
        if ws is None:
            ws = self.ws
        else:
            assert np.all(np.isin(ws, self.ws)), "All ws=%s not in simulation result (ws=%s)" % (ws, self.ws)
        wd, ws = np.atleast_1d(wd), np.atleast_1d(ws)
        l_indices = np.argwhere(wd[:, None] == self.wd)[:, 1]
        k_indices = np.argwhere(ws[:, None] == self.ws)[:, 1]
        X, Y, x_j, y_j, h_j = grid
        lw_j, WS_eff_jlk, TI_eff_jlk = self.windFarmModel._flow_map(
            x_j, y_j, h_j,
            self.x_i, self.y_i, self.h_i, self.type_i, self.yaw_ilk,
            self.localWind.WD_ilk[:, l_indices][:, :, k_indices],
            self.localWind.WS_ilk[:, l_indices][:, :, k_indices],
            self.localWind.TI_ilk[:, l_indices][:, :, k_indices],
            self.WS_eff_ilk[:, l_indices][:, :, k_indices],
            self.TI_eff_ilk[:, l_indices][:, :, k_indices],
            self.ct_ilk[:, l_indices][:, :, k_indices],
            wd, ws)
        if self.yaw_ilk is not None:
            yaw_ilk = self.yaw_ilk[:, l_indices][:, :, k_indices]
        else:
            yaw_ilk = None
        return FlowMap(self, X, Y, lw_j, WS_eff_jlk, TI_eff_jlk, wd, ws,
                       yaw_ilk=yaw_ilk)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
        from py_wake import IEA37SimpleBastankhahGaussian

        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        # NOJ wake model
        wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
        simulation_result = wind_farm_model(x, y)
        fm = simulation_result.flow_map(wd=30)
        fm.plot_wake_map()

        plt.figure()
        fm.plot(fm.power_xylk()[:, :, 0, 0] * 1e-3, "Power [kW]")

        fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=50))
        plt.figure()
        fm.plot(fm.aep_xy(), "AEP [GWh]")
        plt.show()


main()
