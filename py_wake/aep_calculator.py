import warnings
from py_wake.flow_map import HorizontalGrid


class AEPCalculator():

    def __init__(self, wake_model):
        """Initialize AEPCalculator

        Parameters
        ----------
        site : py_wake.site.Site
        windTurbines : WindTurbines
        flow_model : FlowModel
        """
        warnings.warn("""AEPCalculator(wake_model) is deprecated;
wake_model(x_i, y_i, ...) returns a flowModelResult with same functionality as AEPCalculator.""",
                      DeprecationWarning, stacklevel=2)
        self.wake_model = wake_model
        self.site = wake_model.site
        self.windTurbines = wake_model.windTurbines

    def _set_flowModelResult(self, flowModelResult):
        for n in ['WS_eff_ilk', 'TI_eff_ilk', 'power_ilk', 'ct_ilk']:
            setattr(self, n, getattr(flowModelResult, n))
        for n in ['WD_ilk', 'WS_ilk', 'TI_ilk', 'P_ilk']:
            setattr(self, n, getattr(flowModelResult, n))

    def calculate_AEP(self, x_i, y_i, h_i=None, type_i=0, wd=None, ws=None):
        """Calculate AEP

        In addition effective wind speed, turbulence intensity, and the
        power, ct and probability is calculated

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i int, array_like, optional, default is 0
            Wind turbine types\n
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        AEP_GWh_ilk : array_like
            AEP in GWh
        """
        flowModelResult = self.wake_model(x=x_i, y=y_i, h=h_i, type=type_i, wd=wd, ws=ws)
        self._set_flowModelResult(flowModelResult)
        return flowModelResult.aep_ilk()

    def calculate_AEP_no_wake_loss(self, x_i, y_i, h_i=None, type_i=0, wd=None, ws=None):
        """Calculate AEP without wake loss(GWh). Same input as calculate_AEP"""
        flowModelResult = self.wake_model(x=x_i, y=y_i, h=h_i, type=type_i, wd=wd, ws=ws)
        self._set_flowModelResult(flowModelResult)
        return flowModelResult.aep_ilk(with_wake_loss=False)

    def wake_map(self, x_j=None, y_j=None, height_level=None, wt_x=[],
                 wt_y=[], wt_type=0, wt_height=None, wd=None, ws=None):
        """Calculate wake(effective wind speed) map

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        height_level : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x : array_like, optional
            X position of wind turbines
        wt_y : array_like, optional
            Y position of wind turbines
        wt_type : array_like or None, optional
            Type of the wind turbines
        wt_height : array_like or None, optional
            Hub height of the wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        X_j : array_like
            2d array of map x positions
        Y_j : array_like
            2d array of map y positions
        WS_eff_avg : array_like
            2d array of average effective local wind speed taking into account
            the probability of wind direction and speed

        See Also
        --------
        plot_wake_map
        """

        sim_res = self.wake_model(x=wt_x, y=wt_y, type=wt_type, h=wt_height, wd=wd, ws=ws)
        flow_map = sim_res.flow_map(HorizontalGrid(x=x_j, y=y_j, h=height_level))
        X, Y = flow_map.XY
        return X, Y, flow_map.WS_eff_xylk.mean(['wd', 'ws'])

    def ti_map(self, x_j=None, y_j=None, height_level=None, wt_x=[],
               wt_y=[], wt_type=0, wt_height=None, wd=None, ws=None):
        """Calculate turbulence intensity map

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        height_level : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x : array_like, optional
            X position of wind turbines
        wt_y : array_like, optional
            Y position of wind turbines
        wt_type : array_like or None, optional
            Type of the wind turbines
        wt_height : array_like or None, optional
            Hub height of the wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        X_j : array_like
            2d array of map x positions
        Y_j : array_like
            2d array of map y positions
        WS_eff_avg : array_like
            2d array of average effective local wind speed taking into account
            the probability of wind direction and speed

        See Also
        --------
        plot_wake_map
        """
        sim_res = self.wake_model(x=wt_x, y=wt_y, type=wt_type, h=wt_height, wd=wd, ws=ws)
        flow_map = sim_res.flow_map(HorizontalGrid(x=x_j, y=y_j, h=height_level))
        X, Y = flow_map.XY
        return X, Y, flow_map.TI_eff_xylk.mean(['wd', 'ws'])

    def plot_wake_map(self, x_j=None, y_j=None, h=None, wt_x=[], wt_y=[], wt_type=0, wt_height=None,
                      wd=None, ws=None, ax=None, levels=100):
        """Plot wake(effective wind speed) map

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        h : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x : array_like, optional
            X position of wind turbines
        wt_y : array_like, optional
            Y position of wind turbines
        wt_type : array_like or None, optional
            Type of the wind turbines
        wt_height : array_like or None, optional
            Hub height of the wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws
        ax : pyplot or matplotlib axes object, default None
            Axes to plot on
        levels : int or array_like
            levels for pyplot.contourf
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        X, Y, Z = self.wake_map(x_j, y_j, h, wt_x, wt_y, wt_type, wt_height, wd, ws)
        c = ax.contourf(X, Y, Z, levels, cmap='Blues_r')
        plt.colorbar(c, label='wind speed [m/s]')

    def aep_map(self, x_j=None, y_j=None, type_j=None, wt_x=[], wt_y=[], wt_type=0, wt_height=None, wd=None, ws=None):
        """Calculate AEP map

        The map represents the of AEP produced by a new turbine at the specified positions

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points (potential turbine positions)
        y_j : array_like
            Y position of map points (potential turbine positions)
        type_j : int, float or None, optional
            Type of potential turbine positions\n
            If None, default, first turbine type(0) is used
        wt_x : array_like, optional
            X position of the current wind turbines
        wt_y : array_like, optional
            Y position of the current wind turbines
        wt_type : array_like or None, optional
            Type of the current wind turbines
        wt_height : array_like or None, optional
            Hub height of the current wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        X_j : array_like
            2d array of map x positions
        Y_j : array_like
            2d array of map y positions
        WS_eff_avg : array_like
            2d array of average effective local wind speed taking into account
            the probability of wind direction and speed
        """

        h_j = self.windTurbines.hub_height(type_j)
        sim_res = self.wake_model(x=wt_x, y=wt_y, type=wt_type, h=wt_height, wd=wd, ws=ws)
        flow_map = sim_res.flow_map(HorizontalGrid(x=x_j, y=y_j, h=h_j))
        X, Y = flow_map.XY
        aep_xy = flow_map.aep_xy(normalize_probabilities=True)
        return X, Y, aep_xy


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake import NOJ

        # setup site, turbines and flow model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        wake_model = NOJ(site, windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        print(aep_calculator.WS_eff_ilk.shape)

        # plot wake map
        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
