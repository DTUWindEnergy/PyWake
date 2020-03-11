import numpy as np


class FlowMap():
    def __init__(self, simulationResult, X, Y, localWind_j, WS_eff_jlk, TI_eff_jlk, wd, ws, yaw_ilk):
        self.simulationResult = simulationResult
        self.X = X
        self.Y = Y
        self.localWind_j = localWind_j
        self.lw_j = localWind_j
        self.WS_eff_xylk = WS_eff_jlk.reshape(X.shape + WS_eff_jlk.shape[1:])
        self.TI_eff_xylk = TI_eff_jlk.reshape(X.shape + WS_eff_jlk.shape[1:])
        self.wd = wd
        self.ws = ws
        self.yaw_ilk = yaw_ilk
        self.windFarmModel = self.simulationResult.windFarmModel

    @property
    def XY(self):
        return self.X, self.Y

    def power_xylk(self, wt_type=0, with_wake_loss=True):
        if with_wake_loss:
            return self.windFarmModel.windTurbines.power(self.WS_eff_xylk, wt_type)
        else:
            return self.windFarmModel.windTurbines.power(self.lw_j.WS_ilk.reshape(self.WS_eff_xylk.shape), wt_type)

    def aep_xylk(self, wt_type=0, normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production of a potential wind turbine at all grid positions (x,y)
        for all wind directions (l) and wind speeds (k) in GWh.

        Parameters
        ----------
        wt_type : Optional int, defaults to 0
            Type of potential wind turbine
        normalize_propabilities : Optional bool, defaults to False
            In case only a subset of all wind speeds and/or wind directions is simulated,
            this parameter determines whether the returned AEP represents the energy produced in the fraction
            of a year where these flow cases occurs or a whole year of northern wind.
            If for example, wd=[0], then
            - False means that the AEP only includes energy from the faction of year\n
            with northern wind (359.5-0.5deg), i.e. no power is produced the rest of the year.
            - True means that the AEP represents a whole year of northen wind.
            default is False
        with_wake_loss : Optional bool, defaults to True
            If True, wake loss is included, i.e. power is calculated using local effective wind speed\n
            If False, wake loss is neglected, i.e. power is calculated using local free flow wind speed
         """
        power_xylk = self.power_xylk(wt_type, with_wake_loss)
        P_jlk = self.lw_j.P_ilk
        if normalize_probabilities:
            P_jlk /= P_jlk.sum()
        if P_jlk.shape[0] == 1:
            P_xylk = P_jlk[np.newaxis]
#         else:
#             P_xylk = P_jlk.reshape(power_xylk.shape)
        return power_xylk * P_xylk * 24 * 365 * 1e-9

    def aep_xy(self, wt_type=0, normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production of a potential wind turbine at all grid positions (x,y)
        (sum of all wind directions and wind speeds)  in GWh.

        see aep_xylk
        """
        return self.aep_xylk(wt_type, normalize_probabilities, with_wake_loss).sum((2, 3))

    def plot(self, data, clabel, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True, ax=None):
        """Plot data as contouf map

        Parameters
        ----------
        data : array_like
            2D data array to plot
        clabel : str
            colorbar label
        levels : int or array-like, default 100
            Determines the number and positions of the contour lines / regions.
            If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.
            If array-like, draw contour lines at the specified levels. The values must be in increasing order.
        cmap : str or Colormap, defaults 'Blues_r'.
            A Colormap instance or registered colormap name.
            The colormap maps the level values to colors.
        plot_colorbar : bool, default True
            if True (default), colorbar is drawn
        plot_windturbines : bool, default True
            if True (default), lines/circles showing the wind turbine rotors are plotted
        ax : pyplot or matplotlib axes object, default None
        """
        import matplotlib.pyplot as plt
        if cmap is None:
            cmap = 'Blues_r'
        if ax is None:
            ax = plt.gca()
        c = ax.contourf(self.X, self.Y, data.reshape(self.X.shape), levels=levels, cmap=cmap)
        if plot_colorbar:
            plt.colorbar(c, label=clabel)
        if plot_windturbines:
            self.plot_windturbines(ax=ax)
        return c

    def plot_windturbines(self, ax=None):
        fm = self.windFarmModel
        if self.yaw_ilk is None:
            yaw_ilk = 0
        else:
            yaw_ilk = self.yaw_ilk.mean((1, 2))
        fm.windTurbines.plot(self.simulationResult.x_i, self.simulationResult.y_i,
                             wd=self.wd, yaw=yaw_ilk, ax=ax)

    def plot_wake_map(self, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True, ax=None):
        """Plot effective wind speed contourf map

        Parameters
        ----------
        levels : int or array-like, default 100
            Determines the number and positions of the contour lines / regions.
            If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.
            If array-like, draw contour lines at the specified levels. The values must be in increasing order.
        cmap : str or Colormap, defaults 'Blues_r'.
            A Colormap instance or registered colormap name.
            The colormap maps the level values to colors.
        plot_colorbar : bool, default True
            if True (default), colorbar is drawn
        plot_windturbines : bool, default True
            if True (default), lines/circles showing the wind turbine rotors are plotted
        ax : pyplot or matplotlib axes object, default None
        """
        return self.plot(self.WS_eff_xylk.mean((2, 3)), clabel='wind speed [m/s]',
                         levels=levels, cmap=cmap, plot_colorbar=plot_colorbar,
                         plot_windturbines=plot_windturbines, ax=ax)

    def plot_ti_map(self, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True, ax=None):
        """Plot effective turbulence intensity contourf map

        Parameters
        ----------
        levels : int or array-like, default 100
            Determines the number and positions of the contour lines / regions.
            If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.
            If array-like, draw contour lines at the specified levels. The values must be in increasing order.
        cmap : str or Colormap, defaults 'Blues'.
            A Colormap instance or registered colormap name.
            The colormap maps the level values to colors.
        plot_colorbar : bool, default True
            if True (default), colorbar is drawn
        plot_windturbines : bool, default True
            if True (default), lines/circles showing the wind turbine rotors are plotted
        ax : pyplot or matplotlib axes object, default None

        """
        if cmap is None:
            cmap = 'Blues'
        return self.plot(self.TI_eff_xylk.mean((2, 3)), clabel="Turbulence intensity [-]",
                         levels=levels, cmap=cmap, plot_colorbar=plot_colorbar,
                         plot_windturbines=plot_windturbines, ax=ax)


class HorizontalGrid():
    default_resolution = 500

    def __init__(self, x=None, y=None, h=None, resolution=None, extend=.2):
        """Generate a horizontal grid for a flow map

        Parameters
        ----------
        x : array_like, optional
            x coordinates used for generating meshgrid\n
        y : array_like, optional
            y coordinates used for generating meshgrid
        h : array_like, optional
            height above ground, defaults to mean wind turbine hub height
        resolution : int or None, optional
            grid resolution if x or y is not specified. defaults to self.default_resolution
        extend : float, optional
            defines the oversize of the grid if x or y is not specified

        Notes
        -----
        if x or y is not specified then a grid with <resolution> number of points
        covering the wind turbines + <extend> x range
        """
        self.resolution = resolution or self.default_resolution
        self.x = x
        self.y = y
        self.h = h
        self.extend = extend

    def __call__(self, x_i, y_i, h_i):
        # setup horizontal X,Y grid
        def f(x, N=self.resolution, ext=self.extend):
            ext *= max(1000, (max(x) - min(x)))
            return np.linspace(min(x) - ext, max(x) + ext, N)
        x, y, h = self.x, self.y, self.h
        if x is None:
            x = f(x_i)
        if y is None:
            y = f(y_i)
        if self.h is None:
            h = np.mean(h_i)
        else:
            h = self.h

        X, Y = np.meshgrid(x, y)
        H = np.zeros_like(X) + h
        return X, Y, X.flatten(), Y.flatten(), H.flatten()
