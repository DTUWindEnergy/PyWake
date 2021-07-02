import numpy as np
import xarray as xr
from numpy import newaxis as na
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


class FlowBox(xr.Dataset):
    __slots__ = ('simulationResult', 'windFarmModel')

    def __init__(self, simulationResult, X, Y, H, localWind_j, WS_eff_jlk, TI_eff_jlk):
        self.simulationResult = simulationResult
        self.windFarmModel = self.simulationResult.windFarmModel
        lw_j = localWind_j
        wd, ws = lw_j.wd, lw_j.ws

        if X is None and Y is None and H is None:
            coords = localWind_j.coords
            X = localWind_j.i
        else:
            coords = {'x': X[0, :, 0], 'y': Y[:, 0, 0], 'h': H[0, 0, :], 'wd': wd, 'ws': ws}

        def get_da(arr_jlk):
            if len(X.shape) == 1:
                return xr.DataArray(arr_jlk.reshape(X.shape + (len(wd), len(ws))), coords, dims=['i', 'wd', 'ws'])
            else:
                return xr.DataArray(arr_jlk.reshape(X.shape + (len(wd), len(ws))),
                                    coords, dims=['y', 'x', 'h', 'wd', 'ws'])
        JLK = WS_eff_jlk.shape
        xr.Dataset.__init__(self, data_vars={k: get_da(v) for k, v in [
            ('WS_eff', WS_eff_jlk), ('TI_eff', TI_eff_jlk),
            ('WD', lw_j.WD.ilk(JLK)), ('WS', lw_j.WS.ilk(JLK)), ('TI', lw_j.TI.ilk(JLK)), ('P', lw_j.P.ilk(JLK))]})


class FlowMap(FlowBox):
    __slots__ = ('simulationResult', 'windFarmModel', 'X', 'Y', 'plane', 'WS_eff_xylk', 'TI_eff_xylk')

    def __init__(self, simulationResult, X, Y, localWind_j, WS_eff_jlk, TI_eff_jlk, plane):
        self.X = X
        self.Y = Y
        self.plane = plane

        if plane[0] == 'XY':
            X = X[:, :, na]
            Y = Y[:, :, na]
            H = np.reshape(localWind_j.h.data, X.shape)
        elif plane[0] == 'YZ':
            H = Y.T[:, na, :]
            Y = X.T[:, na, :]
            X = np.reshape(localWind_j.x.data, Y.shape)
        elif plane[0] == 'xyz':
            X = None
            Y = None
            H = None
        else:
            raise NotImplementedError()
        FlowBox.__init__(self, simulationResult, X, Y, H, localWind_j, WS_eff_jlk, TI_eff_jlk)

        if plane[0] == "XY":
            # set flowMap.WS_xylk etc.
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                setattr(self.__class__, "%s_xylk" % k, property(lambda self, k=k: self[k].isel(h=0)))
        if plane[0] == "YZ":
            # set flowMap.WS_xylk etc.
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                self[k] = self[k].transpose('h', 'y', ...)
                setattr(self.__class__, "%s_xylk" % k,
                        property(lambda self, k=k: self[k].isel(x=0).transpose('y', 'h', ...)))

    @property
    def XY(self):
        return self.X, self.Y

    def power_xylk(self, wt_type=0, with_wake_loss=True):
        if with_wake_loss:
            ws = self.WS_eff_xylk

        else:
            ws = self.WS_xylk

        type = {'type': wt_type} if wt_type != 0 else {}

        power_xylk = self.windFarmModel.windTurbines.power(ws, **type)
        return xr.DataArray(power_xylk[:, :, na], self.coords, dims=['y', 'x', 'h', 'wd', 'ws'])

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
        P_xylk = self.P_xylk  # .isel.ilk((1,) + power_xylk.shape[2:])
        if normalize_probabilities:
            P_xylk = P_xylk / P_xylk.sum(['wd', 'ws'])
        return power_xylk * P_xylk * 24 * 365 * 1e-9

    def aep_xy(self, wt_type=0, normalize_probabilities=False, with_wake_loss=True):
        """Anual Energy Production of a potential wind turbine at all grid positions (x,y)
        (sum of all wind directions and wind speeds)  in GWh.

        see aep_xylk
        """
        return self.aep_xylk(wt_type, normalize_probabilities, with_wake_loss).sum(['wd', 'ws'])

    def plot(self, data, clabel, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True,
             normalize_with=1, ax=None):
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
        n = normalize_with
        if self.plane[0] == "YZ":
            y = self.X[0]
            x = np.zeros_like(y) + self.plane[1]
            z = self.simulationResult.windFarmModel.site.elevation(x, y)
            c = ax.contourf(self.X, self.Y + z, data.isel(x=0), levels=levels, cmap=cmap)
            if plot_colorbar:
                plt.colorbar(c, label=clabel, ax=ax)
            # plot terrain
            y = np.arange(y.min(), y.max())
            x = np.zeros_like(y) + self.plane[1]
            z = self.simulationResult.windFarmModel.site.elevation(x, y)
            ax.plot(y / n, z / n, 'k')
        else:
            # xarray gives strange levels
            # c = data.isel(h=0).plot(levels=levels, cmap=cmap, ax=ax, add_colorbar=plot_colorbar)
            c = ax.contourf(self.X / n, self.Y / n, data.isel(h=0).data, levels=levels, cmap=cmap)
            if plot_colorbar:
                plt.colorbar(c, label=clabel, ax=ax)

        if plot_windturbines:
            self.plot_windturbines(normalize_with=normalize_with, ax=ax)

        return c

    def plot_windturbines(self, normalize_with=1, ax=None):
        fm = self.windFarmModel
        yaw = self.simulationResult.yaw.sel(wd=self.wd[0]).mean(['ws']).data
        tilt = self.simulationResult.tilt.sel(wd=self.wd[0]).mean(['ws']).data
        if self.plane[0] == "YZ":
            x_i, y_i = self.simulationResult.x.values, self.simulationResult.y.values
            h_i = self.simulationResult.h.values
            z_i = self.simulationResult.windFarmModel.site.elevation(x_i, y_i)
            fm.windTurbines.plot_yz(y_i, z_i, h_i, wd=self.wd, yaw=yaw, tilt=tilt, normalize_with=normalize_with, ax=ax)
        else:  # self.plane[0] == "XY":
            fm.windTurbines.plot_xy(self.simulationResult.x, self.simulationResult.y, self.simulationResult.type.data,
                                    wd=self.wd, yaw=yaw, tilt=tilt, normalize_with=normalize_with, ax=ax)

    def plot_wake_map(self, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True,
                      normalize_with=1, ax=None):
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
        return self.plot((self.WS_eff * self.P / self.P.sum(['wd', 'ws'])).sum(['wd', 'ws']), clabel='wind speed [m/s]',
                         levels=levels, cmap=cmap, plot_colorbar=plot_colorbar,
                         plot_windturbines=plot_windturbines, normalize_with=normalize_with, ax=ax)

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
        c = self.plot(self.TI_eff.mean(['wd', 'ws']), clabel="Turbulence intensity [-]",
                      levels=levels, cmap=cmap, plot_colorbar=plot_colorbar,
                      plot_windturbines=plot_windturbines, ax=ax)

        return c

    def min_WS_eff(self, x=None, h=None):
        if x is None:
            x = self.x
        if h is None:
            h = self.h[0].item()
        WS_eff = self.WS_eff.sel_interp_all(xr.Dataset(coords={'x': x, 'h': h}))
        y = WS_eff.y.values

        def get_min(y, v):
            i = np.argmin(v)
            s = slice(i - 3, i + 4)
            if len(v[s]) < 7 or len(np.unique(v[s])) == 1:
                return np.nan
#             import matplotlib.pyplot as plt
#             plt.plot(y, v)
#             y_ = np.linspace(y[s][0], y[s][-1], 100)
#             plt.plot(y_, InterpolatedUnivariateSpline(y[s], v[s])(y_))
#             plt.axvline(np.interp(0, InterpolatedUnivariateSpline(y[s], v[s]).derivative()(y[s]), y[s]))
#             plt.axvline(0, color='k')
#             plt.show()
            return np.interp(0, InterpolatedUnivariateSpline(y[s], v[s]).derivative()(y[s]), y[s])

        y_min_ws = [get_min(y, ws) for ws in WS_eff.squeeze(['ws', 'wd']).T.values]
        return xr.DataArray(y_min_ws, coords={'x': x, 'h': h}, dims='x')

    def plot_deflection_grid(self, normalize_with=1, ax=None):
        assert self.windFarmModel.deflectionModel is not None
        assert len(self.simulationResult.wt) == 1
        assert len(self.simulationResult.ws) == 1
        assert len(self.simulationResult.wd) == 1
        x, y = self.x, self.y
        y = y[::len(y) // 10]

        X, Y = np.meshgrid(x, y)

        from py_wake.utils.model_utils import get_model_input
        kwargs = get_model_input(self.windFarmModel, X.flatten(), Y.flatten(), ws=self.ws, wd=self.wd,
                                 yaw=self.simulationResult.yaw.ilk())
        dw, hcw, dh = self.windFarmModel.deflectionModel.calc_deflection(**kwargs)
        Yp = -hcw[0, :, 0, 0].reshape(X.shape)
        ax = ax or plt.gca()
        X, Y, Yp = [v / normalize_with for v in [X, Y, Yp]]
        # ax.plot(X[255, :], Y[255, :], 'grey', lw=3)
        for x, y, yp in zip(X, Y, Yp):
            ax.plot(x, y, 'grey', lw=1, zorder=-32)
            ax.plot(x, yp, 'k', lw=1)


class Grid():
    default_resolution = 500


class HorizontalGrid(Grid):

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
        self.plane = "XY", h

    def __call__(self, x_i, y_i, h_i, **_):
        # setup horizontal X,Y grid
        def f(x, N=self.resolution, ext=self.extend):
            ext *= np.max([1000, (np.max(x) - np.min(x))])
            return np.linspace(np.min(x) - ext, np.max(x) + ext, N)
        x, y, h = self.x, self.y, self.h
        if x is None:
            x = f(x_i)
        if y is None:
            y = f(y_i)
        if self.h is None:
            h = np.mean(h_i)
        else:
            h = self.h
        self.plane = "XY", h

        X, Y = np.meshgrid(x, y)
        H = np.broadcast_to(h, X.shape)
        return X, Y, X.flatten(), Y.flatten(), H.flatten()


XYGrid = HorizontalGrid


class YZGrid(Grid):

    def __init__(self, x, y=None, z=None, resolution=None, extend=.2):
        """Generate a vertical grid for a flow map in the yz-plane

        Parameters
        ----------
        x : array_like, optional
            x coordinates for the yz-grid\n
        y : array_like, optional
            y coordinates used for generating meshgrid
        z : array_like, optional
            z coordinates(height above ground) used for generating meshgrid
        resolution : int or None, optional
            grid resolution if x or y is not specified. defaults to self.default_resolution
        extend : float, optional
            defines the oversize of the grid if x or y is not specified

        Notes
        -----
        if y or z is not specified then a grid with <resolution> number of points
        covering the wind turbines + <extend> * range
        """
        self.resolution = resolution or self.default_resolution
        self.x = x
        self.y = y
        self.z = z
        self.extend = extend
        self.plane = "YZ", x

    def __call__(self, x_i, y_i, h_i, d_i):
        # setup horizontal X,Y grid
        def f(x, N=self.resolution, ext=self.extend):
            ext *= max(1000, (max(x) - min(x)))
            return np.linspace(min(x) - ext, max(x) + ext, N)
        x, y, z = self.x, self.y, self.z
        if y is None:
            y = f(y_i)
        if self.z is None:
            z = np.arange(0, (1 + self.extend) * (h_i.max() + d_i.max() / 2), np.diff(y[:2])[0])
        else:
            z = self.z

        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y) + x
        return Y, Z, X.T.flatten(), Y.T.flatten(), Z.T.flatten()


class Points(Grid):
    def __init__(self, x, y, h):
        assert len(x) == len(y) == len(h)
        self.x = x
        self.y = y
        self.h = h

    def __call__(self, **_):
        return None, None, self.x, self.y, self.h
