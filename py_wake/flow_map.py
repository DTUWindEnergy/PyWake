from py_wake import np
import xarray as xr
from numpy import newaxis as na
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from py_wake.utils.xarray_utils import ilk2da


class FlowBox(xr.Dataset):
    __slots__ = ('simulationResult', 'windFarmModel')

    def __init__(self, simulationResult, X, Y, H, localWind_j, WS_eff_jlk, TI_eff_jlk):
        self.simulationResult = simulationResult
        self.windFarmModel = self.simulationResult.windFarmModel
        lw_j = localWind_j
        wd, ws = lw_j.wd, lw_j.ws
        time = 'time' in simulationResult

        if X is None and Y is None and H is None:
            coords = localWind_j.coords
            X = localWind_j.i
        else:
            coords = {'x': X[0, :, 0], 'y': Y[:, 0, 0], 'h': H[0, 0, :]}
            if time:
                coords['time'] = lw_j.coords['time']
        coords.update({k: (dep, v, {'Description': d}) for k, dep, v, d in [
                          ('wd', ('wd', 'time')[time], wd, 'Ambient reference wind direction [deg]'),
                          ('ws', ('ws', 'time')[time], ws, 'Ambient reference wind speed [m/s]')]})

        def get_da(arr_jlk):
            if len(X.shape) == 1:
                return ilk2da(arr_jlk, coords)
            else:
                shape = [X.shape + (len(wd), len(ws)), X.shape + (len(wd), )][time]
                dims = ['y', 'x', 'h'] + [['wd', 'ws'], ['time']][time]
                return xr.DataArray(arr_jlk.reshape(shape), coords, dims)
        JLK = WS_eff_jlk.shape
        xr.Dataset.__init__(self, data_vars={k: get_da(np.broadcast_to(v, JLK)) for k, v in [
            ('WS_eff', WS_eff_jlk), ('TI_eff', TI_eff_jlk),
            ('WD', lw_j.WD_ilk), ('WS', lw_j.WS_ilk), ('TI', lw_j.TI_ilk), ('P', lw_j.P_ilk)]})


class FlowMap(FlowBox):
    __slots__ = ('simulationResult', 'windFarmModel', 'X', 'Y', 'plane', 'WS_eff_xylk', 'TI_eff_xylk')

    def __init__(self, simulationResult, X, Y, localWind_j, WS_eff_jlk, TI_eff_jlk, plane):
        self.X = X
        self.Y = Y
        self.plane = plane

        if plane[0] == 'XY':
            X = X[:, :, na]
            Y = Y[:, :, na]
            H = np.reshape(localWind_j.h, X.shape)
        elif plane[0] == 'YZ':
            H = Y.T[:, na, :]
            Y = X.T[:, na, :]
            X = np.reshape(localWind_j.x, Y.shape)
        elif plane[0] == 'XZ':
            H = Y.T[:, na, :]
            X = X.T[na, :, :]
            Y = np.reshape(localWind_j.y, X.shape)
        elif plane[0] == 'xyz':
            X = None
            Y = None
            H = None
        else:
            raise NotImplementedError()
        FlowBox.__init__(self, simulationResult, X, Y, H, localWind_j, WS_eff_jlk, TI_eff_jlk)

        # set flowMap.WS_xylk etc.
        if plane[0] == "XY":
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                setattr(self.__class__, "%s_xylk" % k, property(lambda self, k=k: self[k].isel(h=0)))
        elif plane[0] == "YZ":
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                self[k] = self[k].transpose('h', 'y', ...)
                setattr(self.__class__, "%s_xylk" % k,
                        property(lambda self, k=k: self[k].isel(x=0).transpose('y', 'h', ...)))
        elif plane[0] == "XZ":
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                self[k] = self[k].transpose('h', 'x', ...)
                setattr(self.__class__, "%s_xylk" % k,
                        property(lambda self, k=k: self[k].isel(x=0).transpose('x', 'h', ...)))
        elif plane[0] == "xyz":
            for k in ['WS_eff', 'TI_eff', 'WS', 'WD', 'TI', 'P']:
                setattr(self.__class__, "%s_xylk" % k, property(lambda self, k=k: self[k].ilk()))

    @property
    def XY(self):
        return self.X, self.Y

    def power_xylk(self, with_wake_loss=True, **wt_kwargs):
        if with_wake_loss:
            ws = self.WS_eff_xylk

        else:
            ws = self.WS_xylk

        power_xylk = self.windFarmModel.windTurbines.power(ws, **wt_kwargs)
        if self.plane[0] == "xyz":
            return power_xylk
        else:
            return xr.DataArray(power_xylk[:, :, na], self.coords, dims=['y', 'x', 'h', 'wd', 'ws'])

    def aep_xylk(self, normalize_probabilities=False, with_wake_loss=True, **wt_kwargs):
        """Anual Energy Production of a potential wind turbine at all grid positions (x,y)
        for all wind directions (l) and wind speeds (k) in GWh.

        Parameters
        ----------
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
        wt_type : Optional arguments
            Additional required/optional arguments needed by the WindTurbines to computer power, e.g. type, Air_density
         """
        power_xylk = self.power_xylk(with_wake_loss, **wt_kwargs)
        P_xylk = self.P_xylk  # .isel.ilk((1,) + power_xylk.shape[2:])
        if normalize_probabilities:
            P_xylk = P_xylk / P_xylk.sum(['wd', 'ws'])
        return power_xylk * P_xylk * 24 * 365 * 1e-9

    def aep_xy(self, normalize_probabilities=False, with_wake_loss=True, **wt_kwargs):
        """Anual Energy Production of a potential wind turbine at all grid positions (x,y)
        (sum of all wind directions and wind speeds)  in GWh.

        see aep_xylk
        """
        return self.aep_xylk(normalize_probabilities, with_wake_loss, **wt_kwargs).sum(['wd', 'ws'])

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
        if cmap is None:
            cmap = 'Blues_r'
        if ax is None:
            ax = plt.gca()
        n = normalize_with
        if self.plane[0] in ['XZ', 'YZ']:
            if self.plane[0] == "YZ":
                y = self.y.values
                x = np.zeros_like(y) + self.plane[1]
                z = self.simulationResult.windFarmModel.site.elevation(x, y)
                c = ax.contourf(self.X, self.Y + z, data.isel(x=0), levels=levels, cmap=cmap)
            elif self.plane[0] == 'XZ':
                x = self.x.values
                y = np.zeros_like(x) + self.plane[1]
                z = self.simulationResult.windFarmModel.site.elevation(x, y)
                c = ax.contourf(self.X, self.Y + z, data.isel(y=0), levels=levels, cmap=cmap)

            if plot_colorbar:
                plt.colorbar(c, label=clabel, ax=ax)
            # plot terrain
            y = np.arange(y.min(), y.max())
            x = np.zeros_like(y) + self.plane[1]
            z = self.simulationResult.windFarmModel.site.elevation(x, y)
            ax.plot(y / n, z / n, 'k')
        elif self.plane[0] == 'XY':

            # xarray gives strange levels
            # c = data.isel(h=0).plot(levels=levels, cmap=cmap, ax=ax, add_colorbar=plot_colorbar)
            c = ax.contourf(self.X / n, self.Y / n, data.isel(h=0).data, levels=levels, cmap=cmap)
            if plot_colorbar:
                plt.colorbar(c, label=clabel, ax=ax)
        else:
            raise NotImplementedError(
                f"Plot not supported for FlowMaps based on Points. Use XYGrid, YZGrid or XZGrid instead")

        if plot_windturbines:
            self.plot_windturbines(normalize_with=normalize_with, ax=ax)

        return c

    def plot_windturbines(self, normalize_with=1, ax=None):
        fm = self.windFarmModel

        x_i, y_i = self.simulationResult.x.values, self.simulationResult.y.values
        type_i = self.simulationResult.type.data
        if self.plane[0] in ['XZ', "YZ"]:
            h_i = self.simulationResult.h.values
            x_ilk, y_ilk = self.simulationResult.x.ilk(), self.simulationResult.y.ilk()

            z_ilk = self.simulationResult.windFarmModel.site.elevation(x_ilk, y_ilk)
            for l in range(x_ilk.shape[1]):
                for k in range(x_ilk.shape[2]):

                    wd = self.wd.isel(wd=l) - 90

                    def get(n):
                        shape = len(x_ilk), len(self.simulationResult.wd), len(self.simulationResult.ws)
                        if n not in self.simulationResult:
                            return 0
                        v = self.simulationResult[n].ilk(shape)
                        return v[:, l, k]
                    yaw, tilt = get('yaw'), get('tilt')
                    if self.plane[0] == 'XZ':
                        fm.windTurbines.plot_yz(x_ilk[:, l, k], z_ilk[:, l, k], h_i, types=type_i, wd=wd, yaw=yaw, tilt=tilt,
                                                normalize_with=normalize_with, ax=ax)
                    else:
                        fm.windTurbines.plot_yz(y_i, z_ilk[:, l, k], h_i, types=type_i, wd=self.wd, yaw=yaw, tilt=tilt,
                                                normalize_with=normalize_with, ax=ax)
        else:  # self.plane[0] == "XY":
            def get(k):
                if k not in self.simulationResult:
                    return 0
                v = self.simulationResult[k]
                if 'wd' in v.dims:
                    v = v.sel(wd=self.wd[0])
                if 'ws' in v.dims:
                    v = v.mean(['ws'])
                return v.data

            yaw, tilt = get('yaw'), get('tilt')
            fm.windTurbines.plot_xy(x_i, y_i, type_i,
                                    wd=self.wd.values, yaw=yaw, tilt=tilt, normalize_with=normalize_with, ax=ax)

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
        sum_dims = [d for d in ['wd', 'time', 'ws'] if d in self.P.dims]
        WS_eff = (self.WS_eff * self.P / self.P.sum(sum_dims)).sum(sum_dims)

        return self.plot(WS_eff, clabel='wind speed [m/s]',
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
                                 yaw=self.simulationResult.yaw.ilk(), tilt=self.simulationResult.tilt.ilk())
        hcw = self.windFarmModel.deflectionModel.calc_deflection(**kwargs)[1]
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
            x coordinate for the yz-grid\n
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
            ext *= max(1000, (np.max(x) - np.min(x)))
            return np.linspace(np.min(x) - ext, np.max(x) + ext, N)
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


class XZGrid(YZGrid):
    def __init__(self, y, x=None, z=None, resolution=None, extend=.2):
        """Generate a vertical grid for a flow map in the xz-plane

        Parameters
        ----------
        y : array_like, optional
            y coordinate used for generating meshgrid
        x : array_like, optional
            x coordinatex for the yz-grid\n
        z : array_like, optional
            z coordinates(height above ground) used for generating meshgrid
        resolution : int or None, optional
            grid resolution if x or y is not specified. defaults to self.default_resolution
        extend : float, optional
            defines the oversize of the grid if x or y is not specified

        Notes
        -----
        if x or z is not specified then a grid with <resolution> number of points
        covering the wind turbines + <extend> * range
        """
        YZGrid.__init__(self, x, y=y, z=z, resolution=resolution, extend=extend)
        self.plane = "XZ", y

    def __call__(self, x_i, y_i, h_i, d_i):
        # setup horizontal X,Y grid
        def f(x, N=self.resolution, ext=self.extend):
            ext *= max(1000, (np.max(x) - np.min(x)))
            return np.linspace(np.min(x) - ext, np.max(x) + ext, N)
        x, y, z = self.x, self.y, self.z
        if x is None:
            x = f(x_i)
        if self.z is None:
            z = np.arange(0, (1 + self.extend) * (h_i.max() + d_i.max() / 2), np.diff(x[:2])[0])
        else:
            z = self.z

        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X) + y
        return X, Z, X.T.flatten(), Y.T.flatten(), Z.T.flatten()


class Points(Grid):
    def __init__(self, x, y, h):
        assert len(x) == len(y) == len(h)
        self.x = x
        self.y = y
        self.h = h
        self.plane = 'xyz', None

    def __call__(self, **_):
        return None, None, self.x, self.y, self.h
