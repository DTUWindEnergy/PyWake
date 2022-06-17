import matplotlib.pyplot as plt
from py_wake import np
from py_wake.site.shear import PowerShear
import py_wake.utils.xarray_utils  # register ilk function @UnusedImport
import xarray as xr
from abc import ABC, abstractmethod
from py_wake.utils.xarray_utils import ilk2da
from numpy import newaxis as na
from py_wake.utils.functions import arg2ilk

"""
suffixs:
- i: Local point (wind turbines)
- j: Local point (downstream turbines or positions)
- l: Wind directions
- k: Wind speeds
"""


class LocalWind(dict):
    def __init__(self, x_i, y_i, h_i, wd, ws, time, wd_bin_size, WD=None, WS=None, TI=None, P=None):
        """

        Parameters
        ----------
        WD : array_like
            local free flow wind directions
        WS : array_like
            local free flow wind speeds
        TI : array_like
            local free flow turbulence intensity
        P : array_like
            Probability/weight
        """
        ws = np.atleast_1d(ws)
        if time is not False:
            assert len(wd) == len(ws)
            if time is True:
                time = np.arange(len(wd))
            coords = {'time': np.atleast_1d(time), 'wd': np.atleast_1d(wd), 'ws': np.atleast_1d(ws)}
        else:
            coords = {'wd': np.atleast_1d(wd), 'ws': np.atleast_1d(ws)}

        assert len(np.atleast_1d(x_i)) == len(np.atleast_1d(y_i))
        n_i = max(len(np.atleast_1d(x_i)), len(np.atleast_1d(h_i)))
        coords['i'] = np.arange(n_i)

        for k, v in [('x', x_i), ('y', y_i), ('h', h_i)]:
            if v is not None:
                coords[k] = np.zeros(n_i) + v

        self.coords = coords
        for k, v in [('WD', WD), ('WS', WS), ('TI', TI), ('P', P)]:
            if v is not None:
                self[k] = v
        self.descriptions = {}

        self['wd_bin_size'] = wd_bin_size

    def __getattribute__(self, name):
        try:
            return dict.__getattribute__(self, name)
        except AttributeError:
            if name != 'coords':  # may not exists when loading from pkl
                keys = self.keys()
                if name in keys:
                    return self[name]
                coords = getattr(self, 'coords', {})
                if name in coords.keys():
                    return coords[name]
                elif name + "_ilk" in keys:
                    return ilk2da(self[name + '_ilk'], coords, self.descriptions.get(name + '_ilk', None))
            raise

    def __contains__(self, key):
        return key in self.keys() or key in self.coords.keys()

    def set_data_array(self, data_array, name, description):
        if data_array is not None:
            self[name] = data_array
            self.descriptions[name] = description

    def add_ilk(self, name, value):
        coords = self.coords
        self[name] = arg2ilk(name, value, I=len(coords['i']), L=len(coords['wd']), K=len(coords['ws']))

    def set_W(self, ws, wd, ti, ws_bins, use_WS=False):
        for da, name, desc in [(ws, 'WS_ilk', 'Local free-stream wind speed [m/s]'),
                               (wd, 'WD_ilk', 'Local free-stream wind direction [deg]'),
                               (ti, 'TI_ilk', 'Local free-stream turbulence intensity')]:
            self.set_data_array(np.atleast_3d(da), name, desc)

        # upper and lower bounds of wind speed bins
        WS_ilk = [self.ws[na, na], self.WS_ilk][use_WS]
        if not hasattr(ws_bins, '__len__') or len(ws_bins) != WS_ilk.shape[2] + 1:
            if WS_ilk.shape[-1] > 1:
                d = np.diff(WS_ilk) / 2
                ws_bins = np.maximum(np.concatenate(
                    [WS_ilk[..., :1] - d[..., :1], WS_ilk[..., :-1] + d, WS_ilk[..., -1:] + d[..., -1:]], -1), 0)
            else:
                # WS is single value
                if ws_bins is None:
                    ws_bins = 1
                ws_bins = WS_ilk + np.array([-ws_bins / 2, ws_bins / 2])
        else:
            ws_bins = np.asarray(ws_bins)

        self.set_data_array(ws_bins[..., :-1], 'ws_lower', 'Lower bound of wind speed bins [m/s]')
        self.set_data_array(ws_bins[..., 1:], 'ws_upper', 'Upper bound of wind speed bins [m/s]')


class Site(ABC):
    def __init__(self, distance):
        self.distance = distance
        self.default_ws = np.arange(3, 26.)
        self.default_wd = np.arange(360)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance
        distance.site = self

    def get_defaults(self, wd=None, ws=None):
        if wd is None:
            wd = self.default_wd
        else:
            wd = np.atleast_1d(wd)
        if ws is None:
            ws = self.default_ws
        else:
            ws = np.atleast_1d(ws)
        return wd, ws

    def local_wind(self, x_i, y_i, h_i=None, wd=None, ws=None, time=False, wd_bin_size=None, ws_bins=None):
        """Local free flow wind conditions

        Parameters
        ----------
        x_i  :  array_like
            Local x coordinate
        y_i : array_like
            Local y coordinate
        h_i : array_like, optional
            Local h coordinate, i.e., heights above ground
        wd : float, int or array_like, optional
            Global wind direction(s). Override self.default_wd
        ws : float, int or array_like, optional
            Global wind speed(s). Override self.default_ws
        time : boolean or array_like
            If True or array_like, wd and ws is interpreted as a time series
            If False, full wd x ws matrix is computed
        wd_bin_size : int or float, optional
            Size of wind direction bins. default is size between first and
            second element in default_wd
        ws_bin : array_like or None, optional
            Wind speed bin edges

        Returns
        -------
        LocalWind object containing:
            WD_ilk : array_like
                local free flow wind directions
            WS_ilk : array_like
                local free flow wind speeds
            TI_ilk : array_like
                local free flow turbulence intensity
            P_ilk : array_like
                Probability/weight
        """
        wd, ws = self.get_defaults(wd, ws)
        wd_bin_size = self.wd_bin_size(wd, wd_bin_size)
        lw = LocalWind(x_i, y_i, h_i, wd, ws, time, wd_bin_size)
        return self._local_wind(lw, ws_bins)

    @abstractmethod
    def _local_wind(self, localWind, ws_bins=None):
        """Local free flow wind conditions

        Parameters
        ----------
        localWind  : LocalWind
            xarray dataset containing coordinates x, y, h, wd, ws
        ws_bin : array_like or None, optional
            Wind speed bin edges

        Returns
        -------
        LocalWind xarray dataset containing:
            WD : DataArray
                local free flow wind directions
            WS : DataArray
                local free flow wind speeds
            TI : DataArray
                local free flow turbulence intensity
            P : DataArray
                Probability/weight
        """

    def wt2wt_distances(self, WD_il):
        return self.distance(WD_il)

    @abstractmethod
    def elevation(self, x_i, y_i):
        """Local terrain elevation (height above mean sea level)

        Parameters
        ----------
        x_i : array_like
            Local x coordinate
        y_i : array_like
            Local y coordinate

        Returns
        -------
        elevation : array_like
        """

    def wd_bin_size(self, wd, wd_bin_size=None):
        wd = np.atleast_1d(wd)
        if wd_bin_size is not None:
            return wd_bin_size
        elif len(wd) > 1 and len(np.unique(np.diff(wd))) == 1:
            return wd[1] - wd[0]
        else:
            return 360 / len(np.atleast_1d(wd))

    def ws_bins(self, WS, ws_bins=None):
        # TODO: delete function
        if not isinstance(WS, xr.DataArray):
            WS = xr.DataArray(WS, [('ws', np.atleast_1d(WS))])
        if not hasattr(ws_bins, '__len__') or len(ws_bins) != len(WS) + 1:

            if len(WS.shape) and WS.shape[-1] > 1:
                d = np.diff(WS) / 2
                ws_bins = np.maximum(np.concatenate(
                    [WS[..., :1] - d[..., :1], WS[..., :-1] + d, WS[..., -1:] + d[..., -1:]], -1), 0)
            else:
                # WS is single value
                if ws_bins is None:
                    ws_bins = 1
                ws_bins = WS.data + np.array([-ws_bins / 2, ws_bins / 2])
        else:
            ws_bins = np.asarray(ws_bins)
        return xr.Dataset({'ws_lower': (WS.dims, ws_bins[..., :-1]),
                           'ws_upper': (WS.dims, ws_bins[..., 1:])},
                          coords=WS.coords)

    def _sector(self, wd):
        sector = np.zeros(360, dtype=int)

        d_wd = (np.diff(np.r_[wd, wd[0]]) % 360) / 2
        assert np.all(d_wd == d_wd[0]), "Wind directions must be equidistant"
        lower = np.ceil(wd - d_wd).astype(int)
        upper = np.ceil(wd + d_wd).astype(int)
        for i, (lo, up) in enumerate(zip(lower, upper)):
            if lo < 0:
                sector[lo % 360 + 1:] = i
                lo = 0
            if up > 359:
                sector[:up % 360 + 1] = i
                up = 359
            sector[lo + 1:up + 1] = i
        return sector

    def plot_ws_distribution(self, x=0, y=0, h=70, wd=[0], ws=np.arange(0.05, 30.05, .1),
                             include_wd_distribution=False, ax=None):
        """Plot wind speed distribution

        Parameters
        ----------
        x : int or float
            Local x coordinate
        y : int or float
            Local y coordinate
        h : int or float
            Local height above ground
        wd : int or array_like
            Wind direction(s) (one curve pr wind direction)
        ws : array_like, optional
            Wind speeds to calculate for
        include_wd_distributeion : bool, default is False
            If true, the wind speed probability distributions are multiplied by
            the wind direction probability. The sector size is set to 360 / len(wd).
            This only makes sense if the wd array is evenly distributed
        ax : pyplot or matplotlib axes object, default None

        """
        if ax is None:
            ax = plt

        lbl = "Wind direction: %d deg"
        if include_wd_distribution:

            P = self.local_wind(x_i=x, y_i=y, h_i=h, wd=np.arange(360), ws=ws, wd_bin_size=1).P
            P.coords['sector'] = ('wd', self._sector(wd))
            P = P.groupby('sector').sum()
            v = 360 / len(wd) / 2
            lbl += r"$\pm$%s deg" % ((int(v), v)[(v % 2) != 0])
        else:
            lw = self.local_wind(x_i=x, y_i=y, h_i=h, wd=wd, ws=ws, wd_bin_size=1)
            P = lw.P
            if 'ws' not in P.dims:
                P = P.broadcast_like(lw.WS).T
            P = P / P.sum('ws')  # exclude wd probability
        if 'wd' not in P.dims and 'sector' not in P.dims:
            P = P.expand_dims({'wd': wd})
        for wd, p in zip(wd, P):
            ax.plot(ws, p * 10, label=lbl % wd)
            ax.xlabel('Wind speed [m/s]')
            ax.ylabel('Probability')
        ax.legend(loc=1)
        return P

    def plot_wd_distribution(self, x=0, y=0, h=70, n_wd=12, ws_bins=None, ax=None):
        """Plot wind direction (and speed) distribution

        Parameters
        ----------
        x : int or float
            Local x coordinate
        y : int or float
            Local y coordinate
        h : int or float
            Local height above ground
        n_wd : int
            Number of wind direction sectors
        ws_bins : None, int or array_like, default is None
            Splits the wind direction sector pies into different colors to show
            the probability of different wind speeds\n
            If int, number of wind speed bins in the range 0-30\n
            If array_like, limits of the wind speed bins limited by ws_bins,
            e.g. [0,10,20], will show 0-10 m/wd_bin_size and 10-20 m/wd_bin_size
        ax : pyplot or matplotlib axes object, default None
        """
        if ax is None:
            ax = plt
        assert 360 % n_wd == 0

        wd_bin_size = 360 // n_wd
        wd = np.arange(0, 360, wd_bin_size)
        theta = wd / 180 * np.pi
        if not ax.__class__.__name__ == 'PolarAxesSubplot':
            if hasattr(ax, 'subplot'):
                ax.clf()
                ax = ax.subplot(111, projection='polar')
            else:
                ax.figure.clf()
                ax = ax.figure.add_subplot(111, projection='polar')
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)

        if ws_bins is None:
            if any(['ws' in v.dims for v in self.ds.data_vars.values()]):
                lw = self.local_wind(x_i=x, y_i=y, h_i=h, wd=np.arange(360), wd_bin_size=1)
                P = lw.P.sum('ws')
            else:
                lw = self.local_wind(x_i=x, y_i=y, h_i=h, wd=np.arange(360),
                                     ws=[100], ws_bins=[0, 200], wd_bin_size=1)
                P = lw.P
        else:
            if not hasattr(ws_bins, '__len__'):
                ws_bins = np.linspace(0, 30, ws_bins)
            else:
                ws_bins = np.asarray(ws_bins)
            ws = ((ws_bins[1:] + ws_bins[:-1]) / 2)
            lw = self.local_wind(x_i=x, y_i=y, h_i=h, wd=np.arange(360), ws=ws, wd_bin_size=1)
            P = lw.P

        P.coords['sector'] = ('wd', self._sector(wd))
        P = P.groupby('sector').sum()

        if ws_bins is None or 'ws' not in P.dims:
            ax.bar(theta, P.values, width=np.deg2rad(wd_bin_size), bottom=0.0)
        else:
            P = P.T
            start_P = np.vstack([np.zeros_like(P[:1]), P.cumsum('ws')[:-1]])
            for ws1, ws2, p_ws0, p_ws in zip(lw.ws_lower, lw.ws_upper, start_P, P):
                ax.bar(theta, p_ws, width=np.deg2rad(wd_bin_size), bottom=p_ws0,
                       label="%s-%s m/s" % (ws1, ws2))
            ax.legend(bbox_to_anchor=(1.15, 1.1))

        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        return P.T


from py_wake.site import xrsite  # @NoMove # nopep8
UniformSite = xrsite.UniformSite
UniformWeibullSite = xrsite.UniformWeibullSite


def get_sector_xr(v, name):
    if isinstance(v, (int, float)):
        return xr.DataArray(v, coords=[], name=name)
    v = np.r_[v, np.atleast_1d(v)[0]]
    return xr.DataArray(v, coords=[('wd', np.linspace(0, 360, len(v)))], name=name)


def main():
    if __name__ == '__main__':
        f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
             0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
        A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
             9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
        k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
             2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
        ti = .1
        h_ref = 100
        alpha = .1
        site = UniformWeibullSite(f, A, k, ti, shear=PowerShear(h_ref=h_ref, alpha=alpha))

        x_i = y_i = np.arange(5)
        wdir_lst = np.arange(0, 360, 90)
        wsp_lst = np.arange(1, 20)
        local_wind = site.local_wind(x_i=x_i, y_i=y_i, h_i=h_ref, wd=wdir_lst, ws=wsp_lst)
        print(local_wind.WS_ilk.shape)

        site.plot_ws_distribution(0, 0, wdir_lst)

        plt.figure()
        z = np.arange(1, 100)
        u = [site.local_wind(x_i=[0], y_i=[0], h_i=[z_], wd=0, ws=10).WS_ilk[0][0] for z_ in z]
        plt.plot(u, z)
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Height [m]')
        plt.show()


main()
