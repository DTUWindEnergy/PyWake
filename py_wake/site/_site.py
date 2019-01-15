import numpy as np
from abc import ABC, abstractmethod

"""
suffixs:
- i: Local point (wind turbines)
- j: Local point (downstream turbines or positions)
- l: Wind directions
- k: Wind speeds
- m: Height above ground
"""
from numpy import newaxis as na
from scipy import interpolate


class Site(ABC):
    def __init__(self):
        self.default_ws = np.arange(3, 26)
        self.default_wd = np.arange(360)

    @abstractmethod
    def local_wind(self, x_i, y_i, h_i=None, wd=None, ws=None, wd_bin_size=None, ws_bin_size=None):
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
        wd_bin_size : int or float, optional
            Size of wind direction bins. default is size between first and
            second element in default_wd
        ws_bin_size : int or float, optional
            Size of wind speed bins. default is size between first and
            second element in default_ws

        Returns
        -------
        WD_ilk : array_like
            local free flow wind directions
        WS_ilk : array_like
            local free flow wind speeds
        TI_ilk : array_like
            local free flow turbulence intensity
        P_lk : array_like
            Probability/weight
        """
        pass

    @abstractmethod
    def probability(self, wd, ws, wd_bin_size, ws_bin_size):
        """Probability of wind situation (wind speed and direction)

        Parameters
        ----------
        wd : int, float or array_like
            Wind direction
        ws : int, float or array_like
            Wind speed
        wd_bin_size : int or float
            size of wind direction sectors
        ws_bin_size : int or float
            size of wind speed bins

        Returns
        -------
        P : float or array_like
            Probability of wind speed and direction
        """
        pass

    @abstractmethod
    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        """Calculate down/crosswind distance between source and destination points

        Parameters
        ----------
        src_x_i : array_like
            Source x position
        src_y_i : array_like
            Source y position
        src_h_i : array_like
            Source height above ground level
        dst_x_j : array_like
            Destination x position
        dst_y_j : array_like
            Destination y position
        dst_h_j : array_like
            Destination height above ground level
        wd_il : array_like, shape (#src, #wd)
            Local wind direction at the source points for all global wind directions


        Returns
        -------
        dw_ijl : array_like
            down wind distances
            negative is upstream
        hcw_ijl : array_like
            horizontal cross wind distances
        dh_ijl : array_like
            vertical distances
        dw_order_indices_l : array_like
            indices that gives the downwind order of source points
        """
        pass

    def wt2wt_distances(self, x_i, y_i, h_i, wd_il):
        return self.distances(x_i, y_i, h_i, x_i, y_i, h_i, wd_il)

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
        pass

    def wd_bin_size(self, wd, wd_bin_size=None):
        if wd_bin_size is not None:
            return wd_bin_size
        else:
            return 360 / len(np.atleast_1d(wd))

    def ws_bin_size(self, ws, ws_bin_size=None):
        if ws_bin_size is None:
            if hasattr(ws, '__len__') and len(ws) > 1:
                return ws[1] - ws[0]
            else:
                return 1
        else:
            return ws_bin_size

    def plot_ws_distribution(self, wd, include_wd_distribution=False, ax=None):
        """Plot wind speed distribution

        Parameters
        ----------
        wd : int or array_like
            Wind direction(s) (one curve pr wind direction)
        include_wwd_distributeion : bool, default is False
            If true, the wind speed probability distributions are multiplied by
            the wind direction probability. The sector size is set to 360 / len(wd).
            This only makes sense if the wd array is evenly distributed
        ax : pyplot or matplotlib axes object, default None

        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt
        ws = np.arange(0.05, 30.05, .1)
        wd = np.atleast_1d(wd)
        for wd_ in wd:
            wd_bin_size = 360 / len(wd)

            if include_wd_distribution:
                v = wd_bin_size / 2
                WD_lk = np.arange(wd_ - v, wd_ + v)[:, na]
                WS_lk = ws
                p = self.probability(WD_lk, WS_lk,
                                     wd_bin_size=1, ws_bin_size=.1).sum(0)
                lbl = r"Wind direction: %d$\pm$%s deg" % (wd_, (int(v), v)[(wd_bin_size % 2) != 0])
            else:
                p = self.probability(wd_, ws, wd_bin_size=wd_bin_size, ws_bin_size=.1)
                p /= p.sum()
                lbl = "Wind direction: %d deg" % (wd_)

            ax.plot(ws, p * 10, label=lbl)
            ax.xlabel('Wind speed [m/s]')
            ax.ylabel('Probability')
        ax.legend(loc=1)

    def plot_wd_distribution(self, n_wd, ws_bins=None, ax=None):
        """Plot wind direction (and speed) distribution

        Parameters
        ----------
        n_wd : int
            Number of wind direction sectors
        ws_bins : None, int or array_like, default is None
            Splits the wind direction sector pies into different colors to show
            the probability of different wind speeds\n
            If int, number of wind speed bins in the range 0-30\n
            If array_like, limits of the wind speed bins limited by ws_bins,
            e.g. [0,10,20], will show 0-10 m/s and 10-20 m/s
        ax : pyplot or matplotlib axes object, default None
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt
        wd = np.linspace(0, 360, n_wd, endpoint=False)
        theta = wd / 180 * np.pi
        if not ax.__class__.__name__ == 'PolarAxesSubplot':
            if hasattr(ax, 'subplot'):
                ax = ax.subplot(111, projection='polar')
            else:
                ax = ax.figure.add_subplot(111, projection='polar')
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)

        s = 360 / n_wd
        if ws_bins is None:
            p = [self.probability(np.arange(wd_ - s / 2, wd_ + s / 2), ws=100,
                                  wd_bin_size=1, ws_bin_size=200).sum(0) for wd_ in wd]
            ax.bar(theta, p, width=s / 180 * np.pi, bottom=0.0)
        else:
            if not hasattr(ws_bins, '__len__'):
                ws_bins = np.linspace(0, 30, ws_bins)
            else:
                ws_bins = np.asarray(ws_bins)
            ws = ((ws_bins[1:] + ws_bins[:-1]) / 2)
            ws_bin_size = ws[1] - ws[0]

            p = [self.probability(np.arange(wd_ - s / 2, wd_ + s / 2)[:, na], ws=ws[na],
                                  wd_bin_size=1, ws_bin_size=ws_bin_size).sum(0) for wd_ in wd]
            cum_p = np.cumsum(p, 1).T
            start_p = np.vstack([np.zeros_like(cum_p[:1]), cum_p[:-1]])

            for ws1, ws2, p_ws1, p_ws2 in zip(ws_bins[:-1], ws_bins[1:], start_p, cum_p):
                ax.bar(theta, p_ws2 - p_ws1, width=s / 180 * np.pi, bottom=p_ws1, label="%s-%s m/s" % (ws1, ws2))
            ax.legend(bbox_to_anchor=(1.15, 1.1))

        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)


class UniformSite(Site):
    """Site with uniform (same wind over all, i.e. flat unuform terrain) and
    constant wind speed probability of 1. Only for one fixed wind speed
    """

    def __init__(self, p_wd, ti, interp_method='piecewise', alpha=None, h_ref=None):
        self.ti = Sector2Subsector(np.atleast_1d(ti), interp_method=interp_method)
        self.alpha = Sector2Subsector(np.atleast_1d(alpha), interp_method=interp_method)
        self.h_ref = h_ref
        super().__init__()
        self.p_wd = Sector2Subsector(p_wd / np.sum(p_wd), interp_method=interp_method) / (360 / len(p_wd))

    def probability(self, wd, ws, wd_bin_size, ws_bin_size):
        P = np.ones_like(ws, dtype=np.float) * self.p_wd[np.round(wd).astype(np.int) % 360] * wd_bin_size
        return P

    def local_wind(self, x_i, y_i, h_i=None, wd=None, ws=None, wd_bin_size=None, ws_bin_size=None):
        if wd is None:
            wd = self.default_wd
        if ws is None:
            ws = self.default_ws

        ws_bin_size = self.ws_bin_size(ws, ws_bin_size)
        wd_bin_size = self.wd_bin_size(wd, wd_bin_size)
        WD_ilk, WS_ilk = [np.tile(W, (len(x_i), 1, 1)).astype(np.float)
                          for W in np.meshgrid(wd, ws, indexing='ij')]
        WD_index_ilk = np.round(WD_ilk).astype(np.int)
        # accouting wind shear when required
        if h_i is not None and self.h_ref is not None and self.alpha is not None:
            h_i = np.array(h_i)
            if not np.all(h_i == self.h_ref):
                wind_shear_ratio = (h_i / self.h_ref)[:, na, na] ** self.alpha[WD_index_ilk]
                WS_ilk = WS_ilk * wind_shear_ratio

        TI_ilk = self.ti[WD_index_ilk]
        P_lk = self.probability(WD_ilk[0], WS_ilk[0], wd_bin_size, ws_bin_size)
        return WD_ilk, WS_ilk, TI_ilk, P_lk

    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        wd_l = np.mean(wd_il, 0)
        dx_ij, dy_ij, dh_ij = [np.subtract(*np.meshgrid(dst_j, src_i, indexing='ij')).T
                               for src_i, dst_j in [(src_x_i, dst_x_j),
                                                    (src_y_i, dst_y_j),
                                                    (src_h_i, dst_h_j)]]
        src_x_i, src_y_i = map(np.asarray, [src_x_i, src_y_i])

        theta_l = np.deg2rad(90 - wd_l)
        cos_l = np.cos(theta_l)
        sin_l = np.sin(theta_l)
        dw_il = (cos_l[na, :] * src_x_i[:, na] + sin_l[na] * src_y_i[:, na])
        dw_ijl = (-cos_l[na, na, :] * dx_ij[:, :, na] - sin_l[na, na, :] * dy_ij[:, :, na])
        hcw_ijl = np.abs(sin_l[na, na, :] * dx_ij[:, :, na] +
                         -cos_l[na, na, :] * dy_ij[:, :, na])
        dh_ijl = np.zeros_like(dw_ijl)
        dh_ijl[:, :, :] = dh_ij[:, :, na]

        dw_order_indices_l = np.argsort(-dw_il, 0).astype(np.int).T

        return dw_ijl, hcw_ijl, dh_ijl, dw_order_indices_l

    def elevation(self, x_i, y_i):
        return np.zeros_like(x_i)


class UniformWeibullSite(UniformSite):
    """Site with uniform (same wind over all, i.e. flat unuform terrain) and
    weibull distributed wind speed
    """

    def __init__(self, p_wd, a, k, ti, interp_method='nearest', alpha=None, h_ref=None):
        """Initialize UniformWeibullSite

        Parameters
        ----------
        p_wd : array_like
            Probability of wind direction sectors
        a : array_like
            Weilbull scaling parameter of wind direction sectors
        k : array_like
            Weibull shape parameter
        ti : float or array_like
            Turbulence intensity
        interp_method : 'nearest', 'linear' or 'spline'
            p_wd, a, k, ti and alpha are interpolated to 1 deg sectors using this
            method
        alpha : float or array_like
            Power shear profile exponent of the wind direction sectors
        h_ref : int or float
            Reference height for the power shear profile


        Notes
        ------
        The wind direction sectors will be: [0 +/- w/2, w +/- w/2, ...]
        where w is 360 / len(p_wd)

        """
        super().__init__(p_wd, ti, interp_method=interp_method, alpha=alpha, h_ref=h_ref)
        self.a = Sector2Subsector(a, interp_method=interp_method)
        self.k = Sector2Subsector(k, interp_method=interp_method)

    def weibull_weight(self, WS, A, k, wsp_bin_size):
        def cdf(ws, A=A, k=k):
            return 1 - np.exp(-(ws / A) ** k)
        dWS = wsp_bin_size / 2
        return cdf(WS + dWS) - cdf(WS - dWS)

    def probability(self, wd, ws, wd_bin_size, ws_bin_size):
        wd = np.round(wd).astype(np.int) % 360
        return self.weibull_weight(ws, self.a[wd], self.k[wd], ws_bin_size) * self.p_wd[wd] * wd_bin_size


def Sector2Subsector(para, axis=-1, wd_binned=None, interp_method='piecewise'):
    """ Expand para on the wind direction dimension, i.e., increase the nubmer
    of sectors (sectors to subsectors), by interpolating between sectors, using
    specified method.

    Parameters
    ----------
    para : array_like
        Parameter to be expand, it can be sector-wise Weibull A, k, frequency.
    axis : integer
        Denotes which dimension of para corresponds to wind direction.
    wd_binned : array_like
        Wind direction of subsectors to be expanded to.
    inter_method : string
        'piecewise'/'linear'/'spline', based on interp1d in scipy.interpolate,
        'spline' means cubic spline.

    --------------------------------------
    Note: the interpolating method for sector-wise Weibull distributions and
    joint distribution of wind speed and wind direction is referred to the
    following paper:
        Feng, J. and Shen, W.Z., 2015. Modelling wind for wind farm layout
        optimization using joint distribution of wind speed and wind direction.
        Energies, 8(4), pp.3075-3092. [https://doi.org/10.3390/en8043075]
    """
    if wd_binned is None:
        wd_binned = np.arange(360)
    para = np.array(para)
    num_sector = para.shape[axis]
    wd_sector = np.linspace(0, 360, num_sector, endpoint=False)

    try:
        interp_index = ['nearest', 'piecewise', 'linear', 'spline'].index(interp_method)
        interp_kind = ['nearest', 'nearest', 'linear', 'cubic'][interp_index]
    except ValueError:
        raise NotImplementedError(
            'interp_method={0} not implemeted yet.'.format(interp_method))
    wd_sector_extended = np.hstack((wd_sector, 360.0))
    para_sector_extended = np.concatenate((para, para.take([0], axis=axis)),
                                          axis=axis)
    if interp_kind == 'cubic' and len(wd_sector_extended) < 4:
        interp_kind = 'linear'
    f_interp = interpolate.interp1d(wd_sector_extended, para_sector_extended,
                                    kind=interp_kind, axis=axis)
    para_expanded = f_interp(wd_binned % 360)

    return para_expanded


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
        site = UniformWeibullSite(f, A, k, ti, alpha=alpha, h_ref=h_ref)

        x_i = y_i = np.arange(5)
        wdir_lst = np.arange(0, 360, 90)
        wsp_lst = np.arange(1, 20)
        WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i=x_i, y_i=y_i, wd=wdir_lst, ws=wsp_lst)
        import matplotlib.pyplot as plt
        for wdir, P_k in zip(wdir_lst, P_lk):
            plt.plot(wsp_lst, P_k, label='%s deg' % wdir)
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Probability')
        plt.legend()

        plt.figure()
        z = np.arange(1, 100)
        u = [site.local_wind(x_i=[0], y_i=[0], h_i=[z_], wd=0, ws=10)[1][0][0] for z_ in z]
        plt.plot(u, z)
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Height [m]')
        plt.show()


main()
