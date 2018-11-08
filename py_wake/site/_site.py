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
        wd_binned = np.linspace(0, 360, 360, endpoint=False)
    para = np.array(para)
    num_sector = para.shape[axis]
    wd_sector = np.linspace(0, 360, num_sector, endpoint=False)

    try:
        interp_index = ['piecewise', 'linear', 'spline'].index(interp_method)
        interp_kind = ['nearest', 'linear', 'cubic'][interp_index]
    except ValueError:
        raise NotImplementedError(
            'interp_method={0} not implemeted yet.'.format(interp_method))
    wd_sector_extended = np.hstack((wd_sector, 360.0))
    para_sector_extended = np.concatenate((para, para.take([0], axis=axis)),
                                          axis=axis)
    f_interp = interpolate.interp1d(wd_sector_extended, para_sector_extended,
                                    kind=interp_kind, axis=axis)
    para_expanded = f_interp(wd_binned % 360)

    return para_expanded


class Site(ABC):
    def __init__(self):
        self.default_ws = np.arange(3, 26)
        self.default_wd = np.arange(360)

    @abstractmethod
    def local_wind(self, x_i, y_i, h_i=None, wd=None, ws=None, wd_bin_size=None, ws_bin_size=None):
        """Local free flow wind conditions

        Parameters
        ----------
        x_i : array_like
            Local x coordinate
        y_i : array_like
            Local y coordinate
        h_i : array_like, optional
            Local h coordinate, i.e., heights above ground
        default_wd : float, int or array_like, optional
            Global wind direction(s). Override self.default_wd
        default_ws : float, int or array_like, optional
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
        pass

    @abstractmethod
    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        """calculate down/crosswind distance between source and destination points

        Parameters
        ----------
        src_h_i : array_like
            height above ground level
        Returns
        -------
        dw_ijl : array_like
            down wind distances
            negative is upstream
        cw_ijl : array_like
            cross wind distances
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


class UniformSite(Site):
    def __init__(self, p_wd, ti, interp_method='piecewise', alpha=0.143, h_ref=None):
        self.ti = ti
        self.alpha = alpha
        self.h_ref = h_ref
        super().__init__()
        p_wd = Sector2Subsector(p_wd, interp_method=interp_method)
        self.p_wd = p_wd / p_wd.sum()

    def probability(self, wd, ws, wd_bin_size, ws_bin_size):
        P = self.p_wd[np.round(wd).astype(np.int) % 360] * wd_bin_size
        return P / P.sum()

    def local_wind(self, x_i, y_i, h_i=None, wd=None, ws=None, h_ref=None, wd_bin_size=None, ws_bin_size=None):
        if wd is None:
            wd = self.default_wd
        if ws is None:
            ws = self.default_ws

        def get_default(w_bin_size, w):
            if w_bin_size is None:
                if hasattr(w, '__len__') and len(w) > 1:
                    return w[1] - w[0]
                else:
                    return 1
            else:
                return w_bin_size

        ws_bin_size = get_default(ws_bin_size, ws)
        wd_bin_size = get_default(wd_bin_size, wd)
        WD_ilk, WS_ilk = [np.tile(W, (len(x_i), 1, 1)).astype(np.float)
                          for W in np.meshgrid(wd, ws, indexing='ij')]
        # accouting wind shear when required
        h_ref = h_ref or self.h_ref
        if h_i is not None and h_ref is not None:
            h_i = np.array(h_i)
            if not np.all(h_i == h_ref):
                wind_shear_ratio = (h_i / h_ref) ** self.alpha
                WS_ilk = WS_ilk * wind_shear_ratio[:, na, na]

        TI_ilk = np.zeros_like(WD_ilk) + self.ti
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
        cw_ijl = np.abs(sin_l[na, na, :] * dx_ij[:, :, na] +
                        -cos_l[na, na, :] * dy_ij[:, :, na])
        dh_ijl = np.zeros_like(dw_ijl)
        dh_ijl[:, :, :] = dh_ij[:, :, na]

        dw_order_indices_l = np.argsort(-dw_il, 0).astype(np.int).T

        return dw_ijl, cw_ijl, dh_ijl, dw_order_indices_l

    def elevation(self, x_i, y_i):
        return np.zeros_like(x_i)


class UniformWeibullSite(UniformSite):
    def __init__(self, p_wd, a, k, ti, interp_method='piecewise', alpha=0.143, h_ref=None):
        super().__init__(p_wd, ti, interp_method=interp_method, alpha=alpha, h_ref=h_ref)
        self.a = Sector2Subsector(a, interp_method=interp_method)
        self.k = Sector2Subsector(k, interp_method=interp_method)

    def weibull_weight(self, WS, A, k, wsp_bin_size):
        def cdf(ws, A=A, k=k):
            return 1 - np.exp(-(ws / A) ** k)
        dWS = wsp_bin_size / 2
        return cdf(WS + dWS) - cdf(WS - dWS)

    def probability(self, wd, ws, wd_bin_size, ws_bin_size):
        wd = np.round(wd).astype(np.int)
        return self.weibull_weight(ws, self.a[wd], self.k[wd], ws_bin_size) * self.p_wd[wd] * wd_bin_size


def main():
    if __name__ == '__main__':
        f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
             0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
        A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
             9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
        k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
             2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
        ti = .1
        site = UniformWeibullSite(f, A, k, ti)

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
        plt.show()
        h_i = np.array([100, 110, 120, 130, 140])
        h_ref = 100
        WD_ilk1, WS_ilk1, TI_ilk1, P_lk1 = site.local_wind(
            x_i=x_i, y_i=y_i, h_i=h_i, h_ref=h_ref, wd=wdir_lst, ws=wsp_lst)


main()
