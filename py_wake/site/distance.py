from py_wake import np
from numpy import newaxis as na
import matplotlib
from py_wake.utils.functions import mean_deg
from py_wake.utils import gradients
from py_wake.utils.gradients import rad2deg, deg2rad


class StraightDistance():

    def __init__(self, wind_direction='wd'):
        """
        Parameters
        ----------
        wind_direction : {'wd','WD_i'}
            'wd': The reference wind direction, wd, is used to calculate downwind and horizontal crooswind distances
            'WD_i': The wind direction at the current (upstream) wind turbine is used to calculate downwind and
            horizontal crossswind distances.

        """
        self.wind_direction = wind_direction

    def _cos_sin(self, wd):
        theta = gradients.deg2rad(90 - wd)
        cos = np.cos(theta)
        sin = np.sin(theta)
        return cos, sin

    def plot(self, WD_ilk=None, wd_l=None, src_idx=slice(None), dst_idx=slice(None)):
        import matplotlib.pyplot as plt

        dw_ijlk, hcw_ijlk, _ = self(WD_ilk=WD_ilk, wd_l=wd_l)
        if self.wind_direction == 'wd':
            WD_ilk = np.asarray(wd_l)[na, :, na]

        wdirs = mean_deg(WD_ilk, (0, 2))
        for l, wd in enumerate(wdirs):
            plt.figure()
            ax = plt.gca()
            theta = np.deg2rad(90 - wd)
            ax.set_title(wd)
            ax.arrow(0, 0, -np.cos(theta) * 20, -np.sin(theta) * 20, width=1)
            colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])]
            f = 2
            for i, x_, y_ in zip(np.arange(len(self.src_x_ilk))[
                                 src_idx], self.src_x_ilk[src_idx, 0, 0], self.src_y_ilk[src_idx, 0, 0]):
                c = colors[i % len(colors)]
                ax.plot(x_, y_, '2', color=c, ms=10, mew=3)
                for j, dst_x, dst_y in zip(np.arange(len(self.dst_x_j))[dst_idx],
                                           self.dst_x_j[dst_idx, 0, 0], self.dst_y_j[dst_idx, 0, 0]):
                    ax.arrow(x_ - j / f, y_ - j / f, -np.cos(theta) * dw_ijlk[i, j, l, 0], -
                             np.sin(theta) * dw_ijlk[i, j, l, 0], width=.3, color=c)
                    ax.plot([dst_x - i / f, dst_x - np.sin(theta) * hcw_ijlk[i, j, l, 0] - i / f],
                            [dst_y - i / f, dst_y + np.cos(theta) * hcw_ijlk[i, j, l, 0] - i / f], '--', color=c)
            plt.plot(self.src_x_ilk[:, 0, 0], self.src_y_ilk[:, 0, 0], 'k2')
            ax.axis('equal')
            ax.legend()

    def setup(self, src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk, dst_xyhz_j=None):
        # ensure 3d and
        # +.0 ensures float or complex
        src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk = [np.expand_dims(v, tuple(range(len(np.shape(v)), 3))) + .0
                                                      for v in [src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk]]
        self.src_x_ilk, self.src_y_ilk, self.src_h_ilk = src_x_ilk, src_y_ilk, src_h_ilk

        self.dx_iilk = src_x_ilk - src_x_ilk[:, na]
        self.dy_iilk = src_y_ilk - src_y_ilk[:, na]
        self.dh_iilk = src_h_ilk - src_h_ilk[:, na]
        self.dz_iilk = src_z_ilk - src_z_ilk[:, na]
        if dst_xyhz_j is None:
            dst_x_j, dst_y_j, dst_h_j, dst_z_j = src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk
            self.dx_ijlk, self.dy_ijlk, self.dh_ijlk, self.dz_ijlk = self.dx_iilk, self.dy_iilk, self.dh_iilk, self.dz_iilk
            self.src_eq_dst = True
        else:
            dst_x_j, dst_y_j, dst_h_j, dst_z_j = map(np.asarray, dst_xyhz_j)
            self.dx_ijlk = dst_x_j[na, :, na, na] - src_x_ilk[:, na]
            self.dy_ijlk = dst_y_j[na, :, na, na] - src_y_ilk[:, na]
            self.dh_ijlk = dst_h_j[na, :, na, na] - src_h_ilk[:, na]
            self.dz_ijlk = dst_z_j[na, :, na, na] - src_z_ilk[:, na]
            self.src_eq_dst = False
        self.dst_x_j, self.dst_y_j, self.dst_h_j, self.dst_z_j = dst_x_j, dst_y_j, dst_h_j, dst_z_j

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in {'src_x_ilk', 'src_y_ilk', 'src_h_ilk', 'dst_x_j', 'dst_y_j', 'dst_h_j',
                             'dx_iilk', 'dy_iilk', 'dh_iilk', 'dx_ijlk', 'dy_ijlk', 'dh_ij', 'src_eq_dst'}}

    def __call__(self, WD_ilk=None, wd_l=None, src_idx=slice(None), dst_idx=slice(None)):
        assert hasattr(self, 'dx_ijlk'), "wind_direction setup must be called first"
        assert self.wind_direction in ['wd', 'WD_i'], "'StraightDistance.wind_direction must be 'wd' or 'WD_i'"

        if self.wind_direction == 'wd':
            assert wd_l is not None, "wd_l must be specified when Distance.wind_direction='wd'"
            WD_ilk = np.asarray(wd_l)[na, :, na]
        else:
            assert WD_ilk is not None, "WD_ilk must be specified when Distance.wind_direction='WD_i'"

        if len(np.shape(dst_idx)) == 2:
            # dst_idx depends on wind direction
            cos_jlk, sin_jlk = self._cos_sin(WD_ilk[0, na])
            i_wd_l = np.arange(np.shape(dst_idx)[1])
            dx_jlk = self.dx_iilk[src_idx, dst_idx, np.minimum(i_wd_l, self.dx_iilk.shape[2] - 1).astype(int)]
            dy_jlk = self.dy_iilk[src_idx, dst_idx, np.minimum(i_wd_l, self.dy_iilk.shape[2] - 1).astype(int)]
            dh_jlk = self.dh_iilk[src_idx, dst_idx, np.minimum(i_wd_l, self.dh_iilk.shape[2] - 1).astype(int)]
            dw_jlk = (-cos_jlk * dx_jlk - sin_jlk * dy_jlk)
            hcw_jlk = (sin_jlk * dx_jlk - cos_jlk * dy_jlk)
            return dw_jlk[na], hcw_jlk[na], dh_jlk[na]
        else:
            # dst_idx independent of wind direction
            cos_ijlk, sin_ijlk = self._cos_sin(WD_ilk[:, na])
            dx_ijlk = self.dx_ijlk[src_idx][:, dst_idx]
            dy_ijlk = self.dy_ijlk[src_idx][:, dst_idx]

            dw_ijlk = -cos_ijlk * dx_ijlk - sin_ijlk * dy_ijlk
            hcw_ijlk = sin_ijlk * dx_ijlk - cos_ijlk * dy_ijlk
            # +0 ~ autograd safe copy (broadcast_to returns readonly array)
            dh_ijlk = np.broadcast_to(self.dh_ijlk[src_idx][:, dst_idx], dw_ijlk.shape) + 0
            return dw_ijlk, hcw_ijlk, dh_ijlk

    def dw_order_indices(self, wd_l):
        assert hasattr(self, 'dx_ijlk'), "method setup must be called first"
        I, J, *_ = self.dx_ijlk.shape
        assert I == J
        cos_l, sin_l = self._cos_sin(np.asarray(wd_l))
        # return np.argsort(-cos_l[:, na] * np.asarray(src_x_ilk)[na] - sin_l[:, na] * np.asarray(src_y_ilk)[na], 1)
        dw_iil = -cos_l[na, na, :, na] * self.dx_ijlk - sin_l[na, na, :, na] * self.dy_ijlk
        dw_order_indices_lkd = np.moveaxis(np.argsort((dw_iil > 0).sum(0), 0), 0, -1)
        return dw_order_indices_lkd


class TerrainFollowingDistance(StraightDistance):
    def __init__(self, distance_resolution=1000, wind_direction='wd', **kwargs):
        super().__init__(wind_direction=wind_direction, **kwargs)
        self.distance_resolution = distance_resolution

    def setup(self, src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk, dst_xyhz_j=None):
        StraightDistance.setup(self, src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk, dst_xyhz_j=dst_xyhz_j)
        if len(src_x_ilk) == 0:
            return
        # Calculate distance between src and dst and project to the down wind direction
        assert self.src_x_ilk.shape[1:] == (
            1, 1), 'TerrainFollowingDistance does not support flowcase dependent positions'
        src_x_ilk, src_y_ilk, src_h_ilk = self.src_x_ilk[:, 0, 0], self.src_y_ilk[:, 0, 0], self.src_h_ilk[:, 0, 0]
        if len(self.dst_x_j.shape) == 3:
            dst_x_j, dst_y_j = self.dst_x_j[:, 0, 0], self.dst_y_j[:, 0, 0]
        else:
            dst_x_j, dst_y_j = self.dst_x_j, self.dst_y_j

        # Generate interpolation lines

        if (self.src_eq_dst and self.dx_ijlk.shape[0] > 1):
            # calculate upper triangle of d_ij(distance from i to j) only
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for i, (src_x, src_y) in enumerate(zip(src_x_ilk, src_y_ilk))
                           for dst_x, dst_y in zip(dst_x_j[i + 1:], dst_y_j[i + 1:])])
            upper_tri_only = True
            self.theta_ij = gradients.arctan2(dst_y_j[na, :] - src_y_ilk[:, na],
                                              dst_x_j[na, :] - src_x_ilk[:, na])
        else:
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for src_x, src_y in zip(src_x_ilk, src_y_ilk)
                           for dst_x, dst_y in zip(dst_x_j, dst_y_j)])
            self.theta_ij = gradients.arctan2(dst_y_j[na, :, ] - src_y_ilk[:, na],
                                              dst_x_j[na, :] - src_x_ilk[:, na])
            upper_tri_only = False
        x, y = xy[:, 0], xy[:, 1]

        # find height along interpolation line
        h = self.site.elevation(x.flatten(), y.flatten()).reshape(x.shape)
        # calculate horizontal and vertical distance between interpolation points
        dxy = np.sqrt((x[:, 1] - x[:, 0])**2 + (y[:, 1] - y[:, 0])**2)
        dh = np.diff(h, 1, 1)
        # calculate distance along terrain following interpolation lines
        s = np.sum(np.sqrt(dxy[:, na]**2 + dh**2), 1)

        if upper_tri_only:
            # d_ij = np.zeros(self.dx_ijlk.shape)
            # d_ij[np.triu(np.eye(len(src_x_ilk)) == 0)] = s  # set upper triangle

            # same as above without item assignment
            n = len(src_x_ilk)
            d_ij = np.array([np.concatenate([[0] * (i + 1),
                                             s[int(n * i - (i * (i + 1) / 2)):][:n - i - 1]])
                             for i in range(n)])  # set upper and lower triangle
            d_ij += d_ij.T
        else:
            d_ij = s.reshape(self.dx_ijlk.shape[:2])
        self.d_ij = d_ij

    def __call__(self, WD_ilk=None, wd_l=None, src_idx=slice(None), dst_idx=slice(None)):
        # project terrain following distance between wts onto downwind direction
        # instead of projecting the distances onto first x,y and then onto down wind direction
        # we offset the wind direction by the direction between source and destination

        _, hcw_ijlk, dh_ijlk = StraightDistance.__call__(self, WD_ilk=WD_ilk, wd_l=wd_l,
                                                         src_idx=src_idx, dst_idx=dst_idx)
        if self.wind_direction == 'wd':
            WD_ilk = np.asarray(wd_l)[na, :, na]

        WD_il = mean_deg(WD_ilk, 2)

        if len(np.shape(dst_idx)) == 2:
            # dst_idx depends on wind direction
            WD_l = WD_il[0]
            dir_jl = 90 - rad2deg(self.theta_ij[src_idx, dst_idx])
            wdir_offset_jl = WD_l[na, :] - dir_jl
            theta_jl = deg2rad(90 - wdir_offset_jl)

            dw_ijlk = (- np.sin(theta_jl) * self.d_ij[src_idx, dst_idx])[na, :, :, na]
        else:
            dir_ij = 90 - rad2deg(self.theta_ij[src_idx, ][:, dst_idx])
            wdir_offset_ijl = np.asarray(WD_il)[:, na] - dir_ij[:, :, na]
            theta_ijl = deg2rad(90 - wdir_offset_ijl)
            dw_ijlk = (- np.sin(theta_ijl) * self.d_ij[src_idx][:, dst_idx][:, :, na])[..., na]

        return dw_ijlk, hcw_ijlk, dh_ijlk
