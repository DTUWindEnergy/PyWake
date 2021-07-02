import numpy as np
from numpy import newaxis as na
import matplotlib
from py_wake.utils.functions import mean_deg


class StraightDistance():

    def _cos_sin(self, wd):
        theta = np.deg2rad(90 - wd)
        cos = np.cos(theta)
        sin = np.sin(theta)
        return cos, sin

    def plot(self, wd_il, src_idx=slice(None), dst_idx=slice(None)):
        import matplotlib.pyplot as plt

        dw_ijl, hcw_ijl, _ = self(wd_il)
        wdirs = mean_deg(wd_il, 0)
        for l, wd in enumerate(wdirs):
            plt.figure()
            ax = plt.gca()
            theta = np.deg2rad(90 - wd)
            ax.set_title(wd)
            ax.arrow(0, 0, -np.cos(theta) * 20, -np.sin(theta) * 20, width=1)
            colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])]
            f = 2
            for i, x_, y_ in zip(np.arange(len(self.src_x_i))[src_idx], self.src_x_i[src_idx], self.src_y_i[src_idx]):
                c = colors[i % len(colors)]
                ax.plot(x_, y_, '2', color=c, ms=10, mew=3)
                for j, dst_x, dst_y in zip(np.arange(len(self.dst_x_j))[dst_idx],
                                           self.dst_x_j[dst_idx], self.dst_y_j[dst_idx]):
                    ax.arrow(x_ - j / f, y_ - j / f, -np.cos(theta) * dw_ijl[i, j, l], -
                             np.sin(theta) * dw_ijl[i, j, l], width=.3, color=c)
                    ax.plot([dst_x - i / f, dst_x - np.sin(theta) * hcw_ijl[i, j, l] - i / f],
                            [dst_y - i / f, dst_y + np.cos(theta) * hcw_ijl[i, j, l] - i / f], '--', color=c)
            plt.plot(self.src_x_i, self.src_y_i, 'k2')
            ax.axis('equal')
            ax.legend()

    def setup(self, src_x_i, src_y_i, src_h_i, dst_xyh_j=None):
        src_x_i, src_y_i, src_h_i = map(np.asarray, [src_x_i, src_y_i, src_h_i])
        self.src_x_i, self.src_y_i, self.src_h_i = src_x_i, src_y_i, src_h_i

        self.dx_ii = src_x_i - src_x_i[:, na]
        self.dy_ii = src_y_i - src_y_i[:, na]
        self.dh_ii = src_h_i - src_h_i[:, na]
        if dst_xyh_j is None:
            dst_x_j, dst_y_j, dst_h_j = src_x_i, src_y_i, src_h_i
            self.dx_ij, self.dy_ij, self.dh_ij = self.dx_ii, self.dy_ii, self.dh_ii
            self.src_eq_dst = True
        else:
            dst_x_j, dst_y_j, dst_h_j = dst_xyh_j
            dst_x_j, dst_y_j, dst_h_j = map(np.asarray, [dst_x_j, dst_y_j, dst_h_j])
            self.dx_ij = dst_x_j - src_x_i[:, na]
            self.dy_ij = dst_y_j - src_y_i[:, na]
            self.dh_ij = dst_h_j - src_h_i[:, na]
            self.src_eq_dst = False
        self.dst_x_j, self.dst_y_j, self.dst_h_j = dst_x_j, dst_y_j, dst_h_j

    def __call__(self, wd_il, src_idx=slice(None), dst_idx=slice(None)):
        assert hasattr(self, 'dx_ij'), "method setup must be called first"
        if len(np.shape(wd_il)) == 1:
            cos_l, sin_l = self._cos_sin(wd_il)
            dx_ij, dy_ij = self.dx_ii[src_idx, dst_idx], self.dy_ii[src_idx, dst_idx]
            dw_jl = (-cos_l[na] * dx_ij - sin_l[na] * dy_ij)
            hcw_jl = (sin_l[na] * dx_ij - cos_l[na] * dy_ij)
            return dw_jl, hcw_jl, self.dh_ii[src_idx, dst_idx]
        else:
            # wd_il
            # mean over all source points
            wd_l = mean_deg(wd_il, (0))
            wd_ijl = wd_l[na, na]
            cos_ijl, sin_ijl = self._cos_sin(wd_ijl)
            dx_ijl = self.dx_ij[src_idx][:, dst_idx][:, :, na]
            dy_ijl = self.dy_ij[src_idx][:, dst_idx][:, :, na]

            dw_ijl = -cos_ijl * dx_ijl - sin_ijl * dy_ijl
            hcw_ijl = sin_ijl * dx_ijl - cos_ijl * dy_ijl
            dh_ijl = np.broadcast_to(self.dh_ij[src_idx][:, dst_idx][:, :, na], dw_ijl.shape)
            return dw_ijl, hcw_ijl, dh_ijl

    def dw_order_indices(self, wd_l):
        assert hasattr(self, 'dx_ij'), "method setup must be called first"
        I, J = self.dx_ij.shape
        assert I == J
        cos_l, sin_l = self._cos_sin(np.asarray(wd_l))
        # return np.argsort(-cos_l[:, na] * np.asarray(src_x_i)[na] - sin_l[:, na] * np.asarray(src_y_i)[na], 1)
        dw_iil = -cos_l[na, na, :] * self.dx_ij[:, :, na] - sin_l[na, na, :] * self.dy_ij[:, :, na]
        dw_order_indices_l = np.argsort((dw_iil > 0).sum(0), 0).T
        return dw_order_indices_l


class TerrainFollowingDistance(StraightDistance):
    def __init__(self, distance_resolution=1000, **kwargs):
        super().__init__(**kwargs)
        self.distance_resolution = distance_resolution

    def setup(self, src_x_i, src_y_i, src_h_i, dst_xyh_j=None):
        StraightDistance.setup(self, src_x_i, src_y_i, src_h_i, dst_xyh_j=dst_xyh_j)
        # Calculate distance between src and dst and project to the down wind direction

        # Generate interpolation lines

        if (self.src_eq_dst and self.dx_ij.shape[0] > 1):
            # calculate upper triangle of d_ij(distance from i to j) only
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for i, (src_x, src_y) in enumerate(zip(self.src_x_i, self.src_y_i))
                           for dst_x, dst_y in zip(self.dst_x_j[i + 1:], self.dst_y_j[i + 1:])])
            upper_tri_only = True
        else:
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for src_x, src_y in zip(self.src_x_i, self.src_y_i)
                           for dst_x, dst_y in zip(self.dst_x_j, self.dst_y_j)])
            upper_tri_only = False
        x, y = xy[:, 0], xy[:, 1]

        # find height and calculate surface distance
        h = self.site.elevation(x.flatten(), y.flatten()).reshape(x.shape)
        dxy = np.sqrt((x[:, 1] - x[:, 0])**2 + (y[:, 1] - y[:, 0])**2)
        dh = np.diff(h, 1, 1)
        s = np.sum(np.sqrt(dxy[:, na]**2 + dh**2), 1)

        if upper_tri_only:
            d_ij = np.zeros(self.dx_ij.shape)
            d_ij[np.triu(np.eye(len(src_x_i)) == 0)] = s  # set upper triangle
            d_ij[np.tril(np.eye(len(src_x_i)) == 0)] = s  # set lower triangle
        else:
            d_ij = s.reshape(self.dx_ij.shape)
        self.d_ij = d_ij
        self.theta_ij = np.arctan2(self.dst_y_j - self.src_y_i[:, na], self.dst_x_j - self.src_x_i[:, na])

    def __call__(self, wd_il, src_idx=slice(None), dst_idx=slice(None)):
        # instead of projecting the distances onto first x,y and then onto down wind direction
        # we offset the wind direction by the direction between source and destination
        _, hcw_ijl, dh_ijl = StraightDistance.__call__(self, wd_il, src_idx, dst_idx)
        if len(np.shape(wd_il)) == 1:
            dir_ij = 90 - np.rad2deg(self.theta_ij[src_idx, dst_idx])
            wdir_offset_ij = np.asarray(wd_il)[na] - dir_ij
            theta_ij = np.deg2rad(90 - wdir_offset_ij)
            sin_ij = np.sin(theta_ij)
            dw_ijl = - sin_ij * self.d_ij[src_idx, dst_idx]
        else:
            dir_ij = 90 - np.rad2deg(self.theta_ij[src_idx, ][:, dst_idx])
            wdir_offset_ijl = np.asarray(wd_il)[:, na] - dir_ij[:, :, na]
            theta_ijl = np.deg2rad(90 - wdir_offset_ijl)
            sin_ijl = np.sin(theta_ijl)
            dw_ijl = - sin_ijl * self.d_ij[src_idx][:, dst_idx][:, :, na]

        return dw_ijl, hcw_ijl, dh_ijl


class TerrainFollowingDistance2():
    def __init__(self, k_star=0.075, r_i=None, calc_all=False, terrain_step=5, **kwargs):
        super().__init__(**kwargs)
        self.k_star = k_star
        self.r_i = r_i
        self.calc_all = calc_all
        self.terrain_step = terrain_step

    def __call__(self, wd_il):
        if not self.src_x_i.shape == self.dst_x_j.shape or not np.allclose(self.src_x_i, self.dst_x_j):
            raise NotImplementedError(
                'Different source and destination postions are not yet implemented for the terrain following distance calculation')
        return self.cal_dist_terrain_following(self.site, self.src_x_i, self.src_y_i, self.src_h_i,
                                               self.dst_x_j, self.dst_y_j, self.dst_h_j, wd_il,
                                               self.terrain_step, self.calc_all)

    def setup(self, src_x_i, src_y_i, src_h_i, dst_xyh_j=None):
        StraightDistance.setup(self, src_x_i, src_y_i, src_h_i, dst_xyh_j=dst_xyh_j)

    def cal_dist_terrain_following(self, site, src_x_i, src_y_i, src_h_i, dst_x_j,
                                   dst_y_j, dst_h_j, wd_il, step, calc_all):
        """ Calculate downwind and crosswind distances between a set of turbine
        sites, for a range of inflow wind directions. This version assumes the
        flow follows terrain at the same height above ground, and calculate the
        terrain following downwind/crosswind distances.

        Parameters
        ----------
        x_i: array:float
            x coordinates [m]

        y_i: array:float
            y coordinates [m]

        H_i: array:float
            hub-heights [m]

        wd: array:float
            local inflow wind direction [deg] (N = 0, E = 90, S = 180, W = 270)

        Note: wd can be a 1D array, which denotes the binned far field inflow
        wind direction, or it can be a num_sites by num_wds 2D array, which
        denotes the local wind direction for each sites. The 2D array version
        is used when the wind farm is at complex terrain, and the differences
        between local wind direction don't want to be neglected.

        elev_interp_func: any:'scipy.interpolate.interpolate.RegularGridInterpolator'
            interperating function to get elevation [m] at any site. It is a
            RegularGridInterpolator based function provided by site_condition.
            Its usage is like: elev = elev_interp_func(x, y), for sites
            outside the legal area, it will return nan.

        step: float
            stepsize when integrating to calculate terrain following distances
            [m], default: 5.0

        calc_all: bool
            If False (default) only distances to sites within wake are corrected
            for effects of terrain elevation

        Returns
        -------
        dist_down: array:float
            downwind distances between turbine sites for different far field
            inflow wind direction [m/s]

        dist_cross: array:float
            crosswind distances between turbine sites for all wd [m/s]

        downwind_order: array:integer
            downwind orders of turbine sites for different far field inflow
            wind direction [-]

        Note: dist_down[i, j, l] denotes downwind distance from site i to site
        j under lth inflow wd. downwind_order[:, l] denotes the downwind order
        of sites under lth inflow wd.
        """

        """ Calculate terrain following downwind distances from (x_start,
        y_starts) to a set of points at (x_points, y_points), assuming
        the wind blows along x direction. elev_interp_func is used to get elevation values,
        cos_rotate_back and sin_rotate_back are used to tranform the
        coordinates to real coordinates (to be used in elev_interp_func).
        Suffixs:
            - i: starting points
            - l: wind direction
            - s: destination points
        """
        wd_il = np.asarray(wd_il)
        if wd_il.ndim == 2 and wd_il.shape[0] == 1:
            wd_il = wd_il[0]
        elev_interp_func = site.elevation_interpolator
        x_i = src_x_i
        y_i = src_y_i
        H_i = src_h_i
        dist_down_straight_iil, dist_cross_iil, downwind_order_il, x_rotated_il, y_rotated_il, cos_wd_il, sin_wd_il, hcw_iil, dh_iil = self._cal_dist(
            x_i, y_i, H_i, dst_x_j, dst_y_j, dst_h_j, wd_il)

        dist_down_isl = dist_down_straight_iil
        dist_cross_isl = dist_cross_iil
        dist_hcw_isl = hcw_iil
        dist_dh_isl = dh_iil
        x_start_il = x_rotated_il
        y_start_il = y_rotated_il
        x_points_sl = x_rotated_il
        cos_rotate_back_il = cos_wd_il
        sin_rotate_back_il = -sin_wd_il

        I, L = x_start_il.shape
        i_wd_l = np.arange(L)
        na = np.newaxis
        if calc_all or not np.array(self.r_i).any():
            mask_isl = np.ones_like(dist_down_isl, dtype=bool)
        else:  # pragma: no cover
            r_i = self.r_i
            # Only consider if: r0 + k_star * dist_down < dist_cross-r1 and straight distance > 2*r
            # where r1 is radius of downwind turbine.
            # dist_down is expected to be less than dist_straight + 20%
            expected_wake_size = 1.2 * dist_down_isl * self.k_star * \
                np.ones(len(r_i))[:, na, na] + r_i[:, na, na] + r_i[na, :, na]
            mask_isl = (expected_wake_size > dist_cross_isl) & (dist_down_isl > (2 * r_i.max()))

        any_sites_in_wake = np.any(mask_isl, 1)
        for l in i_wd_l:
            for i in range(I):
                x_start = x_start_il[i, l]
                y_start = y_start_il[i, l]

                x_points_s = x_points_sl[:, l]

                if not any_sites_in_wake[i, l]:  # pragma: no cover
                    continue

                # find most downstream relevant site
                i_last_point = downwind_order_il[np.where(mask_isl[i, downwind_order_il[:, l], l])[0][-1], l]
                x_last_point = x_points_sl[i_last_point, l]

                cos_rotate_back = cos_rotate_back_il[i, l]
                sin_rotate_back = sin_rotate_back_il[i, l]

                # 2. if starting point is not already the most downwind point
                if x_start < x_last_point:

                    # 3. For points downwind of the starting point, do integration
                    # along a line for starting points to the final point
                    x_integ_line = np.arange(x_start, x_last_point + step, step)
                    x_integ_line[-1] = x_last_point

                    # note these coordinates need to be rotated back to get elevation
                    x_integ_line_back = (x_integ_line * cos_rotate_back +
                                         y_start * sin_rotate_back)
                    y_integ_line_back = (y_start * cos_rotate_back -
                                         x_integ_line * sin_rotate_back)
                    x_max_inds = x_integ_line_back > elev_interp_func.x.max()
                    y_max_inds = y_integ_line_back > elev_interp_func.y.max()
                    x_min_inds = x_integ_line_back < elev_interp_func.x.min()
                    y_min_inds = y_integ_line_back < elev_interp_func.y.min()
                    x_integ_line_back[x_max_inds] = elev_interp_func.x.max()
                    y_integ_line_back[y_max_inds] = elev_interp_func.y.max()
                    x_integ_line_back[x_min_inds] = elev_interp_func.x.min()
                    y_integ_line_back[y_min_inds] = elev_interp_func.y.min()
                    z_integ_line = elev_interp_func(x_integ_line_back,
                                                    y_integ_line_back,
                                                    mode='extrapolate')

                    # integration along the line
                    dx = np.diff(x_integ_line)
                    dz = np.diff(z_integ_line)
                    ds = np.sqrt(dx**2 + dz**2)
                    dist_line = np.cumsum(ds)

                    downwind = x_start < x_points_s
                    within_wake = mask_isl[i, :, l]
                    update = downwind & within_wake
                    dist_down_isl[i, update, l] = np.interp(x_points_s[update], x_integ_line[1:], dist_line)
#                    if np.isnan(dist_down_isl).any():
#                        print(dist_down_isl)
        self.dist_down, self.dist_cross, self.downwind_order = dist_down_isl, dist_hcw_isl, downwind_order_il.T

        return dist_down_isl, dist_hcw_isl, dist_dh_isl

    def _cal_dist(self, x_i, y_i, H_i, dst_x_j, dst_y_j, dst_h_j, wd):
        num_sites = len(x_i)

        if len(wd.shape) == 2:  # pragma: no cover
            num_wds = wd.shape[1]
            wd_mean_l = mean_deg(wd, 0)
            complex_flag = True
        else:
            complex_flag = False
            num_wds = wd.shape[0]
            wd_mean_l = wd

        I, L = num_sites, num_wds
        na = np.newaxis

        # rotate the coordinate so that wind blows along x axis
        rotate_angle_mean_l = (270 - wd_mean_l) * np.pi / 180.0

        dx_ii, dy_ii, dH_ii = [np.subtract(*np.meshgrid(v, v)) for v in [x_i, y_i, H_i]]
        if complex_flag:  # pragma: no cover
            rotate_angle_il = (270 - wd) * np.pi / 180.0
            cos_wd_il = np.cos(rotate_angle_il)
            sin_wd_il = np.sin(rotate_angle_il)
        else:
            cos_wd_il = np.broadcast_to(np.cos(rotate_angle_mean_l), (I, L))
            sin_wd_il = np.broadcast_to(np.sin(rotate_angle_mean_l), (I, L))

        # using local wind direction to rotate
        x_rotated_il = x_i[:, na] * cos_wd_il + y_i[:, na] * sin_wd_il
        y_rotated_il = y_i[:, na] * cos_wd_il - x_i[:, na] * sin_wd_il

        downwind_order_il = np.argsort(x_rotated_il, 0).astype(int)

        dist_down_iil = dx_ii[:, :, na] * cos_wd_il[:, na, :] + dy_ii[:, :, na] * sin_wd_il[:, na, :]
        dy_rotated_iil = dy_ii[:, :, na] * cos_wd_il[:, na, :] - dx_ii[:, :, na] * sin_wd_il[:, na, :]

        # treat crosswind distances as usual
        dist_cross_iil = np.sqrt(dy_rotated_iil**2 + dH_ii[:, :, na]**2)
        hcw_iil = dy_rotated_iil
        dh_iil = np.repeat(dH_ii[:, :, na], 360, axis=-1)

        return dist_down_iil, dist_cross_iil, downwind_order_il, x_rotated_il, y_rotated_il, cos_wd_il, sin_wd_il, hcw_iil, dh_iil
