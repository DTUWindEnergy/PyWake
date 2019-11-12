import numpy as np
from numpy import newaxis as na
import matplotlib


class StraightDistance():
    def _cos_sin(self, wd):
        theta = np.deg2rad(90 - wd)
        cos = np.cos(theta)
        sin = np.sin(theta)
        return cos, sin

    def plot(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        import matplotlib.pyplot as plt
        dw_ijl, hcw_ijl, dh_ijl, dw_order_indices_l = self.distances(
            src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il)
        wdirs = wd_il.mean(0)
        I, J, L = len(src_x_i), len(dst_x_j), len(wdirs)
        for l, wd in enumerate(wdirs):
            plt.figure()
            ax = plt.gca()
            theta = np.deg2rad(90 - wd)
            ax.set_title(wd)
            ax.arrow(0, 0, -np.cos(theta) * 20, -np.sin(theta) * 20, width=1)
            colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])]
            f = 2
            for i in range(I):
                i_dw = dw_order_indices_l[l][i]
                x_, y_ = src_x_i[i_dw], src_y_i[i_dw]
                c = colors[i % len(colors)]
                ax.plot(x_, y_, '2', color=c, ms=10, mew=3, label=i_dw)
                for j in range(J):
                    dst_x, dst_y = dst_x_j[j], dst_y_j[j]
                    ax.arrow(x_ - j / f, y_ - j / f, -np.cos(theta) * dw_ijl[i_dw, j, l], -
                             np.sin(theta) * dw_ijl[i_dw, j, l], width=.3, color=c)
                    ax.plot([dst_x - i / f, dst_x - np.sin(theta) * hcw_ijl[i_dw, j, l] - i / f],
                            [dst_y - i / f, dst_y + np.cos(theta) * hcw_ijl[i_dw, j, l] - i / f], '--', color=c)
            plt.plot(src_x_i, src_y_i, 'k2')
            ax.axis('equal')
            ax.legend()

    def project_distance(self, dx_ij, dy_ij, wd_ijl):
        cos_ijl, sin_ijl = self._cos_sin(wd_ijl)
        dw_ijl = -cos_ijl * dx_ij[:, :, na] - sin_ijl * dy_ij[:, :, na]
        hcw_ijl = sin_ijl * dx_ij[:, :, na] - cos_ijl * dy_ij[:, :, na]
        return dw_ijl, hcw_ijl

    def dw_order_indices(self, src_x_i, src_y_i, wd_l):
        cos_l, sin_l = self._cos_sin(wd_l)
        dx_ii, dy_ii = [np.subtract(*np.meshgrid(dst_j, src_i, indexing='ij')).T
                        for src_i, dst_j in [(src_x_i, src_x_i),
                                             (src_y_i, src_y_i)]]
        dw_iil = -cos_l[na, na, :] * dx_ii[:, :, na] - sin_l[na, na, :] * dy_ii[:, :, na]
        dw_order_indices_l = np.argsort((dw_iil > 0).sum(0), 0).T
        return dw_order_indices_l

    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):

        dx_ij, dy_ij, dh_ij = [np.subtract(*np.meshgrid(dst_j, src_i, indexing='ij')).T
                               for src_i, dst_j in [(src_x_i, dst_x_j),
                                                    (src_y_i, dst_y_j),
                                                    (src_h_i, dst_h_j)]]
        src_x_i, src_y_i = map(np.asarray, [src_x_i, src_y_i])
        # let the wind direction correspont to the mean wind directions of all source points
        wd_l = np.mean(wd_il, (0))
        wd_ijl = wd_l[na, na]

        dw_ijl, hcw_ijl = self.project_distance(dx_ij, dy_ij, wd_ijl)
        dh_ijl = np.zeros_like(dw_ijl)
        dh_ijl[:, :, :] = dh_ij[:, :, na]

        dw_order_indices_l = self.dw_order_indices(src_x_i, src_y_i, wd_l)

        return dw_ijl, hcw_ijl, dh_ijl, dw_order_indices_l


class TerrainFollowingDistance(StraightDistance):
    def __init__(self, distance_resolution=1000, **kwargs):
        super().__init__(**kwargs)
        self.distance_resolution = distance_resolution

    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        _, hcw_ijl, dh_ijl, dw_order_indices_l = StraightDistance.distances(
            self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il)

        # Calculate distance between src and dst and project to the down wind direction
        src_x_i, src_y_i, dst_x_j, dst_y_j = map(np.asarray, [src_x_i, src_y_i, dst_x_j, dst_y_j])

        # Generate interpolation lines
        if len(src_x_i) == len(dst_x_j) and np.all(src_x_i == dst_x_j) and np.all(src_y_i == dst_y_j):
            # calculate upper triangle of d_ij(distance from i to j) only
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for i, (src_x, src_y) in enumerate(zip(src_x_i, src_y_i))
                           for dst_x, dst_y in zip(dst_x_j[i + 1:], dst_y_j[i + 1:])])
            upper_tri_only = True
        else:
            xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                            np.linspace(src_y, dst_y, self.distance_resolution))
                           for src_x, src_y in zip(src_x_i, src_y_i)
                           for dst_x, dst_y in zip(dst_x_j, dst_y_j)])
            upper_tri_only = False
        x, y = xy[:, 0], xy[:, 1]

        # find height and calculate surface distance
        h = self.elevation(x.flatten(), y.flatten()).reshape(x.shape)
        dxy = np.sqrt((x[:, 1] - x[:, 0])**2 + (y[:, 1] - y[:, 0])**2)
        dh = np.diff(h, 1, 1)
        s = np.sum(np.sqrt(dxy[:, na]**2 + dh**2), 1)

        if upper_tri_only:
            d_ij = np.zeros((len(src_x_i), len(dst_x_j)))
            d_ij[np.triu(np.eye(len(src_x_i)) == 0)] = s  # set upper triangle
            d_ij[np.tril(np.eye(len(src_x_i)) == 0)] = s  # set lower triangle
        else:
            d_ij = s.reshape(len(src_x_i), len(dst_x_j))

        # instead of projecting the distances onto first x,y and then onto down wind direction
        # we offset the wind direction by the direction between source and destination
        theta_ij = np.arctan2(dst_y_j - src_y_i[:, na], dst_x_j - src_x_i[:, na])
        dir_ij = 90 - np.rad2deg(theta_ij)
        wdir_offset_ijl = wd_il[:, na] - dir_ij[:, :, na]
        theta_ijl = np.deg2rad(90 - wdir_offset_ijl)
        sin_ijl = np.sin(theta_ijl)
        dw_ijl = - sin_ijl * d_ij[:, :, na]

        return dw_ijl, hcw_ijl, dh_ijl, dw_order_indices_l


class TerrainFollowingDistance2():
    def __init__(self, k_star=0.075, r_i=None, calc_all=False, terrain_step=5, **kwargs):
        super().__init__(**kwargs)
        self.k_star = k_star
        self.r_i = r_i
        self.calc_all = calc_all
        self.terrain_step = terrain_step

    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        if not src_x_i.shape == dst_x_j.shape or not np.allclose(src_x_i, dst_x_j):
            raise NotImplementedError(
                'Different source and destination postions are not yet implemented for the terrain following distance calculation')
        return self.cal_dist_terrain_following(src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il, self.terrain_step, self.calc_all)

    def cal_dist_terrain_following(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il, step, calc_all):
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
        elev_interp_func = self.elevation_interpolator
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
            mask_isl = np.ones_like(dist_down_isl, dtype=np.bool)
        else:
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
                if i == 18:
                    print(i)
                x_start = x_start_il[i, l]
                y_start = y_start_il[i, l]

                x_points_s = x_points_sl[:, l]

                if not any_sites_in_wake[i, l]:
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

        return dist_down_isl, dist_hcw_isl, dist_dh_isl, downwind_order_il.T

    def _cal_dist(self, x_i, y_i, H_i, dst_x_j, dst_y_j, dst_h_j, wd):
        num_sites = len(x_i)

        if len(wd.shape) == 2:
            num_wds = wd.shape[1]
            wd_mean_l = np.mean(wd, 0)
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
        if complex_flag:
            rotate_angle_il = (270 - wd) * np.pi / 180.0
            cos_wd_il = np.cos(rotate_angle_il)
            sin_wd_il = np.sin(rotate_angle_il)
        else:
            cos_wd_il = np.broadcast_to(np.cos(rotate_angle_mean_l), (I, L))
            sin_wd_il = np.broadcast_to(np.sin(rotate_angle_mean_l), (I, L))

        # using local wind direction to rotate
        x_rotated_il = x_i[:, na] * cos_wd_il + y_i[:, na] * sin_wd_il
        y_rotated_il = y_i[:, na] * cos_wd_il - x_i[:, na] * sin_wd_il

        downwind_order_il = np.argsort(x_rotated_il, 0).astype(np.int)

        dist_down_iil = dx_ii[:, :, na] * cos_wd_il[:, na, :] + dy_ii[:, :, na] * sin_wd_il[:, na, :]
        dy_rotated_iil = dy_ii[:, :, na] * cos_wd_il[:, na, :] - dx_ii[:, :, na] * sin_wd_il[:, na, :]

        # treat crosswind distances as usual
        dist_cross_iil = np.sqrt(dy_rotated_iil**2 + dH_ii[:, :, na]**2)
        hcw_iil = dy_rotated_iil
        dh_iil = np.repeat(dH_ii[:, :, na], 360, axis=-1)

        return dist_down_iil, dist_cross_iil, downwind_order_il, x_rotated_il, y_rotated_il, cos_wd_il, sin_wd_il, hcw_iil, dh_iil
