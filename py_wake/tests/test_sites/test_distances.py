import numpy as np
from numpy import newaxis as na
from py_wake.tests import npt
from py_wake.site.distance import StraightDistance, TerrainFollowingDistance, TerrainFollowingDistance2
from py_wake.site._site import UniformSite
import pytest
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.wake_models.noj import NOJ
from py_wake.aep_calculator import AEPCalculator
from py_wake.examples.data.ParqueFicticio.parque_ficticio import ParqueFicticioSite


class FlatTerrainFollowingDistance(UniformSite, TerrainFollowingDistance):
    def __init__(self):
        UniformSite.__init__(self, p_wd=[1], ti=.075)


class HalfCylinder(TerrainFollowingDistance, UniformSite):
    def __init__(self, height, distance_resolution):
        self.height = height
        super().__init__(distance_resolution=distance_resolution, p_wd=[1], ti=0)
        # TerrainFollowingDistance.__init__(self, )

    def elevation(self, x_i, y_i):
        return np.sqrt(np.maximum(self.height**2 - x_i**2, 0))


class Rectangle(TerrainFollowingDistance, UniformSite):
    def __init__(self, height, width, distance_resolution):
        self.height = height
        self.width = width
        super().__init__(distance_resolution=distance_resolution, p_wd=[1], ti=0)
        # TerrainFollowingDistance.__init__(self, )

    def elevation(self, x_i, y_i):
        return np.where(np.abs(x_i) < self.width / 2, self.height, 0)


def ParqueFicticioSiteTerrainFollowingDistance2():
    site = ParqueFicticioSite(distance=TerrainFollowingDistance2())
    x, y = site.initial_position.T
    return site, x, y


@pytest.mark.parametrize('Distance', [StraightDistance(),
                                      FlatTerrainFollowingDistance()])
def test_flat_distances(Distance):
    x = [0, 50, 100, 100]
    y = [100, 100, 100, 0]
    h = [0, 10, 20, 30]
    wdirs = [0, 30, 90]
    distances = Distance.distances
    dw_ijl, hcw_ijl, dh_ijl, dw_indices_l = distances(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=[0], dst_y_j=[0], dst_h_j=[0],
                                                      wd_il=np.array(wdirs)[na])

    if 0:
        Distance.plot(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=[0], dst_y_j=[0], dst_h_j=[0],
                      wd_il=np.array(wdirs)[na])

    npt.assert_array_almost_equal(dw_ijl, [[[100, 86.6025404, 0]],
                                           [[100, 111.602540, 50]],
                                           [[100, 136.602540, 100]],
                                           [[0, 50, 100]]])
    npt.assert_array_almost_equal(hcw_ijl, [[[0, 50, 100]],
                                            [[-50, 6.69872981, 100]],
                                            [[-100, -36.6025404, 100]],
                                            [[-100, -86.6025404, 0]]])
    npt.assert_array_almost_equal(dh_ijl, [[[0, 0, 0]],
                                           [[-10, -10, -10]],
                                           [[-20, -20, -20]],
                                           [[-30, -30, -30]]])
    npt.assert_array_equal(dw_indices_l, [[2, 1, 0, 3],
                                          [2, 1, 0, 3],
                                          [2, 3, 1, 0]])


@pytest.mark.parametrize('Distance', [  # StraightDistance(),
    FlatTerrainFollowingDistance()])
def test_flat_distances_wt2wt(Distance):
    x = [0, 50, 100]
    y = [100, 100, 0]
    h = [0, 10, 20]
    wdirs = [0, 30]
    distances = Distance.distances
    dw_ijl, hcw_ijl, dh_ijl, dw_indices_l = distances(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=x, dst_y_j=y, dst_h_j=[1, 2, 3],
                                                      wd_il=np.array(wdirs)[na])
    if 0:
        Distance.plot(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=x, dst_y_j=y, dst_h_j=[1, 2, 3],
                      wd_ijl=np.array(wdirs)[na, na])

    # check down wind distance wind from North and 30 deg
    npt.assert_array_almost_equal(dw_ijl[:, :, 0], [[0, 0, 100],
                                                    [0, 0, 100],
                                                    [-100, -100, 0]])
    npt.assert_array_almost_equal(dw_ijl[:, :, 1], [[0, -25, 36.60254038],
                                                    [25, 0, 61.60254038],
                                                    [-36.60254038, -61.60254038, 0]])

    # check cross wind distance wind from North and 30 deg
    npt.assert_array_almost_equal(hcw_ijl[:, :, 0], [[0, 50, 100],
                                                     [-50, 0, 50],
                                                     [-100, -50, 0]])
    npt.assert_array_almost_equal(hcw_ijl[:, :, 1], [[0, 43.30127019, 136.60254038],
                                                     [-43.30127019, 0., 93.30127019],
                                                     [-136.60254038, -93.30127019, 0.]])
    # check cross wind distance wind from North
    npt.assert_array_almost_equal(dh_ijl[:, :, 0], [[1, 2, 3],
                                                    [-9, -8, -7],
                                                    [-19, -18, -17]])
    # check dw indices
    npt.assert_array_equal(dw_indices_l, [[1, 0, 2],
                                          [1, 0, 2]])


def test_iea37_distances():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 9, 16, 36, 64
    site = IEA37Site(n_wt)
    x, y = site.initial_position.T
    WD_ilk, _, _, _ = site.local_wind(x_i=x, y_i=y,
                                      wd=site.default_wd,
                                      ws=site.default_ws)
    dw_iil, hcw_iil, _, _ = site.wt2wt_distances(
        x_i=x, y_i=y,
        h_i=np.zeros_like(x),
        wd_il=WD_ilk.mean(2))
    # Wind direction.
    wdir = np.rad2deg(np.arctan2(hcw_iil, dw_iil))
    npt.assert_allclose(
        wdir[:, 0, 0],
        [180, -90, -18, 54, 126, -162, -90, -54, -18, 18, 54, 90, 126, 162, -162, -126],
        atol=1e-4)

    if 0:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(np.arange(len(x))):
            ax.annotate(txt, (x[i], y[i]), fontsize='large')


def test_terrain_following_half_cylinder():

    hc = HalfCylinder(height=100, distance_resolution=100000)

    src_x, src_y = np.array([-100, -50, 0]), [0, 0, 0]
    dst_x, dst_y = np.array([100, 200, 300, 400]), [0, 0, 0, 0]
    x = np.arange(-150, 150)

    dw_ijl, hcw_ijl, _, _ = hc.distances(src_x_i=src_x, src_y_i=src_y, src_h_i=src_x * 0,
                                         dst_x_j=dst_x, dst_y_j=dst_y, dst_h_j=dst_x * 0,
                                         wd_il=np.array([0, 90])[na])

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, hc.elevation(x_i=x, y_i=x * 0))
        plt.plot(src_x, hc.elevation(x_i=src_x, y_i=src_y), '.')
        plt.plot(dst_x, dst_y, 'o')
        plt.axis('equal')
        plt.show()

    dist2flat = np.pi * np.array([1, 2 / 3, .5]) * 100
    dist2flat = dist2flat[:, na] + (np.arange(4) * 100)
    npt.assert_array_almost_equal(dw_ijl[:, :, 1], -dist2flat, 2)
    npt.assert_array_almost_equal(hcw_ijl[:, :, 0], [[200., 300., 400., 500.],
                                                     [150., 250., 350., 450.],
                                                     [100., 200., 300., 400.]], 2)

    # down wind distance for 0 deg and cross wind distance for 30 deg ~ 0
    npt.assert_array_almost_equal(dw_ijl[:, :, 0], 0)
    npt.assert_array_almost_equal(hcw_ijl[:, :, 1], 0)


def test_distance_over_rectangle():
    x, y = [-100, 50], [200, -100]
    windTurbines = IEA37_WindTurbines()
    site = Rectangle(height=200, width=100, distance_resolution=100)
    wake_model = NOJ(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-100, 500, 100)
    y_j = np.linspace(-200, 300, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[270], ws=[9])

    my = np.argmin(np.abs(Y[:, 0] - 200))
    my2 = np.argmin(np.abs(Y[:, 0] + 100))

    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, cmap='Blues_r', levels=100)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        H = site.elevation(X, Y)
        plt.plot(X[my], Z[my] * 10, label='wsp*10')
        plt.plot(X[my2], Z[my2] * 10, label='wsp*10')
        plt.contour(X, Y, H)
        plt.plot(X[my, :50:4], Z[my, :50:4] * 10, '.')
        plt.plot(x_j, site.elevation(x_j, x_j * 0), label='terrain level')
        plt.legend()
        plt.show()

    ref = [9., 3.42, 3.8, 6.02, 6.17, 6.31, 6.43, 7.29, 7.35, 7.41, 7.47, 7.53, 7.58]
    npt.assert_array_almost_equal(Z[my, :50:4], ref, 2)


def test_distance_plot():

    x = [0, 50, 100, 100]
    y = [100, 100, 100, 0]
    h = [0, 10, 20, 30]
    wdirs = [0, 30, 90]
    Distance = StraightDistance()
    distances = Distance.distances
    dw_ijl, hcw_ijl, dh_ijl, dw_indices_l = distances(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=[0], dst_y_j=[0], dst_h_j=[0],
                                                      wd_il=np.array(wdirs)[na])

    Distance.plot(src_x_i=x, src_y_i=y, src_h_i=h, dst_x_j=[0], dst_y_j=[0], dst_h_j=[0],
                  wd_il=np.array(wdirs)[na])
    if 0:
        import matplotlib.pyplot as plt
        plt.show()


# ======================================================================================================================
# TerrainFollowingDistance2
# ======================================================================================================================

def test_distances_ri():
    site, x, y = ParqueFicticioSiteTerrainFollowingDistance2()
    site.calc_all = False
    site.r_i = np.ones(len(x)) * 90
    dw_ijl, cw_ijl, dh_ijl, _ = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                               dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                               wd_il=np.array([[180]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., -207., -477., -710., -1016., -1236., -1456., -1799.]))
    npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([-236.1, 0., 131.1, 167.8, 204.5, 131.1, 131.1, 45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distance2_outside_map_WestEast():
    site = ParqueFicticioSiteTerrainFollowingDistance2()[0]

    import matplotlib.pyplot as plt

    site.plot_map('elev')
    x = np.arange(-1500, 1000, 500) + 264777
    h = x * 0
    y = h + 6505450

    dw = site.distances(src_x_i=x, src_y_i=y, src_h_i=h,
                        dst_x_j=x, dst_y_j=y, dst_h_j=h * 0, wd_il=[270])[0]

    if 0:
        plt.plot(x, y, '.-', label='Terrain line')
        plt.plot(x, y + site.elevation(x, y))
        plt.legend()
        plt.show()
    # distance between points should be >500 m due to terrain, except last point which is outside map
    npt.assert_array_equal(np.round(np.diff(dw[0, :, 0])), [527, 520, 505, 500.])


def test_distance2_outside_map_NorthSouth():
    site = ParqueFicticioSiteTerrainFollowingDistance2()[0]

    import matplotlib.pyplot as plt

    site.plot_map('elev')
    y = np.arange(-1500, 1000, 500) + 6506613.0
    h = y * 0
    x = h + 264200

    dw = site.distances(src_x_i=x, src_y_i=y, src_h_i=h,
                        dst_x_j=x, dst_y_j=y, dst_h_j=h * 0, wd_il=[180])[0]

    if 0:
        plt.plot(x, y, '.-', label='Terrain line')
        plt.plot(x + site.elevation(x, y), y)
        plt.legend()
        plt.show()
    # distance between points should be >500 m due to terrain, except last point which is outside map
    npt.assert_array_equal(np.round(np.diff(dw[0, :, 0])), [510, 505, 507, 500])
