from py_wake import np
from numpy import newaxis as na
from py_wake.tests import npt
from py_wake.site.distance import StraightDistance, TerrainFollowingDistance
from py_wake.site._site import UniformSite
import pytest
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake import NOJ
from py_wake.examples.data.ParqueFicticio import ParqueFicticioSite
from py_wake.flow_map import HorizontalGrid, XYGrid, XZGrid, Points
import matplotlib.pyplot as plt
from py_wake.utils.streamline import VectorField3D
from py_wake.site.jit_streamline_distance import JITStreamlineDistance
from py_wake.examples.data.hornsrev1 import V80

from py_wake.tests.test_wind_farm_models.test_enginering_wind_farm_model import OperatableV80
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, BastankhahGaussian
from py_wake.deficit_models.utils import ct2a_mom1d
import warnings


class FlatSite(UniformSite):
    def __init__(self, distance):
        UniformSite.__init__(self, p_wd=[1], ti=.075)
        self.distance = distance


class HalfCylinder(UniformSite):
    def __init__(self, height, distance_resolution):
        self.height = height
        super().__init__(p_wd=[1], ti=0)
        self.distance = TerrainFollowingDistance(distance_resolution=distance_resolution)

    def elevation(self, x_i, y_i):
        return np.sqrt(np.maximum(self.height**2 - x_i**2, 0))


class Rectangle(UniformSite):
    def __init__(self, height, width, distance_resolution):
        self.height = height
        self.width = width
        super().__init__(p_wd=[1], ti=0)
        self.distance = TerrainFollowingDistance(distance_resolution=distance_resolution)

    def elevation(self, x_i, y_i):
        return np.where(np.abs(x_i) < self.width / 2, self.height, 0)


@pytest.mark.parametrize('distance', [StraightDistance(),
                                      TerrainFollowingDistance()
                                      ])
def test_flat_distances(distance):
    x = [0, 50, 100, 100, 0]
    y = [100, 100, 100, 0, 0]
    h = [0, 10, 20, 30, 0]
    z = [0, 0, 0, 0]
    wdirs = [0, 30, 90]

    site = FlatSite(distance=distance)
    site.distance.setup(src_x_ilk=x, src_y_ilk=y, src_h_ilk=h, src_z_ilk=z)
    dw_ijlk, hcw_ijlk, dh_ijlk = site.distance(wd_l=np.array(wdirs), WD_ilk=None, src_idx=[0, 1, 2, 3], dst_idx=[4])
    dw_indices_lkd = site.distance.dw_order_indices(np.array(wdirs))

    if 0:
        distance.plot(wd_ilk=np.array(wdirs)[na, :, na], src_i=[0, 1, 2, 3], dst_i=[4])
        plt.show()

    npt.assert_array_almost_equal(dw_ijlk[..., 0], [[[100, 86.6025404, 0]],
                                                    [[100, 111.602540, 50]],
                                                    [[100, 136.602540, 100]],
                                                    [[0, 50, 100]]])
    npt.assert_array_almost_equal(hcw_ijlk[..., 0], [[[0, 50, 100]],
                                                     [[-50, 6.69872981, 100]],
                                                     [[-100, -36.6025404, 100]],
                                                     [[-100, -86.6025404, 0]]])
    npt.assert_array_almost_equal(dh_ijlk[..., 0], [[[0, 0, 0]],
                                                    [[-10, -10, -10]],
                                                    [[-20, -20, -20]],
                                                    [[-30, -30, -30]]])
    npt.assert_array_equal(dw_indices_lkd[:, 0, :4], [[2, 1, 0, 3],
                                                      [2, 1, 0, 3],
                                                      [2, 3, 1, 0]])


@pytest.mark.parametrize('distance', [StraightDistance(),
                                      TerrainFollowingDistance()])
def test_flat_distances_src_neq_dst(distance):
    x = [0, 50, 100]
    y = [100, 100, 0]
    h = [0, 10, 20]
    z = [0, 0, 0]
    wdirs = [0, 30]

    site = FlatSite(distance=distance)
    site.distance.setup(src_x_ilk=x, src_y_ilk=y, src_h_ilk=h, src_z_ilk=z, dst_xyhz_j=(x, y, [1, 2, 3], z))
    dw_ijlk, hcw_ijlk, dh_ijlk = site.distance(wd_l=np.array(wdirs), WD_ilk=None)
    dw_indices_lkd = distance.dw_order_indices(wdirs)
    if 0:
        distance.plot(wd_ilk=np.array(wdirs)[na, :, na])
        plt.show()

    # check down wind distance wind from North and 30 deg
    npt.assert_array_almost_equal(dw_ijlk[:, :, 0, 0], [[0, 0, 100],
                                                        [0, 0, 100],
                                                        [-100, -100, 0]])
    npt.assert_array_almost_equal(dw_ijlk[:, :, 1, 0], [[0, -25, 36.60254038],
                                                        [25, 0, 61.60254038],
                                                        [-36.60254038, -61.60254038, 0]])

    # check cross wind distance wind from North and 30 deg
    npt.assert_array_almost_equal(hcw_ijlk[:, :, 0, 0], [[0, 50, 100],
                                                         [-50, 0, 50],
                                                         [-100, -50, 0]])
    npt.assert_array_almost_equal(hcw_ijlk[:, :, 1, 0], [[0, 43.30127019, 136.60254038],
                                                         [-43.30127019, 0., 93.30127019],
                                                         [-136.60254038, -93.30127019, 0.]])
    # check cross wind distance wind from North
    npt.assert_array_almost_equal(dh_ijlk[:, :, 0, 0], [[1, 2, 3],
                                                        [-9, -8, -7],
                                                        [-19, -18, -17]])
    # check dw indices
    npt.assert_array_equal(dw_indices_lkd[:, 0], [[1, 0, 2],
                                                  [1, 0, 2]])


def test_iea37_distances():
    from py_wake.examples.data.iea37 import IEA37Site
    n_wt = 16  # must be 9, 16, 36, 64
    site = IEA37Site(n_wt)
    x, y = site.initial_position.T
    site.distance.wind_direction = 'WD_i'
    lw = site.local_wind(x=x, y=y,
                         wd=site.default_wd,
                         ws=site.default_ws)
    site.distance.setup(x, y, np.zeros_like(x), np.zeros_like(x))
    dw_iilk, hcw_iilk, _ = site.wt2wt_distances(WD_ilk=lw.WD_ilk, wd_l=None)
    # Wind direction.
    wdir = np.rad2deg(np.arctan2(hcw_iilk, dw_iilk))
    npt.assert_allclose(
        wdir[:, 0, 0, 0],
        [180, -90, -18, 54, 126, -162, -90, -54, -18, 18, 54, 90, 126, 162, -162, -126],
        atol=1e-4)

    if 0:
        _, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(np.arange(len(x))):
            ax.annotate(txt, (x[i], y[i]), fontsize='large')
        plt.show()


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind,
                                     All2AllIterative])
@pytest.mark.parametrize('turning', [(0, 0),
                                     (10, 0),
                                     (10, 20)
                                     ])
@pytest.mark.parametrize('method,angle_func', [('wd', lambda a:0),
                                               ('WD_i', lambda a:-a)
                                               ])
def test_straightDistance_turning(wfm_cls, turning, method, angle_func):

    wfm = wfm_cls(UniformSite(), OperatableV80(), wake_deficitModel=BastankhahGaussianDeficit(k=0.00001))
    wfm.site.distance = StraightDistance(wind_direction=method)

    ghost_y = np.linspace(-100, 100, 31)
    ghost_x = np.full(ghost_y.shape, 450)
    operation = [1, 1] + ([0] * len(ghost_x))
    WD = np.array(np.r_[turning, [0] * len(ghost_y)]) + 270
    sim_res = wfm(np.r_[[0, 500], ghost_x], np.r_[[0, 0], ghost_y], operating=operation, wd=[0, 270], WD=WD)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fm_wide = sim_res.flow_map(XYGrid(x=450, y=np.linspace(-200, 200, 1001)), wd=270)

    y = fm_wide.y[np.argmin(fm_wide.WS_eff.squeeze().values)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fm = sim_res.flow_map(XYGrid(x=450, y=np.linspace(y - 2, y + 2, 1001)), wd=270)

    if 0:
        ax1, ax2 = plt.subplots(1, 2)[1]
        plt.suptitle(f'{wfm_cls.__name__}, turning{turning}')
        sim_res.flow_map(wd=270).plot_wake_map(ax=ax1)
        ax1.plot(fm_wide.y.squeeze() * 0 + fm_wide.x.squeeze(), fm_wide.y.squeeze(), '.-')

        fm_wide.WS_eff.plot(ax=ax2)
        fm.WS_eff.plot(ax=ax2)
        ax2.plot(sim_res.y[2:], sim_res.WS_eff.sel(wd=270).squeeze()[2:], 'x')
        plt.show()

    # compare flowmap to ghost turbines
    npt.assert_array_almost_equal(sim_res.WS_eff.sel(wd=270).squeeze()[2:],
                                  np.interp(ghost_y, fm_wide.y, fm_wide.WS_eff.squeeze()), 3)

    # check angle of deficit peak
    i = np.argmin(fm.WS_eff.squeeze().values)
    npt.assert_almost_equal(np.rad2deg(np.arctan2(fm.y[i], 450)), angle_func(turning[0]), 3)


def test_terrain_following_half_cylinder():

    hc = HalfCylinder(height=100, distance_resolution=100000)

    src_x, src_y = np.array([-100, -50, 0]), [0, 0, 0]
    dst_x, dst_y = np.array([100, 200, 300, 400]), [0, 0, 0, 0]
    x = np.arange(-150, 150)

    hc.distance.setup(src_x_ilk=src_x, src_y_ilk=src_y, src_h_ilk=src_x * 0, src_z_ilk=src_x * 0,
                      dst_xyhz_j=(dst_x, dst_y, dst_x * 0, dst_x * 0))
    dw_ijlk, hcw_ijlk, _ = hc.distance(wd_l=np.array([0, 90]), WD_ilk=None)

    if 0:
        plt.plot(x, hc.elevation(x_i=x, y_i=x * 0))
        plt.plot(src_x, hc.elevation(x_i=src_x, y_i=src_y), '.')
        plt.plot(dst_x, dst_y, 'o')
        plt.axis('equal')
        plt.show()

    dist2flat = np.pi * np.array([1, 2 / 3, .5]) * 100
    dist2flat = dist2flat[:, na] + (np.arange(4) * 100)
    npt.assert_array_almost_equal(dw_ijlk[:, :, 1, 0], -dist2flat, 2)
    npt.assert_array_almost_equal(hcw_ijlk[:, :, 0, 0], [[200., 300., 400., 500.],
                                                         [150., 250., 350., 450.],
                                                         [100., 200., 300., 400.]], 2)

    # down wind distance for 0 deg and cross wind distance for 30 deg ~ 0
    npt.assert_array_almost_equal(dw_ijlk[:, :, 0, 0], 0)
    npt.assert_array_almost_equal(hcw_ijlk[:, :, 1, 0], 0)


def test_distance_over_rectangle():
    x, y = [-100, 50], [200, -100]
    windTurbines = IEA37_WindTurbines()
    site = Rectangle(height=200, width=100, distance_resolution=100)
    wf_model = NOJ(site, windTurbines, ct2a=ct2a_mom1d)
    sim_res = wf_model(x, y, wd=[270], ws=[9])
    x_j = np.linspace(-100, 500, 50)
    y_j = np.linspace(-200, 300, 50)
    flow_map = sim_res.flow_map(HorizontalGrid(x_j, y_j))
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]
    X, Y = flow_map.X, flow_map.Y

    my = np.argmin(np.abs(Y[:, 0] - 200))

    if 0:
        flow_map.plot_wake_map()
        H = site.elevation(X, Y)
        plt.plot(X[my], Z[my] * 10, label='wsp*10')

        plt.contour(X, Y, H)
        plt.plot(X[my, :25:2], Y[my, :25:2], '.-', label='wsp points')
        plt.plot(x_j, site.elevation(x_j, x_j * 0), label='terrain level')
        plt.legend()
        plt.show()

    ref = [3., 3.42, 3.8, 6.02, 6.17, 6.31, 6.43, 7.29, 7.35, 7.41, 7.47, 7.53, 7.58]
    npt.assert_array_almost_equal(Z[my, :25:2], ref, 2)


def test_distance_over_rectangle2():
    site = Rectangle(height=200, width=100, distance_resolution=200)

    x_j = np.arange(-200, 500, 10)
    y_j = x_j * 0
    site.distance.setup([-200], [0], [130], [0], (x_j, y_j, y_j + 130, y_j * 0))
    d = site.distance(wd_l=[270])[0][0, :, 0, 0]

    ref = x_j - x_j[0]
    ref[x_j > -50] += 200
    ref[x_j >= 50] += 200

    if 0:
        H = site.elevation(x_j, y_j)
        plt.plot(x_j, H, '.-', label='terrain')
        plt.plot(x_j, d, label='distance')
        plt.plot(x_j, ref, label='reference')
        plt.legend()
        plt.show()

    npt.assert_allclose(d, ref, rtol=0.01)


def test_distance_plot():

    x = [0, 50, 100, 100, 0]
    y = [100, 100, 100, 0, 0]
    h = [0, 10, 20, 30, 0]
    z = [0, 0, 0, 0, 0]
    wdirs = [0, 30, 90]
    distance = StraightDistance()
    distance.setup(src_x_ilk=x, src_y_ilk=y, src_h_ilk=h, src_z_ilk=z)
    distance.plot(wd_l=np.array(wdirs), src_idx=[0], dst_idx=[3])
    if 0:
        plt.show()
    plt.close('all')


def test_JITStreamlinesparquefictio():
    site = ParqueFicticioSite()
    wt = IEA37_WindTurbines()
    vf3d = VectorField3D.from_WaspGridSite(site)
    site.distance = JITStreamlineDistance(vf3d)

    x, y = site.initial_position[:].T
    wfm = NOJ(site, wt)
    wd = np.array([330])
    sim_res = wfm(x, y, wd=wd, ws=10)
    dw = site.distance(wd_l=wd, WD_ilk=np.repeat(wd[na, na], len(x), 0))[0][:, :, 0, 0]
    # streamline downwind distance (positive numbers, upper triangle) cannot be shorter than
    # straight line distances in opposite direction (negative numbers, lower triangle)
    assert (dw + dw.T).min() >= 0
    # average downwind distance increase around 5 m
    npt.assert_almost_equal((dw + dw.T).mean(), 5, 0)

    fm = sim_res.flow_map(XYGrid(x=np.linspace(site.ds.x[0], site.ds.x[-1], 500),
                                 y=np.linspace(site.ds.y[0], site.ds.y[-1], 500)))
    stream_lines = vf3d.stream_lines(wd=np.full(x.shape, wd), start_points=np.array([x, y, np.full(x.shape, 70)]).T,
                                     dw_stop=y - 6504700)
    if 1:
        fm.plot_wake_map()
        for sl in stream_lines:
            plt.plot(sl[:, 0], sl[:, 1])

        plt.show()

    print()


def test_JITStreamlinesparquefictio_yz():
    site = ParqueFicticioSite()
    site.ds.Turning[:] *= 0
    # site.ds.flow_inc[:] *= 5
    wt = IEA37_WindTurbines()
    vf3d = VectorField3D.from_WaspGridSite(site)
    site.distance = JITStreamlineDistance(vf3d)

    x, y = site.initial_position[3].T
    wt_x = np.r_[x - 500, x, x + 500]
    wt_y = np.r_[y, y, y]
    wfm = BastankhahGaussian(site, wt, k=0.03)
    wd = np.array([270])
    sim_res = wfm(wt_x, wt_y, wd=wd, ws=10)
    dw = site.distance(wd_l=wd, WD_ilk=np.repeat(wd[na, na], len(wt_x), 0))[0][:, :, 0, 0]
    # streamline downwind distance (positive numbers, upper triangle) cannot be shorter than
    # straight line distances in opposite direction (negative numbers, lower triangle)
    assert (dw + dw.T).min() >= 0
    # average downwind distance increase around 5 m
    # npt.assert_almost_equal((dw + dw.T)[0, 1], 2, 0)

    fm = sim_res.flow_map(XZGrid(x=np.linspace(x - 700, x + 700, 500),
                                 z=np.linspace(30, 200, 50), y=y))
    stream_lines = vf3d.stream_lines(wd=wd, start_points=np.array([wt_x, wt_y, np.full(wt_x.shape, wt.hub_height())]).T,
                                     dw_stop=np.array([1200, 700, 200]))
    px, py, pz = stream_lines[0, 7]
    pz -= np.diff(site.elevation(*stream_lines[0, (0, 7), :2].T))

    z_lst = np.linspace(-20, 20, 9)
    fm_points = sim_res.flow_map(Points(z_lst * 0 + px, z_lst * 0 + y, z_lst + pz))

    if 0:
        fm.plot_wake_map()
        for sl in stream_lines:
            plt.plot(sl[:, 0], sl[:, 2] + site.elevation(sl[0, 0], sl[0, 1]), '.-')
        plt.plot(fm.x, site.elevation(fm.x, fm.x.values * 0 + fm.y.values))
        plt.plot(fm.x, site.elevation(fm.x, fm.x.values * 0 + fm.y.values) + 110, '--k')
        site.ds.flow_inc.interp(y=y, wd=270)
        plt.plot(z_lst * 0 + px, z_lst + pz + site.elevation(px, py), '.')
        plt.figure()
        fm_points.WS_eff.plot()
        plt.show()

    assert np.argmin(fm_points.WS_eff.values) == 4  # minimum WS_eff should be at streamline

    fm = sim_res.flow_map(Points(wt_x[1:] - 1e-6, wt_y[1:], np.full_like(wt_x[1:], wt.hub_height())))
    npt.assert_allclose(fm.WS_eff, sim_res.WS_eff.sel(wt=[1, 2]).squeeze())
