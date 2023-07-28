from py_wake.flow_map import HorizontalGrid, YZGrid, Points, XYGrid, XZGrid
from py_wake.tests import npt
import matplotlib.pyplot as plt
from py_wake import np
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.site.distance import StraightDistance
from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
import pytest
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.wind_turbines._wind_turbines import WindTurbines, WindTurbine
from py_wake.examples.data import wtg_path, hornsrev1
from py_wake.utils.profiling import timeit
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.examples.data.hornsrev1 import V80
from py_wake.literature.iea37_case_study1 import IEA37CaseStudy1
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum
import warnings


@pytest.fixture(autouse=True)
def close_plots():
    yield
    try:
        plt.close('all')
    except Exception:
        pass


def test_power_xylk():
    wind_farm_model = IEA37CaseStudy1(16)
    x, y = wind_farm_model.site.initial_position.T
    simulation_result = wind_farm_model(x, y)
    fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=3))
    npt.assert_array_almost_equal(fm.power_xylk(with_wake_loss=False)[:, :, 0, 0] * 1e-6, 3.35)

    fm = simulation_result.flow_map(grid=Points(
        [-1820., 0., 1820., -1820., 0., 1820., -1820., 0., 1820.],
        [-1730.9229, -1730.9229, -1730.9229, 0., 0., 0., 1730.9229, 1730.9229, 1730.9229],
        [110., 110., 110., 110., 110., 110., 110., 110., 110.]))
    npt.assert_array_almost_equal(fm.power_xylk(with_wake_loss=False)[:, 0, 0] * 1e-6, 3.35)


def test_power_xylk_wt_args():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = WindTurbines.from_WAsP_wtg(wtg_path + "Vestas V112-3.0 MW.wtg", default_mode=None)

    wind_farm_model = PropagateDownwind(site, windTurbines,
                                        wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                        superpositionModel=SquaredSum())
    simulation_result = wind_farm_model(x, y, wd=[0, 270], ws=[6, 8, 10], mode=0)
    fm = simulation_result.flow_map(XYGrid(resolution=3))
    npt.assert_array_almost_equal(fm.power_xylk(mode=1).sum(['wd', 'ws']).isel(h=0),
                                  [[7030000., 6378864., 7029974.],
                                   [7030000., 559767., 4902029.],
                                   [7030000., 7030000., 7029974.]], 0)
    npt.assert_array_almost_equal(fm.power_xylk(mode=8).sum(['wd', 'ws']).isel(h=0),
                                  [[8330000., 7577910., 8329970.],
                                   [8330000., 699980., 5837139.],
                                   [8330000., 8330000., 8329970.]], 0)
    # print(np.round(fm.power_xylk(mode=8).sum(['wd', 'ws']).squeeze()))

    npt.assert_array_almost_equal(fm.aep_xylk(mode=1).sum(['x', 'y']).isel(h=0),
                                  [[9., 22., 43.],
                                   [69., 175., 343.]], 0)

    npt.assert_array_almost_equal(fm.aep_xy(mode=1).isel(h=0),
                                  [[88., 86., 88.],
                                   [88., 6., 40.],
                                   [88., 88., 88.]], 0)


def test_YZGrid_perpendicular():

    wind_farm_model = IEA37CaseStudy1(16)
    x, y = wind_farm_model.site.initial_position.T
    m = x < -1000

    simulation_result = wind_farm_model(x[m], y[m], wd=270)
    fm = simulation_result.flow_map(grid=YZGrid(-1000, z=110, resolution=20))
    if 0:
        simulation_result.flow_map(grid=YZGrid(-1000)).plot_wake_map()
        plt.plot(fm.X[0], fm.Y[0], '.')
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).data.tolist())
        plt.plot(fm.X[0], fm.WS_eff_xylk[:, 0, 0, 0] * 100, label='ws*100')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [9.8, 9.8, 8.42, 5.24, 9.74, 9.8, 9.8, 9.8, 9.76, 7.61, 7.61,
                                   9.76, 9.8, 9.8, 9.8, 9.74, 5.24, 8.42, 9.8, 9.8], 2)


def test_YZGrid_parallel():
    wind_farm_model = IEA37CaseStudy1(16)
    x, y = wind_farm_model.site.initial_position.T
    m = x < -1000
    simulation_result = wind_farm_model(x[m], y[m], wd=0)
    fm = simulation_result.flow_map(grid=YZGrid(-1000, z=110, resolution=20))
    if 0:
        simulation_result.flow_map(grid=YZGrid(-1000)).plot_wake_map()
        plt.plot(fm.X[0], fm.Y[0], '.')
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).data.tolist())
        plt.plot(fm.X[0], fm.WS_eff_xylk[:, 0, 0, 0] * 100, label='ws*100')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [7.32, 7.02, 6.63, 8.86, 8.79, 8.71, 8.63, 8.53, 8.42, 8.3, 8.16,
                                   7.99, 7.81, 7.59, 7.33, 7.0, 6.52, 9.8, 9.8, 9.8], 2)


def test_YZGrid_plot_wake_map_perpendicular():
    wfm = IEA37CaseStudy1(16)
    x, y = wfm.site.initial_position.T

    sim_res = wfm(x, y)
    sim_res.flow_map(grid=YZGrid(x=-100, y=None, resolution=100, extend=.1), wd=270, ws=None).plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')


def test_YZGrid_variables():
    wfm = IEA37CaseStudy1(16)
    x, y = [0], [0]
    sim_res = wfm(x, y)

    fm = sim_res.flow_map(grid=YZGrid(x=100, y=None, resolution=100, extend=.1), wd=270, ws=None)
    fm.WS_eff.plot()
    plt.plot(fm.y[::10], fm.y[::10] * 0 + 110, '.')

    if 0:
        print(np.round(fm.WS_eff.interp(h=110)[::10].squeeze().values, 4))
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(fm.WS_eff.interp(h=110)[::10].squeeze(),
                                  [9.1461, 8.4157, 7.3239, 6.058, 5.022, 4.6455, 5.1019, 6.182, 7.446, 8.506], 4)


def test_YZGrid_plot_wake_map_parallel():
    wfm = IEA37CaseStudy1(16)
    x, y = wfm.site.initial_position.T

    sim_res = wfm(x, y)
    sim_res.flow_map(wd=0, ws=None).plot_wake_map()
    plt.axvline(-450, ls='--')
    plt.figure()
    sim_res.flow_map(grid=YZGrid(x=-450, y=None, resolution=100, extend=.1), wd=0, ws=None).plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')


@pytest.mark.parametrize('wind_direction, ref', [
    ('wd', [5.35, 8.76, 8.42, 6.18, 5.78, 10.87, 5.03, 11.59, 5.31, 10.25, 13.38, 8.63, 8.67,
            12.25, 5.49, 10.92, 9.71, 6.75, 10.69, 7.03]),
    ('WD_i', [5.41, 9.02, 8.42, 6.32, 5.65, 10.8, 5.05, 11.64, 5.31, 10.24, 13.38, 8.57, 8.73,
              12.5, 5.47, 10.66, 10.92, 6.55, 10.33, 8.7])])
def test_YZGrid_terrain_perpendicular(wind_direction, ref):
    site = ParqueFicticioSite(distance=StraightDistance(wind_direction))
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wfm = PropagateDownwind(site, windTurbines,
                            wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                            superpositionModel=SquaredSum())

    simulation_result = wfm(x, y, wd=270, ws=10)
    x = x.max() + 10
    fm = simulation_result.flow_map(grid=YZGrid(x, z=110, resolution=20, extend=0))
    y = fm.X[0]
    x = np.zeros_like(y) + x
    z = site.elevation(x, y)
    simulation_result.flow_map(XYGrid(extend=.005)).plot_wake_map()
    if 0:
        plt.plot(x, y, '.')
        plt.figure()
        simulation_result.flow_map(grid=YZGrid(fm.x.item(), y=fm.y,
                                               z=np.arange(30, 210, 10))).plot_wake_map()
        plt.plot(y, z + 110, '.')
        plt.plot(y, fm.WS_eff_xylk[:, 0, 0, 0] * 100, label="ws*100")
        plt.legend()
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).values.tolist())
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0], ref, 2)


def test_YZGrid_terrain_parallel():
    site = ParqueFicticioSite(distance=StraightDistance('WD_i'))
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wfm = PropagateDownwind(site, windTurbines,
                            wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                            superpositionModel=SquaredSum())

    simulation_result = wfm(x, y, wd=0, ws=10)
    x = 264000
    fm = simulation_result.flow_map(grid=YZGrid(x, z=110, resolution=20, extend=0))
    if 0:
        y = fm.X[0]
        x = np.zeros_like(y) + x
        z = site.elevation(x, y)
        simulation_result.flow_map(XYGrid(extend=0.005)).plot_wake_map()
        plt.plot(x, y, '.')
        plt.figure()
        simulation_result.flow_map(grid=YZGrid(fm.x.item(), fm.y, z=np.arange(30, 210, 10))).plot_wake_map()
        plt.plot(y, z + 110, '.')
        plt.plot(y, fm.WS_eff_xylk[:, 0, 0, 0] * 100, label="ws*100")
        plt.legend()
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).values.tolist())
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [4.79, 4.8, 3.98, 2.99, 5.12, 4.84, 7.91, 7.62, 7.35, 7.46, 7.42, 6.73, 6.65, 5.42,
                                   11.29, 11.72, 12.05, 12.17, 11.03, 9.89], 2)


def test_Points():
    wfm = IEA37CaseStudy1(16)
    x, y = wfm.site.initial_position.T
    sim_res = wfm(x, y)

    flow_map = sim_res.flow_map(Points(x, y, x * 0 + wfm.windTurbines.hub_height()), wd=0, ws=None)
    if 0:
        flow_map.WS_eff.plot()
        plt.show()
    plt.close('all')


def test_not_implemented_plane():
    wfm = IEA37CaseStudy1(16)
    x, y = wfm.site.initial_position.T
    sim_res = wfm(x, y)
    grid = YZGrid(x=-100, y=None, resolution=100, extend=.1)
    grid = grid(x, y, wfm.windTurbines.hub_height(x * 0), wfm.windTurbines.hub_height(x * 0))
    with pytest.raises(NotImplementedError):
        sim_res.flow_map(grid=grid, wd=270, ws=None).plot_wake_map()


def test_FlowBox():
    wfm = IEA37CaseStudy1(16)
    x, y = wfm.site.initial_position.T
    sim_res = wfm(x, y)
    sim_res.flow_box(x=np.arange(0, 100, 10), y=np.arange(0, 100, 10), h=np.arange(0, 100, 10))


def test_min_ws_eff_line():
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T

    yaw_ilk = np.reshape([-30, 30, 0], (3, 1, 1))

    plt.figure(figsize=(14, 3))
    fm = wfm(x, y, yaw=yaw_ilk, tilt=0, wd=270, ws=10).flow_map(
        XYGrid(x=np.arange(-100, 2000, 10), y=np.arange(-500, 500, 10)))
    min_ws_line = fm.min_WS_eff()

    if 0:
        fm.plot_wake_map()
        min_ws_line.plot()
        print(np.round(min_ws_line[::10], 2))
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(min_ws_line[::10],
                                  [np.nan, 0, 11.6, 21.64, 30.42, 38.17, 45.09, 2.6,
                                   -8.65, -18.66, -27.51, -35.37, -42.38, -0.8, -1.09, -1.34,
                                   -1.59, -1.83, -2.07, -2.31, -2.56], 2)


def test_plot_windturbines_with_wd_ws_dependent_yaw():
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T

    yaw_ilk = np.broadcast_to(np.reshape([-30, 30, 0], (3, 1, 1)), (3, 4, 2))

    plt.figure(figsize=(14, 3))
    fm = wfm(x, y, yaw=yaw_ilk, tilt=0, wd=[0, 90, 180, 270], ws=[9, 10]).flow_map(
        XYGrid(x=np.arange(-100, 2000, 10), y=np.arange(-500, 500, 10)))

    fm.plot_windturbines()
    if 0:
        plt.show()
    plt.close('all')


def flow_map_j_wd_chunks():
    # demonstrate that wd chunkification is more efficient than j chunkification
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T

    yaw_ilk = np.reshape([-30, 30, 0], (3, 1, 1))

    sim_res = wfm(x, y, yaw=yaw_ilk, wd=np.arange(320), ws=10)

    t_all = timeit(sim_res.flow_map, verbose=1)(XYGrid(x=np.linspace(-100, 2000, 64), y=np.linspace(-500, 500, 100)))
    t_j = timeit(sim_res.flow_map, verbose=1)(XYGrid(x=np.linspace(-100, 2000, 2), y=np.linspace(-500, 500, 100)))
    t_wd = timeit(sim_res.flow_map, verbose=1)(XYGrid(x=np.linspace(-100, 2000, 64), y=np.linspace(-500, 500, 100)),
                                               wd=np.arange(10))
    print(np.mean(t_all[1]) / np.mean(t_j[1]))
    print(np.mean(t_all[1]) / np.mean(t_wd[1]))


def test_aep_map():
    wfm = IEA37CaseStudy1(16)
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T

    sim_res = wfm(x, y)
    grid = XYGrid(x=np.linspace(-100, 2000, 50), y=np.linspace(-500, 500, 25))
    aep_map = sim_res.aep_map(grid, normalize_probabilities=True)
    fm = sim_res.flow_map(grid)
    npt.assert_array_almost_equal(fm.aep_xy(normalize_probabilities=True).sel(h=110), aep_map)

    grid = Points(x=np.linspace(-100, 2000, 50), y=np.full(50, -500), h=np.full(50, wfm.windTurbines.hub_height()))
    aep_line = sim_res.aep_map(grid, normalize_probabilities=True)
    npt.assert_array_almost_equal(aep_map[0], aep_line)


def test_aep_map_type():
    site = IEA37Site(16)

    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
    v80 = V80()
    v120 = WindTurbine('V80_low_induc', 80, 70, powerCtFunction=PowerCtTabular(
        hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1] * 1.5, 'w', hornsrev1.ct_curve[:, 1]))

    windTurbines = WindTurbines.from_WindTurbine_lst([v80, v120])
    wfm = PropagateDownwind(site, windTurbines,
                            wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                            superpositionModel=SquaredSum())
    sim_res = wfm(x, y)
    grid = XYGrid(x=np.linspace(-100, 2000, 50), y=np.linspace(-500, 500, 25))
    aep_map0 = sim_res.aep_map(grid, normalize_probabilities=True)
    aep_map1 = sim_res.aep_map(grid, type=1, normalize_probabilities=True)
    npt.assert_array_almost_equal(aep_map0 * 1.5, aep_map1)


def test_aep_map_parallel():
    wfm = IEA37CaseStudy1(16)
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T

    sim_res = wfm(x, y)
    grid = XYGrid(x=np.linspace(-100, 2000, 50), y=np.linspace(-500, 500, 25))
    aep_map = sim_res.aep_map(grid, normalize_probabilities=True, n_cpu=2)

    fm = sim_res.flow_map(grid)
    npt.assert_array_almost_equal(fm.aep_xy(normalize_probabilities=True).sel(h=110), aep_map)

    fm = sim_res.flow_map(grid, wd=[0])
    aep_map = sim_res.aep_map(grid, normalize_probabilities=True, wd=[0], n_cpu=2)
    npt.assert_array_almost_equal(fm.aep_xy(normalize_probabilities=True).sel(h=110), aep_map)


def test_aep_map_smartstart_griddedsite_terrainfollowingdistance():
    site = ParqueFicticioSite()
    x, y = site.initial_position[:3].T
    windTurbines = IEA37_WindTurbines()

    wfm = PropagateDownwind(site, windTurbines,
                            wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                            superpositionModel=SquaredSum())

    for i in range(3):
        sim_res = wfm(x[:i], y[:i], wd=[0, 1])
        grid = XYGrid(x=site.ds.x, y=site.ds.y)
        sim_res.aep_map(grid, normalize_probabilities=True)


def test_wd_dependent_flow_map():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0], y=[0], wd=[0, 90, 180])
    for wd in [[0], [0, 90], None]:
        fm = sim_res.flow_map(wd=wd)
        fm.plot_wake_map()
        if 0:
            plt.show()
        plt.close('all')


def test_ws_dependent_flow_map():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0], y=[0], ws=[8, 9, 10], wd=270)
    for ws in [[8], [8, 9], None]:
        fm = sim_res.flow_map(ws=ws)
        fm.plot_wake_map()
        if 0:
            plt.show()
        plt.close('all')


def test_time_dependent_flow_map():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0], y=[0], wd=[0, 90, 180], ws=[8, 9, 10], time=True)
    for t in [[0], [0, 1], None]:
        fm = sim_res.flow_map(time=t)
        fm.plot_wake_map()
        if 0:
            plt.show()
        plt.close('all')


def test_i_dependent_flow_map():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0], y=[0], wd=[0, 90, 180])
    X, Y = np.meshgrid(np.linspace(-2000, 2000, 50), np.linspace(-2000, 2000, 50))
    fm = sim_res.flow_map(Points(x=X.flatten(), y=Y.flatten(), h=X.flatten() * 0 + 110))
    with pytest.raises(NotImplementedError, match="Plot not supported for FlowMaps based on Points. Use XYGrid, YZGrid or XZGrid instead"):
        fm.plot_wake_map()


def test_plot_windturbines_with_tilt_and_yaw():
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    sim_res = wfm(x=[0], y=[0], wd=[270], yaw=[20], tilt=10)
    sim_res.flow_map(XZGrid(y=0)).plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')


def test_WS_WD_TI():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0], y=[0], wd=[270], WS=8, WD=260, TI=0.01)
    fm = sim_res.flow_map()
    npt.assert_array_equal(fm.WS, 8)
    npt.assert_array_equal(fm.WD, 260)
    npt.assert_array_equal(fm.TI, 0.01)


def test_wt_dependent_WS():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[0, 500], y=[0, 0], wd=[270], WS=[8, 9])
    s = 'The WT dependent WS that was provided for the simulation is not available at the flow map points and therefore ignored'
    with pytest.raises(UserWarning, match=s):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            sim_res.flow_map()
