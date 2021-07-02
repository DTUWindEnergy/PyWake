from py_wake.flow_map import HorizontalGrid, YZGrid, Grid, Points, XYGrid
from py_wake.tests import npt
import matplotlib.pyplot as plt
import numpy as np
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.site.distance import StraightDistance
from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
from py_wake import IEA37SimpleBastankhahGaussian
import pytest
from py_wake.deflection_models.jimenez import JimenezWakeDeflection


@pytest.fixture(autouse=True)
def close_plots():
    yield
    try:
        plt.close('all')
    except Exception:
        pass


def test_power_xylk():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    # NOJ wake model
    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y)
    fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=3))
    npt.assert_array_almost_equal(fm.power_xylk(with_wake_loss=False)[:, :, 0, 0] * 1e-6, 3.35)


def test_YZGrid_perpendicular():

    site = IEA37Site(16)
    x, y = site.initial_position.T
    m = x < -1000
    windTurbines = IEA37_WindTurbines()

    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
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
    site = IEA37Site(16)
    x, y = site.initial_position.T
    m = x < -1000
    windTurbines = IEA37_WindTurbines()

    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
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
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)
    sim_res.flow_map(grid=YZGrid(x=-100, y=None, resolution=100, extend=.1), wd=270, ws=None).plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')


def test_YZGrid_variables():
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)

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
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)
    sim_res.flow_map(wd=0, ws=None).plot_wake_map()
    plt.axvline(-450, ls='--')
    plt.figure()
    sim_res.flow_map(grid=YZGrid(x=-450, y=None, resolution=100, extend=.1), wd=0, ws=None).plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')


def test_YZGrid_terrain_perpendicular():
    site = ParqueFicticioSite(distance=StraightDistance())
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y, wd=270, ws=10)
    x = x.max() + 10
    fm = simulation_result.flow_map(grid=YZGrid(x, z=110, resolution=20, extend=0))
    y = fm.X[0]
    x = np.zeros_like(y) + x
    z = site.elevation(x, y)
    simulation_result.flow_map(XYGrid(extend=.005)).plot_wake_map()
    plt.plot(x, y, '.')
    plt.figure()
    simulation_result.flow_map(grid=YZGrid(fm.x.item(), y=fm.y,
                                           z=np.arange(30, 210, 10))).plot_wake_map()
    plt.plot(y, z + 110, '.')
    plt.plot(y, fm.WS_eff_xylk[:, 0, 0, 0] * 100, label="ws*100")
    plt.legend()
    if 0:
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).values.tolist())
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [5.39, 8.48, 8.42, 6.42, 5.55, 11.02, 4.99, 11.47, 5.32, 10.22, 13.39, 8.79,
                                   8.51, 12.4, 5.47, 10.78, 10.12, 6.54, 10.91, 7.18], 2)


def test_YZGrid_terrain_parallel():
    site = ParqueFicticioSite(distance=StraightDistance())
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y, wd=0, ws=10)
    x = 264000
    fm = simulation_result.flow_map(grid=YZGrid(x, z=110, resolution=20, extend=0))
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
    if 0:
        print(np.round(fm.WS_eff_xylk[:, 0, 0, 0], 2).values.tolist())
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [4.55, 3.89, 3.15, 2.31, 4.41, 4.3, 7.41, 7.2, 7.03, 7.23, 7.32, 6.77, 6.83,
                                   5.58, 11.01, 11.51, 11.93, 12.17, 11.03, 9.89], 2)


def test_Points():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)

    flow_map = sim_res.flow_map(Points(x, y, x * 0 + windTurbines.hub_height()), wd=0, ws=None)
    if 0:
        flow_map.WS_eff.plot()
        plt.show()
    plt.close('all')


def test_not_implemented_plane():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)
    grid = YZGrid(x=-100, y=None, resolution=100, extend=.1)
    grid = grid(x, y, windTurbines.hub_height(x * 0), windTurbines.hub_height(x * 0))
    with pytest.raises(NotImplementedError):
        sim_res.flow_map(grid=grid, wd=270, ws=None).plot_wake_map()


def test_FlowBox():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)
    sim_res.flow_box(x=np.arange(0, 100, 10), y=np.arange(0, 100, 10), h=np.arange(0, 100, 10))


def test_min_ws_eff_line():

    site = IEA37Site(16)
    x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
    windTurbines = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, deflectionModel=JimenezWakeDeflection())

    yaw_ilk = np.reshape([-30, 30, 0], (3, 1, 1))

    plt.figure(figsize=(14, 3))
    fm = wfm(x, y, yaw=yaw_ilk, wd=270, ws=10).flow_map(
        XYGrid(x=np.arange(-100, 2000, 10), y=np.arange(-500, 500, 10)))
    min_ws_line = fm.min_WS_eff()

    if 0:
        fm.plot_wake_map()
        min_ws_line.plot()
        print(np.round(min_ws_line[::10], 2))
        plt.show()
    npt.assert_array_almost_equal(min_ws_line[::10],
                                  [np.nan, np.nan, 11.6, 21.64, 30.42, 38.17, 45.09, 51.27,
                                   -8.65, -18.66, -27.51, -35.37, -42.38, -48.58, -1.09, -1.34,
                                   -1.59, -1.83, -2.07, -2.31, -2.56], 2)
