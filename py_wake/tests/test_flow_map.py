from py_wake.flow_map import HorizontalGrid, YZGrid
from py_wake.tests import npt
import matplotlib.pyplot as plt
import numpy as np
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.site.distance import StraightDistance
from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
from py_wake import IEA37SimpleBastankhahGaussian
import pytest


def test_power_xylk():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    # NOJ wake model
    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y)
    fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=3))
    npt.assert_array_almost_equal(fm.power_xylk(with_wake_loss=False)[:, :, 0, 0] * 1e-6, 3.35)


def test_YZGrid():
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
        print(np.round(fm.WS_eff_xylk[0, :, 0, 0], 2).data.tolist())
        plt.show()
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [9.8, 9.8, 8.42, 5.24, 9.74, 9.8, 9.8, 9.8, 9.76, 7.61, 7.61,
                                   9.76, 9.8, 9.8, 9.8, 9.74, 5.24, 8.42, 9.8, 9.8], 2)


def test_YZGrid_plot_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    sim_res = wf_model(x, y)
    sim_res.flow_map(grid=YZGrid(x=-100, y=None, resolution=100, extend=.1), wd=270, ws=None).plot_wake_map()
    if 0:
        plt.show()
    plt.close()


def test_YZGrid_terrain():
    site = ParqueFicticioSite(distance=StraightDistance())
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y, wd=270, ws=10)
    x = x.max() + 10
    fm = simulation_result.flow_map(grid=YZGrid(x, z=110, resolution=20))
    y = fm.X[0]
    x = np.zeros_like(y) + x
    z = site.elevation(x, y)
    simulation_result.flow_map().plot_wake_map()
    plt.plot(x, y, '.')
    plt.figure()
    simulation_result.flow_map(grid=YZGrid(x.max() + 10)).plot_wake_map()
    plt.plot(y, z + 110, '.')
    if 0:
        print(np.round(fm.WS_eff_xylk[0, :, 0, 0], 2).tolist())
        plt.show()
    plt.close()
    npt.assert_array_almost_equal(fm.WS_eff_xylk[:, 0, 0, 0],
                                  [10.0, 10.0, 11.69, 5.8, 8.7, 5.15, 10.63, 4.99, 10.73, 8.16, 13.24, 5.59,
                                   12.4, 7.02, 10.91, 7.21, 8.77, 8.41, 9.99, 10.0], 2)


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
    flow_box = sim_res.flow_box(x=np.arange(0, 100, 10), y=np.arange(0, 100, 10), h=np.arange(0, 100, 10))
