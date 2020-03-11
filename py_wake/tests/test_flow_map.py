from py_wake.flow_map import HorizontalGrid
from py_wake.tests import npt


def test_power_xylk():
    from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
    from py_wake import IEA37SimpleBastankhahGaussian

    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    # NOJ wake model
    wind_farm_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    simulation_result = wind_farm_model(x, y)
    fm = simulation_result.flow_map(grid=HorizontalGrid(resolution=3))
    npt.assert_array_almost_equal(fm.power_xylk(with_wake_loss=False)[:, :, 0, 0] * 1e-6, 3.35)
