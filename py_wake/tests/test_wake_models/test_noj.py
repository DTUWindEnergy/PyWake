import pytest
import numpy as np

from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windfarm
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.wake_models.noj import NOJ
from py_wake.wind_turbines import WindTurbines
from py_wake.aep_calculator import AEPCalculator


# Two turbines, 0: Nibe-A, 1:Ct=0
NibeA0 = WindTurbines(names=['Nibe-A'] * 2, diameters=[40] * 2,
                      hub_heights=[50] * 2,
                      ct_funcs=[lambda _: 8 / 9, lambda _: 0],
                      power_funcs=[lambda _: 0] * 2, power_unit='w')


def test_NOJ_Nibe_result():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, NibeA0)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    WS_eff_ilk = wake_model.calc_wake(x_i, y_i, h_i, [0, 1, 1], 0.0, 8.1)[0]
    npt.assert_array_almost_equal(WS_eff_ilk[:, 0, 0], [8.1, 4.35, 5.7])


def test_NOJ_Nibe_result_wake_map():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).
    def ct_func(_):
        return 8 / 9

    def power_func(*_):
        return 0
    windTurbines = NibeA0
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, windTurbines)
    aep = AEPCalculator(wake_model)
    _, _, WS_eff_yx = aep.wake_map(x_j=[0], y_j=[0, -40, -100], height_level=50, wt_x=[0], wt_y=[0], wd=[0], ws=[8.1])
    npt.assert_array_almost_equal(WS_eff_yx[:, 0], [8.1, 4.35, 5.7])


@pytest.mark.parametrize('wdir,x,y', [(0, [0, 0, 0], [0, -40, -100]),
                                      (90, [0, -40, -100], [0, 0, 0]),
                                      (180, [0, 0, 0], [0, 40, 100]),
                                      (270, [0, 40, 100], [0, 0, 0])])
def test_NOJ_two_turbines_in_row(wdir, x, y):
    # Two turbines in a row, North-South
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    windTurbines = NibeA0
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, windTurbines)
    h_i = [50, 50, 50]
    WS_eff_ilk = wake_model.calc_wake(x, y, h_i, [0, 0, 0], wdir, 8.1)[0]
    ws_wt3 = 8.1 - np.hypot(8.1 * 2 / 3 * (20 / 26)**2, 8.1 * 2 / 3 * (20 / 30)**2)
    npt.assert_array_almost_equal(WS_eff_ilk[:, 0, 0], [8.1, 4.35, ws_wt3])


def test_NOJ_6_turbines_in_row():
    n_wt = 6
    x = [0] * n_wt
    y = - np.arange(n_wt) * 40 * 2

    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, NibeA0)
    WS_eff_ilk = wake_model.calc_wake(x, y, [50] * n_wt, [0.0] * n_wt, 0.0, 11.0)[0]
    np.testing.assert_array_almost_equal(
        WS_eff_ilk[1:, 0, 0], 11 - np.sqrt(np.cumsum(((11 * 2 / 3 * 20**2)**2) / (20 + 8 * np.arange(1, 6))**4)))


def test_wake_map():
    site = IEA37Site(16)

    x, y = site.initial_position.T

    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = NOJ(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])

    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)

        plt.show()

    ref = [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]
    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


if __name__ == '__main__':
    test_wake_map()
